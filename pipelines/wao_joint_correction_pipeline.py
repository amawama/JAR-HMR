from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/root/data1/experiments/Human3R/wao_hmr")
METHOD_DIR = ROOT / "method"
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from wao_joint_correction import (
    JointCorrectionRegressor,
    build_feature,
    cap_delta,
    load_checkpoint,
    load_external_observations,
    rescale_detection_item,
    match_frame_rows_to_detections,
    coco_to_j14,
    bbox_from_points,
    bbox_iou,
    center_distance,
)

H3R = Path("/root/data1/models/Human3R")
TRAIN_ROOT = ROOT / "training" / "t1_jcr"
SCRIPTS = ROOT / "full_3dpw" / "scripts"
LEGACY_RUNNER = SCRIPTS / "run_wao_hmr_unified_with_progress.py"


def read_stream(path):
    rows = []
    with Path(path).open("rb") as f:
        while True:
            try:
                rows.append(pickle.load(f))
            except EOFError:
                break
    return rows


def inspect_stream(path):
    rows = read_stream(path)
    seq = defaultdict(int)
    joint_counts = defaultdict(int)
    n2_counts = defaultdict(int)
    errors = []

    for row in rows:
        seq[row["sequence"]] += 1
        raw = np.asarray(row["raw_j3d_rel"], np.float32)
        gt = np.asarray(row["gt_j3d_rel"], np.float32)
        joint_counts[int(raw.shape[0])] += 1
        n2_counts[int(row.get("n_j2d_common", len(row["valid2d"])))] += 1
        errors.append(float(np.linalg.norm(raw - gt, axis=-1).mean() * 1000.0))

    result = {
        "rows": len(rows),
        "sequences": len(seq),
        "rows_per_sequence": dict(seq),
        "joint_counts": dict(joint_counts),
        "common_2d_joint_counts": dict(n2_counts),
        "mean_raw_pelvis_relative_mpjpe_mm": (
            float(np.mean(errors)) if errors else None
        ),
    }
    print("===== JOINT-CORRECTION DUMP SUMMARY =====")
    print(json.dumps(result, indent=2))
    return result


class TrainingDataset(Dataset):
    def __init__(self, path, augment, seed):
        self.rows = read_stream(path)
        if not self.rows:
            raise RuntimeError(f"empty dataset: {path}")
        self.augment = augment
        self.seed = int(seed)
        self.joint_count = int(np.asarray(self.rows[0]["raw_j3d_rel"]).shape[0])
        for r in self.rows:
            if np.asarray(r["raw_j3d_rel"]).shape != (self.joint_count, 3):
                raise RuntimeError("mixed joint protocol in one stream")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        raw3 = np.asarray(r["raw_j3d_rel"], np.float32).copy()
        gt3 = np.asarray(r["gt_j3d_rel"], np.float32).copy()
        raw2 = np.asarray(r["raw_j2d"], np.float32).copy()
        obs2 = np.asarray(r["gt_j2d"], np.float32).copy()
        mask = np.asarray(r["valid2d"], np.float32).copy()

        rng = np.random.default_rng(None if self.augment else self.seed + idx)

        if self.augment:
            mask[rng.random(self.joint_count) < 0.18] = 0.0
            if self.joint_count >= 4 and rng.random() < 0.35:
                length = max(2, self.joint_count // 5)
                start = int(rng.integers(0, max(1, self.joint_count-length+1)))
                mask[start:start+length] = 0.0
            noise_std = 0.015
        else:
            mask[rng.random(self.joint_count) < 0.22] = 0.0
            if self.joint_count >= 4 and rng.random() < 0.30:
                length = max(2, self.joint_count // 5)
                start = int(rng.integers(0, max(1, self.joint_count-length+1)))
                mask[start:start+length] = 0.0
            noise_std = 0.010

        # Build same normalized residual as successful experiment, then add noise.
        x = build_feature(raw3, raw2, obs2, mask)
        d3 = self.joint_count * 3
        d2 = self.joint_count * 2
        resid = x[d3:d3+d2].reshape(self.joint_count, 2)
        resid += rng.normal(0.0, noise_std, resid.shape).astype(np.float32)
        resid[mask < 0.5] = 0.0
        x[d3:d3+d2] = resid.reshape(-1)

        target = (gt3 - raw3).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(target),
            torch.from_numpy(raw3),
            torch.from_numpy(gt3),
            str(r["sequence"]),
        )


def mpjpe(a, b):
    return torch.linalg.vector_norm(a-b, dim=-1).mean(dim=-1)


@torch.no_grad()
def evaluate(model, loader, device, alpha):
    model.eval()
    gains, raw_errors, corr_errors, seqs = [], [], [], []

    for x, target, raw3, gt3, sequence in loader:
        x = x.to(device)
        raw3 = raw3.to(device)
        gt3 = gt3.to(device)

        delta = cap_delta(model(x), 0.06)
        corrected = raw3 + float(alpha) * delta

        raw_e = mpjpe(raw3, gt3) * 1000.0
        corr_e = mpjpe(corrected, gt3) * 1000.0
        gain = raw_e - corr_e

        gains.extend(gain.cpu().tolist())
        raw_errors.extend(raw_e.cpu().tolist())
        corr_errors.extend(corr_e.cpu().tolist())
        seqs.extend(list(sequence))

    gains = np.asarray(gains, np.float64)
    raw_errors = np.asarray(raw_errors, np.float64)
    corr_errors = np.asarray(corr_errors, np.float64)
    seqs = np.asarray(seqs, dtype=object)

    result = {
        "n": int(len(gains)),
        "alpha": float(alpha),
        "raw_mpjpe": float(raw_errors.mean()),
        "corrected_mpjpe": float(corr_errors.mean()),
        "mean_gain_mpjpe": float(gains.mean()),
        "median_gain_mpjpe": float(np.median(gains)),
        "improved_ratio_mpjpe": float((gains > 0).mean()),
        "harm_gt_1mm_ratio": float((gains < -1.0).mean()),
    }
    per_seq = {}
    for s in sorted(set(seqs.tolist())):
        idx = seqs == s
        g = gains[idx]
        per_seq[s] = {
            "n": int(idx.sum()),
            "mean_gain_mpjpe": float(g.mean()),
            "improved_ratio_mpjpe": float((g > 0).mean()),
            "harm_gt_1mm_ratio": float((g < -1.0).mean()),
        }
    result["positive_sequence_count"] = int(
        sum(v["mean_gain_mpjpe"] > 0 for v in per_seq.values())
    )
    result["sequence_count"] = int(len(per_seq))
    return result, per_seq


def cmd_train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tr = TrainingDataset(args.train, True, args.seed)
    va = TrainingDataset(args.validation, False, args.seed + 999)
    if tr.joint_count != va.joint_count:
        raise RuntimeError("train/validation joint_count mismatch")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dl = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    vl = DataLoader(
        va, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointCorrectionRegressor(tr.joint_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.015)

    best = None
    history = []

    print("===== MAGNITUDE-CALIBRATED JOINT CORRECTION =====")
    print("joint_count:", tr.joint_count)
    print("train_rows:", len(tr))
    print("validation_rows:", len(va))
    print("eval_alpha:", args.alpha)
    print("human3r_backbone_trained: False")
    print("uses_test_gt: False")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, target, raw3, gt3, sequence in dl:
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        calibrated, per_seq = evaluate(model, vl, device, args.alpha)
        full_scale, _ = evaluate(model, vl, device, 1.0)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "calibrated": calibrated,
            "full_scale_diagnostic": full_scale,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        score = (
            calibrated["mean_gain_mpjpe"]
            + 0.5 * calibrated["improved_ratio_mpjpe"]
            - 2.0 * calibrated["harm_gt_1mm_ratio"]
        )
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "epoch": epoch,
                "metrics": calibrated,
                "per_sequence": per_seq,
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "joint_count": tr.joint_count,
                    "epoch": epoch,
                    "eval_alpha": float(args.alpha),
                    "metrics": calibrated,
                },
                out / "best.pt",
            )

    m = best["metrics"]
    passed = (
        m["mean_gain_mpjpe"] >= 1.0
        and m["improved_ratio_mpjpe"] >= 0.60
        and m["harm_gt_1mm_ratio"] <= 0.20
        and m["positive_sequence_count"] == m["sequence_count"]
    )
    summary = {
        "experiment": "Multi-Sequence Magnitude-Calibrated Joint Correction Regressor",
        "joint_count": tr.joint_count,
        "train_rows": len(tr),
        "validation_rows": len(va),
        "eval_alpha": float(args.alpha),
        "best_epoch": best["epoch"],
        "best_metrics": m,
        "best_per_sequence": best["per_sequence"],
        "decision": (
            "PROCEED_TO_EXTERNAL_KEYPOINT_TRANSFER"
            if passed else "DO_NOT_ADVANCE"
        ),
        "uses_test_gt": False,
        "human3r_backbone_trained": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("===== MAGNITUDE-CALIBRATED SUMMARY =====")
    print(json.dumps(summary, indent=2))


def cmd_dump(args):
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dump_path = out_root / f"t1_joint_dump_{args.split}.pklstream"
    if args.reset and dump_path.exists():
        dump_path.unlink()

    env = os.environ.copy()
    env.update({
        "WAO_HMR_STAGE": "t1_joint_dump",
        "WAO_3DPW_SPLIT": args.split,
        "WAO_T1_MAX_SEQUENCES": str(args.max_sequences),
        "WAO_T1_OUT_ROOT": str(out_root),
        "HUMAN3R_NO_VIEWER": "1",
        "TORCH_HOME": str(H3R / "cache/torch"),
        "TORCH_EXTENSIONS_DIR": str(H3R / "cache/torch_extensions"),
        "PYTHONPATH": "src:.",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "LD_LIBRARY_PATH": (
            "/root/data1/cuda-12.1/lib64:"
            + env.get("LD_LIBRARY_PATH", "")
        ),
    })

    cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "--num_machines", "1",
        "--mixed_precision", "no",
        "--dynamo_backend", "no",
        "--main_process_port", str(args.port),
        "eval/global_human/launch.py",
        "--weights", "./src/human3r_896L.pth",
        "--output_dir", f"./eval_results/global_human/{args.run_name}",
        "--eval_dataset", "3dpw",
        "--size", "512",
        "--reset_interval", "100",
        "--use_ttt3r",
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=H3R, env=env, check=True)




def cmd_export_manifest(args):
    rows = read_stream(args.dump)
    if not rows:
        raise RuntimeError("empty dump")

    required = defaultdict(set)
    for row in rows:
        required[str(row["sequence"])].add(int(row["frame_id"]))

    payload = {
        "format_version": 1,
        "source_dump": str(args.dump),
        "uses_gt_coordinates": False,
        "purpose": (
            "sequence/frame manifest for external observation generation"
        ),
        "sequences": {
            sequence: sorted(int(x) for x in frame_ids)
            for sequence, frame_ids in sorted(required.items())
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("===== EXTERNAL OBSERVATION MANIFEST =====")
    print("sequences:", len(payload["sequences"]))
    print("frames:", sum(len(v) for v in payload["sequences"].values()))
    print("uses_gt_coordinates: False")
    print("output:", out)


def cmd_prepare_vitpose_obs(args):
    """
    Generate a real external 2D-keypoint cache for exactly the sequence/frame
    set referenced by a joint-correction dump.

    This command intentionally uses the frozen WHAM preprocessing detector
    (YOLO person boxes + ViTPose COCO-17 keypoints).  It does not use GT for
    detections, keypoints, person matching, or model input.
    """
    import cv2

    wham_root = Path(args.wham_root)
    if not wham_root.exists():
        raise FileNotFoundError(wham_root)

    if str(wham_root) not in sys.path:
        sys.path.insert(0, str(wham_root))

    try:
        from lib.models.preproc.detector import DetectionModel
    except Exception as exc:
        raise RuntimeError(
            "Could not import WHAM DetectionModel. Run this subcommand with "
            "the Python/conda environment that has WHAM's ultralytics+mmpose "
            "dependencies installed."
        ) from exc

    required = defaultdict(set)

    if getattr(args, "manifest", None):
        manifest_path = Path(args.manifest)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        sequences = payload.get("sequences", {})
        if not isinstance(sequences, dict) or not sequences:
            raise RuntimeError(f"invalid/empty manifest: {manifest_path}")

        for sequence, frame_ids in sequences.items():
            for frame_id in frame_ids:
                required[str(sequence)].add(int(frame_id))
        input_source = f"manifest:{manifest_path}"
    else:
        rows = read_stream(args.dump)
        if not rows:
            raise RuntimeError("empty dump")
        for row in rows:
            required[str(row["sequence"])].add(int(row["frame_id"]))
        input_source = f"dump:{args.dump}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.reset:
        with out_path.open("rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            raise RuntimeError("existing observation cache is not a dict")
        sequences_out = cache.setdefault("sequences", {})
    else:
        cache = {
            "format_version": 1,
            "source": "WHAM DetectionModel (YOLO person boxes + ViTPose COCO-17)",
            "coordinate_space": "original_image",
            "keypoint_format": "COCO-17",
            "uses_gt_for_detection": False,
            "uses_gt_for_keypoints": False,
            "sequences": {},
        }
        sequences_out = cache["sequences"]

    image_root = Path(args.image_root)
    detector = DetectionModel(args.device.lower())

    total_required = sum(len(v) for v in required.values())
    processed = 0
    total_detections = 0

    print("===== PREPARE EXTERNAL VITPOSE OBSERVATIONS =====")
    print("required_sequences:", sorted(required))
    print("required_frames:", total_required)
    print("input_source:", input_source)
    print("output:", out_path)
    print("uses_gt_for_detection: False")
    print("uses_gt_for_keypoints: False")
    print("source: WHAM DetectionModel")

    for sequence in sorted(required):
        frame_ids = sorted(required[sequence])
        seq_dir = image_root / sequence
        files = sorted(
            list(seq_dir.glob("*.jpg"))
            + list(seq_dir.glob("*.jpeg"))
            + list(seq_dir.glob("*.png"))
        )
        if not files:
            raise FileNotFoundError(f"no images for {sequence}: {seq_dir}")

        detector.initialize_tracking()
        seq_out = sequences_out.setdefault(sequence, {})

        for frame_id in frame_ids:
            if frame_id in seq_out and not args.reset:
                processed += 1
                continue

            if frame_id < 0 or frame_id >= len(files):
                raise IndexError(
                    f"{sequence} frame_id={frame_id} outside image list "
                    f"[0,{len(files)-1}]"
                )

            image_path = files[frame_id]
            img = cv2.imread(str(image_path))
            if img is None:
                raise RuntimeError(f"failed to read image: {image_path}")

            height, width = img.shape[:2]
            before = len(detector.tracking_results["keypoints"])

            # Tracking IDs are not used by the later Human3R-person matcher;
            # detector.track is used only as WHAM's standard detector+pose frontend.
            detector.track(img, float(args.fps), len(files))

            after = len(detector.tracking_results["keypoints"])
            frame_items = []

            for idx in range(before, after):
                kp = np.asarray(
                    detector.tracking_results["keypoints"][idx],
                    np.float32,
                )
                if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
                    continue

                conf = kp[:17, 2]
                valid = conf >= float(args.kp_conf_thr)
                if int(valid.sum()) < int(args.min_valid_joints):
                    continue

                valid_conf = conf[valid]
                # Cache score is a pose-confidence summary, not GT information.
                score = float(valid_conf.mean()) if len(valid_conf) else 0.0

                frame_items.append(
                    {
                        # Store plain Python objects so the cache is portable
                        # between the WHAM and Human3R NumPy versions.
                        "kp2d": kp[:17].astype(np.float32).tolist(),
                        "res": [int(width), int(height)],
                        "score": float(score),
                        "source": "wham_yolov8x_vitpose",
                        "image_path": str(image_path),
                    }
                )

            seq_out[int(frame_id)] = frame_items
            total_detections += len(frame_items)
            processed += 1

            if processed == 1 or processed % 100 == 0:
                print(
                    "[prepare-vitpose-obs]",
                    f"{processed}/{total_required}",
                    "sequence=", sequence,
                    "frame=", frame_id,
                    "detections=", len(frame_items),
                    flush=True,
                )

        # Save after each sequence to survive interruption.
        with out_path.open("wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    with out_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("===== EXTERNAL OBSERVATION CACHE COMPLETE =====")
    print("processed_frames:", processed)
    print("total_detections:", total_detections)
    print("output:", out_path)

def cmd_check_obs(args):
    rows = read_stream(args.dump)
    required = sorted(set(str(r["sequence"]) for r in rows))
    obs = load_external_observations(args.obs)
    available = sorted(obs.keys())
    print("===== EXTERNAL OBS COVERAGE =====")
    print("required_sequences:", required)
    print("observation_sequences:", len(available))
    missing = [s for s in required if s not in obs]
    print("missing_sequences:", missing)
    frame_req = defaultdict(set)
    for r in rows:
        frame_req[str(r["sequence"])].add(int(r["frame_id"]))
    for s in required:
        req = frame_req[s]
        have = set(int(k) for k in obs.get(s, {}).keys())
        print(
            s,
            "required_frames=", len(req),
            "covered_frames=", len(req & have),
            "coverage=", (len(req & have)/len(req) if req else 0.0),
        )
    if missing:
        raise SystemExit(2)


_H3R_GEOMETRY_CACHE = {}


def _image_path(image_root, sequence, frame_id):
    seq_dir = Path(image_root) / sequence
    files = sorted(
        list(seq_dir.glob("*.jpg"))
        + list(seq_dir.glob("*.png"))
        + list(seq_dir.glob("*.jpeg"))
    )
    if frame_id < 0 or frame_id >= len(files):
        return None
    return files[frame_id]


def _human3r_image_geometry(image_path, h3r_size=512):
    """
    Return:
      ori_width, ori_height, h3r_width, h3r_height

    Human3R official evaluation is run with --size 512 and default
    no_crop=True, so its projected 2D joints live in the resized
    `true_shape` coordinate system.  We ask Human3R's own load_images()
    for that exact shape rather than reimplementing resize arithmetic.
    """
    from PIL import Image

    image_path = Path(image_path)
    with Image.open(image_path) as im:
        ori_width, ori_height = im.size

    key = (int(ori_width), int(ori_height), int(h3r_size))
    cached = _H3R_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached

    h3r_src = H3R / "src"
    if str(h3r_src) not in sys.path:
        sys.path.insert(0, str(h3r_src))

    from dust3r.utils.image import load_images_for_eval as load_images

    loaded = load_images(
        [str(image_path)],
        size=int(h3r_size),
        verbose=False,
        crop=False,
    )
    if not loaded:
        raise RuntimeError(f"Human3R load_images returned nothing: {image_path}")

    true_shape = np.asarray(loaded[0]["true_shape"]).reshape(-1, 2)[0]
    h3r_height = int(true_shape[0])
    h3r_width = int(true_shape[1])

    value = (
        int(ori_width),
        int(ori_height),
        h3r_width,
        h3r_height,
    )
    _H3R_GEOMETRY_CACHE[key] = value

    print(
        "[EXTERNAL-COORD]",
        f"original={ori_width}x{ori_height}",
        f"human3r={h3r_width}x{h3r_height}",
        f"scale=({h3r_width/max(ori_width,1):.6f},"
        f"{h3r_height/max(ori_height,1):.6f})",
        flush=True,
    )
    return value


def _frame_external_match_data(
    *,
    frame_rows,
    frame_obs,
    image_path,
    h3r_size,
    det_score_thr,
    iou_thr,
    center_thr,
):
    ori_w, ori_h, h3r_w, h3r_h = _human3r_image_geometry(
        image_path,
        h3r_size=h3r_size,
    )

    detections = []
    for item in frame_obs:
        # Cache keypoints are in original-image coordinates and carry
        # res=[original_width, original_height].  rescale_detection_item()
        # therefore maps them exactly into Human3R's resized true_shape.
        det = rescale_detection_item(
            item,
            h3r_w,
            h3r_h,
            det_score_thr=det_score_thr,
        )
        if det is not None:
            detections.append(det)

    matches = match_frame_rows_to_detections(
        frame_rows,
        detections,
        h3r_w,
        h3r_h,
        iou_thr=iou_thr,
        center_thr=center_thr,
    )

    return detections, matches, {
        "ori_width": ori_w,
        "ori_height": ori_h,
        "h3r_width": h3r_w,
        "h3r_height": h3r_h,
    }


def _hungarian_match_diagnostic(frame_rows, detections, width, height):
    """
    Ungated Hungarian assignments for diagnostics only.
    Uses only Human3R predicted raw_j2d and external detections.
    No GT joint coordinates or GT person IDs are accessed.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:
        raise RuntimeError("scipy is required for match diagnostics") from exc

    raw_boxes = []
    row_ids = []
    for idx, row in enumerate(frame_rows):
        box = bbox_from_points(
            np.asarray(row["raw_j2d"], np.float32)
        )
        if box is not None:
            raw_boxes.append(box)
            row_ids.append(idx)

    if not raw_boxes or not detections:
        return []

    cost = np.full(
        (len(raw_boxes), len(detections)),
        100.0,
        np.float32,
    )
    stats = {}

    for r, raw_box in enumerate(raw_boxes):
        for c, det in enumerate(detections):
            iou = bbox_iou(raw_box, det["bbox"])
            center = center_distance(
                raw_box,
                det["bbox"],
                width,
                height,
            )
            cost[r, c] = (1.0 - iou) + 2.5 * center
            stats[(r, c)] = (float(iou), float(center))

    rr, cc = linear_sum_assignment(cost)
    out = []
    for r, c in zip(rr, cc):
        iou, center = stats[(r, c)]
        out.append(
            {
                "row_index": int(row_ids[r]),
                "detection_id": int(c),
                "iou": iou,
                "center": center,
            }
        )
    return out


def cmd_check_match(args):
    rows = read_stream(args.dump)
    if not rows:
        raise RuntimeError("empty dump")

    obs = load_external_observations(args.obs)
    grouped = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[
            (str(row["sequence"]), int(row["frame_id"]))
        ].append((idx, row))

    total_rows = len(rows)
    accepted_rows = 0
    assigned_ious = []
    assigned_centers = []
    accepted_ious = []
    accepted_centers = []
    per_seq = defaultdict(
        lambda: {
            "rows": 0,
            "accepted": 0,
            "assigned_iou": [],
            "assigned_center": [],
        }
    )
    geometry_counts = defaultdict(int)
    frame_with_detections = 0

    for (sequence, frame_id), items in grouped.items():
        image_path = _image_path(
            args.image_root,
            sequence,
            frame_id,
        )
        if image_path is None:
            continue

        frame_obs = obs.get(sequence, {}).get(frame_id, [])
        frame_rows = [row for _, row in items]

        detections, matches, geom = _frame_external_match_data(
            frame_rows=frame_rows,
            frame_obs=frame_obs,
            image_path=image_path,
            h3r_size=args.h3r_size,
            det_score_thr=args.det_score_thr,
            iou_thr=args.iou_thr,
            center_thr=args.center_thr,
        )

        key = (
            geom["ori_width"],
            geom["ori_height"],
            geom["h3r_width"],
            geom["h3r_height"],
        )
        geometry_counts[key] += len(items)

        if detections:
            frame_with_detections += 1

        diag = _hungarian_match_diagnostic(
            frame_rows,
            detections,
            geom["h3r_width"],
            geom["h3r_height"],
        )

        q = per_seq[sequence]
        q["rows"] += len(items)
        q["accepted"] += len(matches)
        accepted_rows += len(matches)

        for d in diag:
            assigned_ious.append(d["iou"])
            assigned_centers.append(d["center"])
            q["assigned_iou"].append(d["iou"])
            q["assigned_center"].append(d["center"])

        for info in matches.values():
            accepted_ious.append(info["iou"])
            accepted_centers.append(info["center"])

    def qtile(values, q):
        if not values:
            return None
        return float(np.quantile(np.asarray(values, np.float64), q))

    result = {
        "experiment": "External Person Matching Check",
        "coordinate_space": "human3r_true_shape",
        "h3r_size": int(args.h3r_size),
        "uses_gt_joint_coordinates": False,
        "uses_gt_for_person_matching": False,
        "thresholds_frozen": {
            "det_score_thr": float(args.det_score_thr),
            "iou_thr": float(args.iou_thr),
            "center_thr": float(args.center_thr),
        },
        "total_rows": int(total_rows),
        "accepted_rows": int(accepted_rows),
        "match_coverage": (
            float(accepted_rows / total_rows)
            if total_rows else 0.0
        ),
        "frame_with_detections": int(frame_with_detections),
        "assigned_before_gate": {
            "n": int(len(assigned_ious)),
            "iou_mean": (
                float(np.mean(assigned_ious))
                if assigned_ious else None
            ),
            "iou_median": qtile(assigned_ious, 0.50),
            "iou_p10": qtile(assigned_ious, 0.10),
            "iou_p90": qtile(assigned_ious, 0.90),
            "center_mean": (
                float(np.mean(assigned_centers))
                if assigned_centers else None
            ),
            "center_median": qtile(assigned_centers, 0.50),
            "center_p90": qtile(assigned_centers, 0.90),
        },
        "accepted": {
            "mean_iou": (
                float(np.mean(accepted_ious))
                if accepted_ious else None
            ),
            "mean_center": (
                float(np.mean(accepted_centers))
                if accepted_centers else None
            ),
        },
        "geometry_counts": {
            f"ori_{ow}x{oh}_to_h3r_{hw}x{hh}": int(n)
            for (ow, oh, hw, hh), n in geometry_counts.items()
        },
        "per_sequence": {},
    }

    for sequence, q in sorted(per_seq.items()):
        result["per_sequence"][sequence] = {
            "rows": int(q["rows"]),
            "accepted": int(q["accepted"]),
            "match_coverage": (
                float(q["accepted"] / q["rows"])
                if q["rows"] else 0.0
            ),
            "assigned_iou_mean": (
                float(np.mean(q["assigned_iou"]))
                if q["assigned_iou"] else None
            ),
            "assigned_center_mean": (
                float(np.mean(q["assigned_center"]))
                if q["assigned_center"] else None
            ),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("===== EXTERNAL PERSON MATCH CHECK =====")
    print(json.dumps(result, indent=2))


def _metric_summary(gains, raw_errors, corrected_errors):
    gains = np.asarray(gains, np.float64)
    raw_errors = np.asarray(raw_errors, np.float64)
    corrected_errors = np.asarray(corrected_errors, np.float64)
    if len(gains) == 0:
        return {
            "n": 0, "raw_mpjpe": None, "corrected_mpjpe": None,
            "mean_gain_mpjpe": None, "improved_ratio_mpjpe": None,
            "harm_gt_1mm_ratio": None,
        }
    return {
        "n": int(len(gains)),
        "raw_mpjpe": float(raw_errors.mean()),
        "corrected_mpjpe": float(corrected_errors.mean()),
        "mean_gain_mpjpe": float(gains.mean()),
        "median_gain_mpjpe": float(np.median(gains)),
        "improved_ratio_mpjpe": float((gains > 0).mean()),
        "harm_gt_1mm_ratio": float((gains < -1.0).mean()),
    }


@torch.no_grad()
def cmd_external_transfer(args):
    rows = read_stream(args.dump)
    if not rows:
        raise RuntimeError("empty dump")

    obs = load_external_observations(args.obs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, alpha = load_checkpoint(args.checkpoint, device=device)

    if int(ckpt.get("joint_count", 14)) != int(
        np.asarray(rows[0]["raw_j3d_rel"]).shape[0]
    ):
        raise RuntimeError("checkpoint/dump joint protocol mismatch")

    grouped = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[
            (str(row["sequence"]), int(row["frame_id"]))
        ].append((idx, row))

    corrected = [None] * len(rows)
    matched = np.zeros(len(rows), dtype=bool)
    match_iou = np.full(len(rows), np.nan, np.float32)
    match_center = np.full(len(rows), np.nan, np.float32)
    valid_joint_count = np.zeros(len(rows), np.int32)
    geometry_counts = defaultdict(int)

    for (sequence, frame_id), items in grouped.items():
        image_path = _image_path(
            args.image_root,
            sequence,
            frame_id,
        )
        if image_path is None:
            continue

        frame_obs = obs.get(sequence, {}).get(frame_id, [])
        frame_rows = [row for _, row in items]

        detections, matches, geom = _frame_external_match_data(
            frame_rows=frame_rows,
            frame_obs=frame_obs,
            image_path=image_path,
            h3r_size=args.h3r_size,
            det_score_thr=args.det_score_thr,
            iou_thr=args.iou_thr,
            center_thr=args.center_thr,
        )

        geometry_counts[
            (
                geom["ori_width"],
                geom["ori_height"],
                geom["h3r_width"],
                geom["h3r_height"],
            )
        ] += len(items)

        for local_idx, info in matches.items():
            global_idx, row = items[local_idx]
            det = detections[info["detection_id"]]

            # det["kp"] is already in Human3R true_shape coordinates.
            obs14, mask14, conf14 = coco_to_j14(
                det["kp"],
                conf_thr=args.kp_conf_thr,
            )
            if mask14.sum() < args.min_valid_joints:
                continue

            raw3 = np.asarray(row["raw_j3d_rel"], np.float32)
            raw2 = np.asarray(row["raw_j2d"], np.float32)
            if len(raw3) != 14 or raw2.shape != (14, 2):
                continue

            x = build_feature(raw3, raw2, obs14, mask14)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            delta = cap_delta(model(xt), 0.06)[0].cpu().numpy()

            corrected[global_idx] = raw3 + alpha * delta
            matched[global_idx] = True
            match_iou[global_idx] = info["iou"]
            match_center[global_idx] = info["center"]
            valid_joint_count[global_idx] = int(mask14.sum())

    all_g, all_r, all_c = [], [], []
    mat_g, mat_r, mat_c = [], [], []
    per_seq = defaultdict(
        lambda: {
            "g": [], "r": [], "c": [],
            "mg": [], "mr": [], "mc": [],
            "matched": 0, "total": 0,
        }
    )

    for idx, row in enumerate(rows):
        raw = np.asarray(row["raw_j3d_rel"], np.float32)
        gt = np.asarray(row["gt_j3d_rel"], np.float32)
        corr = corrected[idx] if matched[idx] else raw

        raw_e = float(
            np.linalg.norm(raw - gt, axis=-1).mean() * 1000.0
        )
        corr_e = float(
            np.linalg.norm(corr - gt, axis=-1).mean() * 1000.0
        )
        gain = raw_e - corr_e

        all_g.append(gain)
        all_r.append(raw_e)
        all_c.append(corr_e)

        sequence = str(row["sequence"])
        q = per_seq[sequence]
        q["g"].append(gain)
        q["r"].append(raw_e)
        q["c"].append(corr_e)
        q["total"] += 1

        if matched[idx]:
            mat_g.append(gain)
            mat_r.append(raw_e)
            mat_c.append(corr_e)
            q["mg"].append(gain)
            q["mr"].append(raw_e)
            q["mc"].append(corr_e)
            q["matched"] += 1

    overall = _metric_summary(all_g, all_r, all_c)
    matched_only = _metric_summary(mat_g, mat_r, mat_c)

    overall["match_coverage"] = float(matched.mean())
    overall["alpha"] = float(alpha)
    overall["mean_match_iou"] = (
        float(np.nanmean(match_iou))
        if matched.any() else None
    )
    overall["mean_match_center"] = (
        float(np.nanmean(match_center))
        if matched.any() else None
    )
    overall["mean_valid_external_joints"] = (
        float(valid_joint_count[matched].mean())
        if matched.any() else None
    )

    per_sequence = {}
    for sequence, q in per_seq.items():
        p = _metric_summary(q["g"], q["r"], q["c"])
        pm = _metric_summary(q["mg"], q["mr"], q["mc"])
        p["matched"] = int(q["matched"])
        p["total"] = int(q["total"])
        p["match_coverage"] = (
            q["matched"] / q["total"] if q["total"] else 0.0
        )
        p["matched_only"] = pm
        per_sequence[sequence] = p

    positive_sequences = sum(
        p["mean_gain_mpjpe"] is not None
        and p["mean_gain_mpjpe"] > 0
        for p in per_sequence.values()
    )

    # No method decision is allowed when matching failed operationally.
    if overall["match_coverage"] < 0.60:
        decision = "MATCHING_COVERAGE_INSUFFICIENT_DO_NOT_JUDGE_MODEL"
    else:
        strong_pass = (
            overall["mean_gain_mpjpe"] >= 1.0
            and matched_only["mean_gain_mpjpe"] is not None
            and matched_only["mean_gain_mpjpe"] >= 1.0
            and matched_only["improved_ratio_mpjpe"] >= 0.60
            and matched_only["harm_gt_1mm_ratio"] <= 0.20
            and positive_sequences == len(per_sequence)
        )

        weak_pass = (
            overall["mean_gain_mpjpe"] > 0
            and positive_sequences >= max(2, len(per_sequence) - 1)
        )

        if strong_pass:
            decision = "PROCEED_TO_JOINT_TO_SMPL_BOUNDED_INTEGRATION"
        elif weak_pass:
            decision = (
                "EXTERNAL_OBSERVATION_DOMAIN_GAP_"
                "FINE_TUNE_ON_TRAIN_SPLIT"
            )
        else:
            decision = (
                "EXTERNAL_TRANSFER_FAIL_ADD_VISUAL_OR_"
                "EXTERNAL_OBS_TRAINING"
            )

    result = {
        "experiment": "External-Keypoint Joint Correction Transfer",
        "checkpoint": str(args.checkpoint),
        "alpha_from_checkpoint": float(alpha),
        "coordinate_space": "human3r_true_shape",
        "h3r_size": int(args.h3r_size),
        "uses_gt_for_model_input": False,
        "uses_gt_for_person_matching": False,
        "gt_role": "post-hoc validation metric only",
        "thresholds_frozen": {
            "det_score_thr": float(args.det_score_thr),
            "kp_conf_thr": float(args.kp_conf_thr),
            "iou_thr": float(args.iou_thr),
            "center_thr": float(args.center_thr),
            "min_valid_joints": int(args.min_valid_joints),
        },
        "geometry_counts": {
            f"ori_{ow}x{oh}_to_h3r_{hw}x{hh}": int(n)
            for (ow, oh, hw, hh), n in geometry_counts.items()
        },
        "overall_with_raw_fallback": overall,
        "matched_only": matched_only,
        "positive_sequence_count": int(positive_sequences),
        "sequence_count": len(per_sequence),
        "per_sequence": per_sequence,
        "decision": decision,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("===== EXTERNAL-KEYPOINT JOINT CORRECTION TRANSFER =====")
    print(json.dumps(result, indent=2))


def cmd_legacy(args):
    if not LEGACY_RUNNER.exists():
        raise FileNotFoundError(LEGACY_RUNNER)
    cmd = [sys.executable, str(LEGACY_RUNNER)] + args.legacy_args
    subprocess.run(cmd, check=True)



# ===== WAO_J2S_PIPELINE_BEGIN =====
def _j2s_eval_command(out_name, port):
    return [
        "accelerate",
        "launch",
        "--num_processes", "1",
        "--num_machines", "1",
        "--mixed_precision", "no",
        "--dynamo_backend", "no",
        "--main_process_port", str(port),
        "eval/global_human/launch.py",
        "--weights", "./src/human3r_896L.pth",
        "--output_dir",
        f"./eval_results/global_human/{out_name}",
        "--eval_dataset", "3dpw",
        "--size", "512",
        "--reset_interval", "100",
        "--use_ttt3r",
    ]


def _summarize_j2s_audit(path):
    path = Path(path)
    if not path.exists():
        return {
            "records": 0,
            "matched": 0,
            "accepted": 0,
        }

    rows = []
    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    initial = []
    final = []
    improvements = []
    max_delta = []
    matched = 0
    accepted = 0

    frozen_max = {
        "root_delta_max": 0.0,
        "transl_delta_max": 0.0,
        "shape_delta_max": 0.0,
        "hand_jaw_delta_max": 0.0,
    }

    for row in rows:
        matched += int(row.get("matched", 0))
        accepted += int(row.get("accepted", 0))
        initial.extend(
            row.get("initial_target_mpjpe_mm", [])
        )
        final.extend(
            row.get("final_target_mpjpe_mm", [])
        )
        improvements.extend(
            row.get("fit_improvement_ratio", [])
        )
        max_delta.extend(
            row.get("max_local_pose_delta_rad", [])
        )

        for key in frozen_max:
            frozen_max[key] = max(
                frozen_max[key],
                float(row.get(key, 0.0)),
            )

    return {
        "records": len(rows),
        "matched": int(matched),
        "accepted": int(accepted),
        "accept_ratio": (
            float(accepted / matched)
            if matched else 0.0
        ),
        "initial_target_mpjpe_mm_mean": (
            float(np.mean(initial))
            if initial else None
        ),
        "final_target_mpjpe_mm_mean": (
            float(np.mean(final))
            if final else None
        ),
        "fit_improvement_ratio_mean": (
            float(np.mean(improvements))
            if improvements else None
        ),
        "max_local_pose_delta_rad": (
            float(np.max(max_delta))
            if max_delta else None
        ),
        **frozen_max,
    }


def _run_j2s_eval(
    *,
    out_name,
    port,
    sequences,
    stage,
    args,
    audit_path=None,
):
    env = os.environ.copy()

    env.update(
        {
            "HUMAN3R_NO_VIEWER": "1",
            "TORCH_HOME": str(H3R / "cache/torch"),
            "TORCH_EXTENSIONS_DIR": str(
                H3R / "cache/torch_extensions"
            ),
            "PYTHONPATH": "src:.",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "LD_LIBRARY_PATH": (
                "/root/data1/cuda-12.1/lib64:"
                + env.get("LD_LIBRARY_PATH", "")
            ),
            "WAO_3DPW_SPLIT": "validation",
            "WAO_J2S_SEQ_FILTER": ",".join(sequences),
            "WAO_HMR_STAGE": stage,
        }
    )

    # Start from raw Human3R. Do not trigger historical
    # A2/A4 sequence-level WAO_HMR_SAFE behavior.
    env.pop("WAO_HMR_SAFE", None)

    if stage == "joint_to_smpl_bounded":
        env.update(
            {
                "WAO_JCR_CKPT": str(args.checkpoint),
                "WAO_JCR_OBS": str(args.obs),
                "WAO_J2S_AUDIT": str(audit_path),
                "WAO_J2S_DET_SCORE_THR": "0.0",
                "WAO_J2S_KP_CONF_THR": "0.30",
                "WAO_J2S_IOU_THR": "0.55",
                "WAO_J2S_CENTER_THR": "0.04",
                "WAO_J2S_MIN_VALID_JOINTS": "6",
                "WAO_J2S_MAX_JOINT_DELTA": "0.06",
                "WAO_J2S_STEPS": "24",
                "WAO_J2S_LR": "0.005",
                "WAO_J2S_ANCHOR_WEIGHT": "0.02",
                "WAO_J2S_MIN_FIT_IMPROVEMENT": "0.05",
                "WAO_J2S_PATIENCE": "6",
            }
        )

    out_dir = (
        H3R / "eval_results"
        / "global_human" / out_name
    )
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    cmd = _j2s_eval_command(out_name, port)
    print("running:", " ".join(cmd), flush=True)

    subprocess.run(
        cmd,
        cwd=H3R,
        env=env,
        check=True,
    )
    return out_dir


def cmd_integrate_smpl(args):
    sequences = [
        item.strip()
        for item in args.sequences.split(",")
        if item.strip()
    ]
    if not sequences:
        raise RuntimeError("empty --sequences")

    allowed = {
        "courtyard_dancing_00",
        "courtyard_drinking_00",
        "courtyard_hug_00",
    }
    invalid = [
        item for item in sequences
        if item not in allowed
    ]
    if invalid:
        raise RuntimeError(
            "integrate-smpl is locked to held-out "
            f"validation sequences: {invalid}"
        )

    result_root = (
        ROOT / "results_safe"
        / "joint_correction"
        / "j2s_bounded"
    )
    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        result_root / f"{args.tag}_audit.jsonl"
    )
    summary_path = (
        result_root / f"{args.tag}_audit_summary.json"
    )

    if audit_path.exists():
        audit_path.unlink()

    baseline_dir = None
    if args.with_baseline:
        baseline_dir = _run_j2s_eval(
            out_name=(
                "3dpw_j2s_raw_validation_"
                + args.tag
            ),
            port=args.port,
            sequences=sequences,
            stage="j2s_raw_baseline",
            args=args,
        )

    method_dir = _run_j2s_eval(
        out_name=(
            "3dpw_j2s_bounded_validation_"
            + args.tag
        ),
        port=args.port + 1,
        sequences=sequences,
        stage="joint_to_smpl_bounded",
        args=args,
        audit_path=audit_path,
    )

    summary = _summarize_j2s_audit(
        audit_path
    )
    summary.update(
        {
            "experiment": (
                "Joint-to-SMPL Bounded "
                "Local Pose Integration"
            ),
            "split": "validation",
            "sequences": sequences,
            "checkpoint": str(args.checkpoint),
            "obs": str(args.obs),
            "alpha": 0.20,
            "max_joint_delta_rad": 0.06,
            "steps": 24,
            "lr": 0.005,
            "anchor_weight": 0.02,
            "min_fit_improvement": 0.05,
            "root_orientation_frozen": True,
            "translation_frozen": True,
            "shape_frozen": True,
            "camera_frozen": True,
            "hands_jaw_frozen": True,
            "baseline_dir": (
                str(baseline_dir)
                if baseline_dir is not None
                else None
            ),
            "method_dir": str(method_dir),
        }
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("===== J2S BOUNDED AUDIT SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("baseline:", baseline_dir)
    print("method:", method_dir)
    print("audit:", summary_path)
# ===== WAO_J2S_PIPELINE_END =====

def main():
    ap = argparse.ArgumentParser(
        description="Unified WAO-HMR joint-correction pipeline"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("--stream", required=True)
    p.set_defaults(func=lambda a: inspect_stream(a.stream))

    p = sub.add_parser("dump")
    p.add_argument("--split", choices=["train", "validation", "test"], required=True)
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--out-root", default=str(TRAIN_ROOT))
    p.add_argument("--run-name", required=True)
    p.add_argument("--port", type=int, default=29571)
    p.add_argument("--gpu", default="0")
    p.add_argument("--reset", action="store_true")
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("train")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_train)


    p = sub.add_parser("export-manifest")
    p.add_argument("--dump", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_export_manifest)

    p = sub.add_parser("prepare-vitpose-obs")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dump")
    source.add_argument("--manifest")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--image-root",
        default="/root/data1/datasets/3DPW/imageFiles",
    )
    p.add_argument(
        "--wham-root",
        default="/root/data1/models/WHAM",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--kp-conf-thr", type=float, default=0.30)
    p.add_argument("--min-valid-joints", type=int, default=6)
    p.add_argument("--reset", action="store_true")
    p.set_defaults(func=cmd_prepare_vitpose_obs)

    p = sub.add_parser("check-obs")
    p.add_argument("--dump", required=True)
    p.add_argument("--obs", required=True)
    p.set_defaults(func=cmd_check_obs)

    p = sub.add_parser("check-match")
    p.add_argument("--dump", required=True)
    p.add_argument("--obs", required=True)
    p.add_argument(
        "--image-root",
        default="/root/data1/datasets/3DPW/imageFiles",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--h3r-size", type=int, default=512)
    p.add_argument("--det-score-thr", type=float, default=0.0)
    p.add_argument("--iou-thr", type=float, default=0.55)
    p.add_argument("--center-thr", type=float, default=0.04)
    p.set_defaults(func=cmd_check_match)

    p = sub.add_parser("external-transfer")
    p.add_argument("--dump", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--obs", required=True)
    p.add_argument(
        "--image-root",
        default="/root/data1/datasets/3DPW/imageFiles",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--h3r-size", type=int, default=512)
    p.add_argument("--det-score-thr", type=float, default=0.75)
    p.add_argument("--kp-conf-thr", type=float, default=0.30)
    p.add_argument("--iou-thr", type=float, default=0.55)
    p.add_argument("--center-thr", type=float, default=0.04)
    p.add_argument("--min-valid-joints", type=int, default=6)
    p.set_defaults(func=cmd_external_transfer)

    p = sub.add_parser("integrate-smpl")
    p.add_argument(
        "--sequences",
        default="courtyard_dancing_00",
    )
    p.add_argument(
        "--checkpoint",
        default=str(
            TRAIN_ROOT
            / "model_multiseq_magnitude_calibrated"
            / "best.pt"
        ),
    )
    p.add_argument(
        "--obs",
        default=str(
            ROOT
            / "assets"
            / "validation_courtyard_vitpose_observations.pkl"
        ),
    )
    p.add_argument("--tag", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument(
        "--port",
        type=int,
        default=29621,
    )
    p.add_argument(
        "--with-baseline",
        action="store_true",
    )
    p.set_defaults(func=cmd_integrate_smpl)

    p = sub.add_parser("legacy")
    p.add_argument("legacy_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_legacy)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
