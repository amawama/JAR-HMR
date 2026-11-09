from __future__ import annotations

import os
import pickle
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

ROOT = Path("/root/data1/experiments/Human3R/wao_hmr")
DEFAULT_TRAIN_ROOT = ROOT / "training" / "t1_jcr"

# ---------- model ----------

class JointCorrectionRegressor(nn.Module):
    def __init__(self, joint_count: int):
        super().__init__()
        self.joint_count = int(joint_count)
        dim_in = self.joint_count * 6
        dim_out = self.joint_count * 3
        self.net = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(512, dim_out),
        )

    def forward(self, x):
        return self.net(x).view(-1, self.joint_count, 3)


def cap_delta(delta: torch.Tensor, cap_m: float = 0.06) -> torch.Tensor:
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
    return delta * torch.clamp(float(cap_m) / norm, max=1.0)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    joint_count = int(ckpt.get("joint_count", 14))
    model = JointCorrectionRegressor(joint_count)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    alpha = float(ckpt.get("eval_alpha", 0.20))
    return model, ckpt, alpha


# ---------- common feature construction ----------

def observation_scale(obs2d: np.ndarray, mask: np.ndarray) -> float:
    obs2d = np.asarray(obs2d, np.float32)
    mask = np.asarray(mask).astype(bool)
    pts = obs2d[mask]
    if len(pts):
        lo = np.nanmin(pts, axis=0)
        hi = np.nanmax(pts, axis=0)
        return float(max(hi[0] - lo[0], hi[1] - lo[1], 32.0))
    return 256.0


def build_feature(raw3d, raw2d, obs2d, mask) -> np.ndarray:
    raw3d = np.asarray(raw3d, np.float32)
    raw2d = np.asarray(raw2d, np.float32)
    obs2d = np.asarray(obs2d, np.float32)
    mask = np.asarray(mask, np.float32).copy()
    if raw3d.shape[0] != raw2d.shape[0] or raw2d.shape != obs2d.shape:
        raise ValueError(
            f"shape mismatch raw3={raw3d.shape} raw2={raw2d.shape} obs2={obs2d.shape}"
        )
    scale = observation_scale(obs2d, mask > 0.5)
    resid = (obs2d - raw2d) / scale
    resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
    resid[mask < 0.5] = 0.0
    return np.concatenate(
        [raw3d.reshape(-1), resid.reshape(-1), mask.reshape(-1)]
    ).astype(np.float32)


# ---------- Human3R official J14 dump hook ----------

_DUMP_LOCK = threading.Lock()
_DUMP_PRINTED = False
_DUMP_SHAPE_PRINTED = False
_DUMP_WRITTEN = 0

def _np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_pelvis_idx(pelvis_idx, nj):
    if torch.is_tensor(pelvis_idx):
        pelvis_idx = pelvis_idx.detach().cpu().reshape(-1).tolist()
    elif isinstance(pelvis_idx, np.ndarray):
        pelvis_idx = pelvis_idx.reshape(-1).tolist()
    elif not isinstance(pelvis_idx, (list, tuple)):
        pelvis_idx = [pelvis_idx]
    out = []
    for value in pelvis_idx:
        try:
            value = int(value)
        except Exception:
            continue
        if 0 <= value < nj:
            out.append(value)
    return out or [0]


def _pelvis_center(joints, pelvis_idx):
    joints = np.asarray(joints, np.float32)
    idx = _normalize_pelvis_idx(pelvis_idx, len(joints))
    return joints[idx].mean(axis=0, keepdims=True)


def _pairs(best_match, n_pred, n_gt):
    if best_match is None:
        return []
    arr = _np(best_match)
    if arr is None:
        return []
    arr = np.asarray(arr)
    out = []
    if arr.ndim == 2 and arr.shape[1] >= 2:
        for row in arr:
            try:
                pi, gi = int(row[0]), int(row[1])
            except Exception:
                continue
            if 0 <= pi < n_pred and 0 <= gi < n_gt:
                out.append((pi, gi))
    elif arr.ndim == 1:
        for gi, pi in enumerate(arr.tolist()):
            try:
                pi = int(pi)
            except Exception:
                continue
            if 0 <= pi < n_pred and gi < n_gt:
                out.append((pi, gi))
    return list(dict.fromkeys(out))


def _dump_contract(out, split):
    global _DUMP_PRINTED
    if _DUMP_PRINTED:
        return
    _DUMP_PRINTED = True
    print("===== WAO JOINT-CORRECTION DUMP CONTRACT =====", flush=True)
    print("split:", split, flush=True)
    print("output:", out, flush=True)
    print("modifies_predictions: False", flush=True)
    print("uses_test_gt_for_training: False", flush=True)
    print("joint_protocol: dynamic official pred/GT common count", flush=True)


def record_joint_dump(
    *,
    sequence,
    frame_id,
    pred_j2d,
    pred_j3d_c,
    gt_j2d,
    gt_j3d_c,
    best_match,
    pelvis_idx,
):
    global _DUMP_SHAPE_PRINTED, _DUMP_WRITTEN

    split = os.getenv("WAO_3DPW_SPLIT", "validation").strip().lower() or "validation"
    out_root = Path(os.getenv("WAO_T1_OUT_ROOT", str(DEFAULT_TRAIN_ROOT)))
    out = out_root / f"t1_joint_dump_{split}.pklstream"
    _dump_contract(out, split)

    pj3, pj2, gj3, gj2 = map(
        _np, [pred_j3d_c, pred_j2d, gt_j3d_c, gt_j2d]
    )
    if any(x is None for x in [pj3, pj2, gj3, gj2]):
        return 0
    if any(np.asarray(x).ndim != 3 for x in [pj3, pj2, gj3, gj2]):
        return 0

    pairs = _pairs(best_match, len(pj3), len(gj3))
    if not pairs:
        return 0

    if not _DUMP_SHAPE_PRINTED:
        _DUMP_SHAPE_PRINTED = True
        print(
            "[JOINT-DUMP] first shapes:",
            "pred_j3d=", tuple(pj3.shape),
            "gt_j3d=", tuple(gj3.shape),
            "pred_j2d=", tuple(pj2.shape),
            "gt_j2d=", tuple(gj2.shape),
            "bestMatch=", np.asarray(best_match).shape,
            flush=True,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with _DUMP_LOCK, out.open("ab") as stream:
        for pi, gi in pairs:
            if not (
                0 <= pi < len(pj3)
                and 0 <= gi < len(gj3)
                and 0 <= pi < len(pj2)
                and 0 <= gi < len(gj2)
            ):
                continue

            raw3_all = np.asarray(pj3[pi], np.float32)
            gt3_all = np.asarray(gj3[gi], np.float32)
            nj = min(len(raw3_all), len(gt3_all))
            n2 = min(len(pj2[pi]), len(gj2[gi]), nj)
            if nj <= 0 or n2 <= 0:
                continue

            raw3 = raw3_all[:nj]
            gt3 = gt3_all[:nj]

            raw2 = np.zeros((nj, 2), np.float32)
            gt2 = np.zeros((nj, 2), np.float32)
            mask = np.zeros((nj,), np.uint8)

            rr = np.asarray(pj2[pi], np.float32)[:n2]
            gg = np.asarray(gj2[gi], np.float32)[:n2]
            finite = np.isfinite(rr).all(axis=1) & np.isfinite(gg).all(axis=1)
            raw2[:n2] = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
            gt2[:n2] = np.nan_to_num(gg, nan=0.0, posinf=0.0, neginf=0.0)
            mask[:n2] = finite.astype(np.uint8)

            rec = {
                "sequence": str(sequence),
                "frame_id": int(frame_id),
                "pred_index": int(pi),
                "gt_index": int(gi),
                "joint_count": int(nj),
                "raw_j3d_rel": (
                    raw3 - _pelvis_center(raw3, pelvis_idx)
                ).astype(np.float32),
                "gt_j3d_rel": (
                    gt3 - _pelvis_center(gt3, pelvis_idx)
                ).astype(np.float32),
                "raw_j2d": raw2,
                "gt_j2d": gt2,
                "valid2d": mask,
                "n_j2d_common": int(n2),
            }
            pickle.dump(rec, stream, protocol=pickle.HIGHEST_PROTOCOL)
            wrote += 1
            _DUMP_WRITTEN += 1

    if _DUMP_WRITTEN == 1 or (_DUMP_WRITTEN and _DUMP_WRITTEN % 500 == 0):
        print("[JOINT-DUMP] written=", _DUMP_WRITTEN, flush=True)
    return wrote


# ---------- external observation utilities ----------

# COCO 17 -> Human3R official J14:
# 0 RAnkle, 1 RKnee, 2 RHip, 3 LHip, 4 LKnee, 5 LAnkle,
# 6 RWrist, 7 RElbow, 8 RShoulder, 9 LShoulder, 10 LElbow,
# 11 LWrist, 12 Neck, 13 Head.
COCO_DIRECT = {
    0: 16, 1: 14, 2: 12, 3: 11, 4: 13, 5: 15,
    6: 10, 7: 8, 8: 6, 9: 5, 10: 7, 11: 9,
}


def bbox_from_points(points, conf=None, conf_thr=0.30):
    points = np.asarray(points, np.float32)
    valid = np.isfinite(points[:, :2]).all(axis=1)
    if conf is not None:
        valid &= np.asarray(conf) >= float(conf_thr)
    if valid.sum() < 3:
        return None
    xy = points[valid, :2]
    x1, y1 = xy.min(axis=0)
    x2, y2 = xy.max(axis=0)
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    return np.array(
        [x1 - 0.08*w, y1 - 0.08*h, x2 + 0.08*w, y2 + 0.08*h],
        np.float32,
    )


def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def center_distance(a, b, width, height):
    ca = np.asarray([(a[0]+a[2])/2.0, (a[1]+a[3])/2.0])
    cb = np.asarray([(b[0]+b[2])/2.0, (b[1]+b[3])/2.0])
    diag = max(float(np.hypot(width, height)), 1.0)
    return float(np.linalg.norm(ca-cb) / diag)


def coco_to_j14(kp17, conf_thr=0.30):
    kp17 = np.asarray(kp17, np.float32)
    out = np.zeros((14, 2), np.float32)
    mask = np.zeros((14,), np.float32)
    conf = np.zeros((14,), np.float32)

    for j14, coco in COCO_DIRECT.items():
        if coco >= len(kp17):
            continue
        c = float(kp17[coco, 2]) if kp17.shape[1] >= 3 else 1.0
        if np.isfinite(kp17[coco, :2]).all() and c >= conf_thr:
            out[j14] = kp17[coco, :2]
            mask[j14] = 1.0
            conf[j14] = c

    # Neck: midpoint of COCO shoulders if both reliable.
    if len(kp17) > 6:
        c5 = float(kp17[5, 2]) if kp17.shape[1] >= 3 else 1.0
        c6 = float(kp17[6, 2]) if kp17.shape[1] >= 3 else 1.0
        if (
            c5 >= conf_thr and c6 >= conf_thr
            and np.isfinite(kp17[[5, 6], :2]).all()
        ):
            out[12] = 0.5 * (kp17[5, :2] + kp17[6, :2])
            mask[12] = 1.0
            conf[12] = min(c5, c6)

    # Head is deliberately masked: COCO nose/eyes are not the same
    # official J14 head definition. Training already contains dropout.
    mask[13] = 0.0
    conf[13] = 0.0
    return out, mask, conf


def load_external_observations(path):
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "sequences" in obj:
        return obj["sequences"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"unsupported observation pickle format: {type(obj)}")


def rescale_detection_item(item, width, height, det_score_thr=0.75):
    score = float(item.get("score", 0.0))
    if score < det_score_thr:
        return None
    kp = np.asarray(item["kp2d"], np.float32).copy()
    res = np.asarray(item.get("res", [width, height]), np.float32).reshape(-1)
    if len(res) < 2 or kp.ndim != 2 or kp.shape[1] < 2:
        return None
    kp[:, 0] *= float(width) / max(float(res[0]), 1.0)
    kp[:, 1] *= float(height) / max(float(res[1]), 1.0)
    conf = kp[:, 2] if kp.shape[1] >= 3 else np.ones(len(kp), np.float32)
    box = bbox_from_points(kp, conf)
    if box is None:
        return None
    return {"kp": kp, "bbox": box, "score": score}


def match_frame_rows_to_detections(
    frame_rows,
    detections,
    width,
    height,
    iou_thr=0.55,
    center_thr=0.04,
):
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for external person matching")

    raw_boxes = []
    row_ids = []
    for idx, row in enumerate(frame_rows):
        box = bbox_from_points(np.asarray(row["raw_j2d"], np.float32))
        if box is not None:
            row_ids.append(idx)
            raw_boxes.append(box)

    det_ids = list(range(len(detections)))
    if not row_ids or not det_ids:
        return {}

    cost = np.full((len(row_ids), len(det_ids)), 100.0, np.float32)
    meta = {}

    for r, row_idx in enumerate(row_ids):
        for c, det_idx in enumerate(det_ids):
            iou = bbox_iou(raw_boxes[r], detections[det_idx]["bbox"])
            center = center_distance(
                raw_boxes[r], detections[det_idx]["bbox"], width, height
            )
            cost[r, c] = (1.0 - iou) + 2.5 * center
            meta[(r, c)] = (iou, center)

    rr, cc = linear_sum_assignment(cost)
    matches = {}
    for r, c in zip(rr, cc):
        iou, center = meta[(r, c)]
        if iou >= iou_thr and center <= center_thr:
            matches[row_ids[r]] = {
                "detection_id": det_ids[c],
                "iou": float(iou),
                "center": float(center),
            }
    return matches

# ===== WAO_J2S_BOUNDED_RUNTIME_BEGIN =====
_J2S_RUNTIME_CACHE = {}

def _j2s_env_float(name, default):
    return float(os.environ.get(name, str(default)))

def _j2s_env_int(name, default):
    return int(os.environ.get(name, str(default)))

def _j2s_pelvis_indices(pelvis_idx, nj):
    if torch.is_tensor(pelvis_idx):
        vals = pelvis_idx.detach().cpu().reshape(-1).tolist()
    elif isinstance(pelvis_idx, np.ndarray):
        vals = pelvis_idx.reshape(-1).tolist()
    elif isinstance(pelvis_idx, (list, tuple)):
        vals = list(pelvis_idx)
    else:
        vals = [pelvis_idx]
    out = []
    for v in vals:
        try: v = int(v)
        except Exception: continue
        if 0 <= v < nj: out.append(v)
    return out or [0]

def _j2s_rel(joints, pelvis_idx):
    ids = _j2s_pelvis_indices(pelvis_idx, joints.shape[1])
    return joints - joints[:, ids].mean(dim=1, keepdim=True)

def _j2s_root(joints, pelvis_idx):
    ids = _j2s_pelvis_indices(pelvis_idx, joints.shape[1])
    return joints[:, ids].mean(dim=1)

def _j2s_bound_pose_delta(raw_delta, cap_rad):
    n = torch.linalg.vector_norm(raw_delta, dim=-1, keepdim=True).clamp_min(1e-8)
    return raw_delta * (float(cap_rad) * torch.tanh(n / float(cap_rad)) / n)

def _j2s_runtime(device):
    ckpt_path = os.environ.get('WAO_JCR_CKPT','').strip()
    obs_path = os.environ.get('WAO_JCR_OBS','').strip()
    if not ckpt_path or not obs_path:
        raise RuntimeError('WAO_JCR_CKPT and WAO_JCR_OBS are required')
    key = (str(device), ckpt_path, obs_path)
    if key not in _J2S_RUNTIME_CACHE:
        model, ckpt, alpha = load_checkpoint(ckpt_path, device=device)
        _J2S_RUNTIME_CACHE[key] = {
            'model': model, 'ckpt': ckpt, 'alpha': float(alpha),
            'observations': load_external_observations(obs_path),
        }
        print('[WAO_J2S_STABLE] runtime loaded alpha=', alpha, flush=True)
    return _J2S_RUNTIME_CACHE[key]

def _j2s_write_audit(record):
    p = os.environ.get('WAO_J2S_AUDIT','').strip()
    if not p: return
    import json
    from pathlib import Path
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

def _j2s_person_error(a,b):
    return torch.linalg.vector_norm(a-b, dim=-1).mean(dim=-1)

def _refine_joint_to_smpl_frame_iterative24(*, sequence, frame_id, image, pred,
        raw_pred_j3d_c, raw_pred_j2d, smpl_layer, smpl_model, K, pelvis_idx):
    if raw_pred_j3d_c is None or raw_pred_j2d is None or not torch.is_tensor(raw_pred_j3d_c):
        return False
    n_humans = int(raw_pred_j3d_c.shape[0])
    if n_humans <= 0: return False

    device = raw_pred_j3d_c.device
    rt = _j2s_runtime(device)
    model, alpha, observations = rt['model'], rt['alpha'], rt['observations']
    frame_obs = observations.get(str(sequence),{}).get(int(frame_id),[])
    if not frame_obs: return False

    h, w = int(image.shape[-2]), int(image.shape[-1])
    dets=[]
    for item in frame_obs:
        d = rescale_detection_item(item, w, h,
            det_score_thr=_j2s_env_float('WAO_J2S_DET_SCORE_THR',0.0))
        if d is not None: dets.append(d)
    if not dets: return False

    raw2 = raw_pred_j2d.detach().cpu().numpy()
    rows=[{'raw_j2d':raw2[i]} for i in range(n_humans)]
    matches = match_frame_rows_to_detections(
        rows, dets, w, h,
        iou_thr=_j2s_env_float('WAO_J2S_IOU_THR',0.55),
        center_thr=_j2s_env_float('WAO_J2S_CENTER_THR',0.04))
    if not matches: return False

    raw3 = raw_pred_j3d_c.detach().float().cpu().numpy()
    pelvis_ids = _j2s_pelvis_indices(pelvis_idx, raw_pred_j3d_c.shape[1])
    selected, targets = [], []
    with torch.no_grad():
        for pid, info in sorted(matches.items()):
            obs14, mask14, _ = coco_to_j14(
                dets[info['detection_id']]['kp'],
                conf_thr=_j2s_env_float('WAO_J2S_KP_CONF_THR',0.30))
            if int(mask14.sum()) < _j2s_env_int('WAO_J2S_MIN_VALID_JOINTS',6): continue
            if raw3[pid].shape != (14,3): continue
            rr = (raw3[pid] - raw3[pid][pelvis_ids].mean(axis=0,keepdims=True)).astype(np.float32)
            feat = build_feature(rr, raw2[pid].astype(np.float32), obs14, mask14)
            dj = cap_delta(model(torch.from_numpy(feat).unsqueeze(0).to(device)),0.06)[0]
            targets.append(torch.from_numpy(rr).to(device) + float(alpha)*dj)
            selected.append(int(pid))
    if not selected: return False

    pose_frame = pred['rotvec'][frame_id]
    pids = torch.as_tensor(selected, dtype=torch.long, device=pose_frame.device)
    target_rel = torch.stack(targets,0).detach()
    raw_pose_all = pose_frame.detach().clone()
    root_before = raw_pose_all[:,0].clone()
    tail_before = raw_pose_all[:,22:].clone()
    transl_before = pred['transl'][frame_id].detach().clone()
    shape_before = pred['shape'][frame_id].detach().clone()
    expr_before = pred['expression'][frame_id].detach().clone()

    base = raw_pose_all[pids].detach()
    shape = pred['shape'][frame_id][pids].detach()
    transl = pred['transl'][frame_id][pids].detach()
    expr = pred['expression'][frame_id][pids].detach()
    for p in smpl_layer.parameters(): p.requires_grad_(False)

    cap = _j2s_env_float('WAO_J2S_MAX_JOINT_DELTA',0.06)
    steps = _j2s_env_int('WAO_J2S_STEPS',24)
    lr = _j2s_env_float('WAO_J2S_LR',0.005)
    anchor = _j2s_env_float('WAO_J2S_ANCHOR_WEIGHT',0.02)
    min_gain = _j2s_env_float('WAO_J2S_MIN_FIT_IMPROVEMENT',0.05)
    patience = _j2s_env_int('WAO_J2S_PATIENCE',6)
    Kb = K.to(base.device).unsqueeze(0).expand(len(selected),-1,-1)

    def forward(param):
        bd = _j2s_bound_pose_delta(param, cap)
        pose = torch.cat([base[:,:1], base[:,1:22]+bd, base[:,22:]],1)
        out = smpl_layer(pose, shape, transl, None, None, K=Kb, expression=expr)
        verts = smpl_model.smplx2smpl @ out['smpl_v3d']
        joints = smpl_model.j_regressor @ verts
        return (
            pose,
            _j2s_rel(joints,pelvis_idx),
            _j2s_root(joints,pelvis_idx),
            bd,
        )

    raw_rel = _j2s_rel(raw_pred_j3d_c[pids].detach(), pelvis_idx)
    raw_root = _j2s_root(raw_pred_j3d_c[pids].detach(), pelvis_idx)

    zero = torch.zeros((len(selected),21,3),dtype=base.dtype,device=base.device)
    with torch.no_grad():
        _, zero_rel, zero_root, zero_bd = forward(zero)
        zero_recon = _j2s_person_error(zero_rel, raw_rel)
        zero_root_drift_mm = (
            torch.linalg.vector_norm(zero_root - raw_root, dim=-1)
            * 1000.0
        )
        init_err = _j2s_person_error(zero_rel, target_rel)

    # Exact raw pose is the per-person floor.
    best_err = init_err.clone()
    best_local = base[:,1:22].clone()
    best_bd = zero_bd.clone()
    best_root = zero_root.clone()
    best_step = torch.zeros(len(selected),dtype=torch.long,device=device)

    param = torch.zeros_like(zero, requires_grad=True)
    opt = torch.optim.Adam([param], lr=lr)
    stall=0; prev=float(best_err.mean().cpu()); executed=0
    for s in range(1,max(1,steps)+1):
        executed=s
        _, rel, _, bd = forward(param)
        r = rel-target_rel
        per = torch.nn.functional.smooth_l1_loss(
            r, torch.zeros_like(r), beta=0.003, reduction='none').mean(dim=(1,2))
        per = per + float(anchor)*bd.square().mean(dim=(1,2))
        loss=per.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([param],1.0); opt.step()
        with torch.no_grad():
            pose2, rel2, root2, bd2 = forward(param)
            err2 = _j2s_person_error(rel2,target_rel)
            better = err2 < best_err
            if bool(better.any().item()):
                best_err[better]=err2[better]
                best_local[better]=pose2[better,1:22]
                best_bd[better]=bd2[better]
                best_root[better]=root2[better]
                best_step[better]=s
            cur=float(best_err.mean().cpu())
            if cur < prev-1e-7: prev=cur; stall=0
            else: stall+=1
            if stall>=patience: break

    with torch.no_grad():
        gain=(init_err-best_err)/init_err.clamp_min(1e-8)
        accepted=(
            (zero_recon*1000.0 <= 1.0)
            & (best_step>0)
            & (gain>=float(min_gain))
        )

        # Analytic root-preserving compensation:
        # local pose is kept, while translation is shifted exactly by
        # raw_root - refined_root. This is not optimized/tuned.
        root_comp = raw_root - best_root
        root_comp_mm = (
            torch.linalg.vector_norm(root_comp, dim=-1)
            * 1000.0
        )

        accepted_pids=[]
        for j,pid in enumerate(selected):
            if bool(accepted[j].item()):
                pred['rotvec'][frame_id][pid,1:22].copy_(best_local[j])
                pred['transl'][frame_id][pid].add_(root_comp[j])
                accepted_pids.append(pid)

        # Verify the compensated SMPL root using the same official
        # differentiable forward path.
        compensated_root_drift_mm = torch.zeros_like(root_comp_mm)
        if accepted_pids:
            pose_after = pred['rotvec'][frame_id][pids].detach()
            transl_after = pred['transl'][frame_id][pids].detach()

            out_after = smpl_layer(
                pose_after,
                shape,
                transl_after,
                None,
                None,
                K=Kb,
                expression=expr,
            )
            verts_after = smpl_model.smplx2smpl @ out_after['smpl_v3d']
            joints_after = smpl_model.j_regressor @ verts_after
            root_after = _j2s_root(joints_after, pelvis_idx)

            compensated_root_drift_mm = (
                torch.linalg.vector_norm(root_after - raw_root, dim=-1)
                * 1000.0
            )

    def maxnorm(x):
        return float(torch.linalg.vector_norm(x.reshape(x.shape[0],-1),dim=-1).max().detach().cpu()) if x.numel() else 0.0
    rd=maxnorm(pred['rotvec'][frame_id][:,:1]-root_before[:,None])
    td=maxnorm(pred['transl'][frame_id]-transl_before)
    sd=maxnorm(pred['shape'][frame_id]-shape_before)
    ed=maxnorm(pred['expression'][frame_id]-expr_before)
    hd=maxnorm(pred['rotvec'][frame_id][:,22:]-tail_before)

    # Root orientation / shape / expression / hand-jaw remain hard frozen.
    # Translation may change only by the deterministic root compensation.
    if max(rd,sd,ed,hd)>1e-8:
        raise RuntimeError('J2S root-preserving frozen-state violation')

    # Numerical self-check: accepted instances must restore raw absolute root.
    if accepted_pids:
        accepted_mask = accepted
        max_post_root_drift = float(
            compensated_root_drift_mm[accepted_mask].max().detach().cpu()
        )
        if max_post_root_drift > 0.01:
            raise RuntimeError(
                'J2S root compensation failed: '
                f'post_root_drift_mm={max_post_root_drift}'
            )
    else:
        max_post_root_drift = 0.0

    dn=torch.linalg.vector_norm(best_bd,dim=-1)
    _j2s_write_audit({
        'sequence':str(sequence),'frame_id':int(frame_id),
        'matched':len(selected),'accepted':len(accepted_pids),
        'zero_reconstruction_mpjpe_mm':(zero_recon.cpu().numpy()*1000).astype(float).tolist(),
        'initial_target_mpjpe_mm':(init_err.cpu().numpy()*1000).astype(float).tolist(),
        # compatibility with existing summary + explicit best-state field
        'final_target_mpjpe_mm':(best_err.cpu().numpy()*1000).astype(float).tolist(),
        'best_target_mpjpe_mm':(best_err.cpu().numpy()*1000).astype(float).tolist(),
        'fit_improvement_ratio':gain.cpu().numpy().astype(float).tolist(),
        'precomp_root_drift_mm':root_comp_mm.cpu().numpy().astype(float).tolist(),
        'translation_compensation_mm':root_comp_mm.cpu().numpy().astype(float).tolist(),
        'postcomp_root_drift_mm':compensated_root_drift_mm.cpu().numpy().astype(float).tolist(),
        'best_step':best_step.cpu().numpy().astype(int).tolist(),
        'max_local_pose_delta_rad':dn.max(dim=-1).values.cpu().numpy().astype(float).tolist(),
        'root_delta_max':rd,'translation_compensation_param_delta_max':td,
        'shape_delta_max':sd,'expression_delta_max':ed,'hand_jaw_delta_max':hd,
        'max_postcomp_root_drift_mm':max_post_root_drift,
        'root_preservation_mode':'analytic_translation_compensation',
        'steps_executed':executed,'lr':lr,'alpha':float(alpha)})

    if int(frame_id)==0 or int(frame_id)%100==0:
        print('[WAO_J2S_STABLE]',sequence,frame_id,
              'matched=',len(selected),'accepted=',len(accepted_pids),
              'zero_mm=',round(float(zero_recon.mean().cpu())*1000,6),
              'target_mm=',round(float(init_err.mean().cpu())*1000,3),
              'best_mm=',round(float(best_err.mean().cpu())*1000,3),
              'precomp_root_mm=',round(float(root_comp_mm.mean().cpu()),3),
              'postcomp_root_mm=',round(float(compensated_root_drift_mm.mean().cpu()),6),
              'steps=',executed,flush=True)
    return bool(accepted_pids)
# ===== WAO_J2S_BOUNDED_RUNTIME_END =====

# ===== WAO_J2S_LEARNED_BOUNDED_BEGIN =====
_J2S_LEARNED_RUNTIME_CACHE = {}
_DEFAULT_LEARNED_J2S_CKPT = (
    ROOT / 'assets' / 'learned_bounded_pose' / 'joint_to_smpl_learned_bounded.pt'
)

class _J2SLearnedBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(d * 2, d),
        )

    def forward(self, x):
        return x + self.net(x)


class _J2SLearnedProposal(nn.Module):
    def __init__(self, input_dim, feature_mean, feature_std):
        super().__init__()
        self.register_buffer(
            'feature_mean',
            torch.as_tensor(feature_mean, dtype=torch.float32),
        )
        self.register_buffer(
            'feature_std',
            torch.as_tensor(feature_std, dtype=torch.float32),
        )
        self.inp = nn.Sequential(
            nn.Linear(int(input_dim), 384),
            nn.LayerNorm(384),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            _J2SLearnedBlock(384),
            _J2SLearnedBlock(384),
            _J2SLearnedBlock(384),
        )
        self.head = nn.Linear(384, 21 * 3)

    def forward(self, x):
        return self.head(self.blocks(self.inp(x))).reshape(-1, 21, 3)


def _j2s_learned_runtime(device):
    ckpt_path = Path(
        os.environ.get(
            'WAO_J2S_LEARNED_CKPT',
            str(_DEFAULT_LEARNED_J2S_CKPT),
        )
    )
    key = (str(device), str(ckpt_path))
    if key not in _J2S_LEARNED_RUNTIME_CACHE:
        if not ckpt_path.is_file():
            raise RuntimeError(f'learned J2S checkpoint missing: {ckpt_path}')
        state = torch.load(ckpt_path, map_location='cpu')
        mean = np.asarray(state['feature_mean'], np.float32)
        std = np.asarray(state['feature_std'], np.float32)
        model = _J2SLearnedProposal(
            int(state['input_dim']), mean, std
        )
        model.load_state_dict(state['model_state'], strict=True)
        model.to(device).eval()
        _J2S_LEARNED_RUNTIME_CACHE[key] = {
            'model': model,
            'mean': mean,
            'std': std,
            'state': state,
        }
        print(
            '[WAO_J2S_LEARNED] runtime loaded ckpt=',
            ckpt_path,
            flush=True,
        )
    return _J2S_LEARNED_RUNTIME_CACHE[key]


def _j2s_learned_feature(
    raw_local_pose,
    raw_j3d_rel,
    jcr_delta_rel,
    mask14,
    conf14,
    shape,
    match_iou,
    match_center,
    valid_joint_count,
):
    return np.concatenate([
        np.asarray(raw_local_pose, np.float32).reshape(-1),
        np.asarray(raw_j3d_rel, np.float32).reshape(-1),
        np.asarray(jcr_delta_rel, np.float32).reshape(-1),
        np.asarray(mask14, np.float32).reshape(-1),
        np.asarray(conf14, np.float32).reshape(-1),
        np.asarray(shape, np.float32).reshape(-1),
        np.asarray([
            float(match_iou),
            float(match_center),
            float(valid_joint_count) / 14.0,
        ], np.float32),
    ]).astype(np.float32)


def _refine_joint_to_smpl_frame_learned(*, sequence, frame_id, image, pred,
        raw_pred_j3d_c, raw_pred_j2d, smpl_layer, smpl_model, K, pelvis_idx):
    if raw_pred_j3d_c is None or raw_pred_j2d is None or not torch.is_tensor(raw_pred_j3d_c):
        return False
    n_humans = int(raw_pred_j3d_c.shape[0])
    if n_humans <= 0:
        return False

    device = raw_pred_j3d_c.device
    rt = _j2s_runtime(device)
    jcr_model, alpha, observations = rt['model'], rt['alpha'], rt['observations']
    learned = _j2s_learned_runtime(device)
    proposal, feat_mean, feat_std = learned['model'], learned['mean'], learned['std']

    frame_obs = observations.get(str(sequence), {}).get(int(frame_id), [])
    if not frame_obs:
        return False

    h, w = int(image.shape[-2]), int(image.shape[-1])
    dets = []
    for item in frame_obs:
        d = rescale_detection_item(
            item,
            w,
            h,
            det_score_thr=_j2s_env_float('WAO_J2S_DET_SCORE_THR', 0.0),
        )
        if d is not None:
            dets.append(d)
    if not dets:
        return False

    raw2 = raw_pred_j2d.detach().cpu().numpy()
    rows = [{'raw_j2d': raw2[i]} for i in range(n_humans)]
    matches = match_frame_rows_to_detections(
        rows,
        dets,
        w,
        h,
        iou_thr=_j2s_env_float('WAO_J2S_IOU_THR', 0.55),
        center_thr=_j2s_env_float('WAO_J2S_CENTER_THR', 0.04),
    )
    if not matches:
        return False

    raw3 = raw_pred_j3d_c.detach().float().cpu().numpy()
    pelvis_ids = _j2s_pelvis_indices(pelvis_idx, raw_pred_j3d_c.shape[1])
    pose_frame = pred['rotvec'][frame_id]
    raw_pose_all = pose_frame.detach().clone()
    root_before = raw_pose_all[:, 0].clone()
    tail_before = raw_pose_all[:, 22:].clone()
    transl_before = pred['transl'][frame_id].detach().clone()
    shape_before = pred['shape'][frame_id].detach().clone()
    expr_before = pred['expression'][frame_id].detach().clone()

    selected = []
    targets = []
    features = []
    meta = []

    with torch.no_grad():
        for pid, info in sorted(matches.items()):
            obs14, mask14, conf14 = coco_to_j14(
                dets[info['detection_id']]['kp'],
                conf_thr=_j2s_env_float('WAO_J2S_KP_CONF_THR', 0.30),
            )
            valid_count = int(mask14.sum())
            if valid_count < _j2s_env_int('WAO_J2S_MIN_VALID_JOINTS', 6):
                continue
            if raw3[pid].shape != (14, 3):
                continue

            rr = (
                raw3[pid]
                - raw3[pid][pelvis_ids].mean(axis=0, keepdims=True)
            ).astype(np.float32)
            feat_jcr = build_feature(
                rr,
                raw2[pid].astype(np.float32),
                obs14,
                mask14,
            )
            dj = cap_delta(
                jcr_model(
                    torch.from_numpy(feat_jcr).unsqueeze(0).to(device)
                ),
                0.06,
            )[0]
            target = torch.from_numpy(rr).to(device) + float(alpha) * dj

            shape_np = (
                pred['shape'][frame_id][pid]
                .detach().cpu().numpy().astype(np.float32)
            )
            local_np = (
                raw_pose_all[pid, 1:22]
                .detach().cpu().numpy().astype(np.float32)
            )
            jcr_delta_np = (
                (target - torch.from_numpy(rr).to(device))
                .detach().cpu().numpy().astype(np.float32)
            )
            x = _j2s_learned_feature(
                local_np,
                rr,
                jcr_delta_np,
                mask14,
                conf14,
                shape_np,
                info['iou'],
                info['center'],
                valid_count,
            )
            x = (x - feat_mean) / feat_std

            selected.append(int(pid))
            targets.append(target)
            features.append(x)
            meta.append({
                'iou': float(info['iou']),
                'center': float(info['center']),
                'valid_joint_count': valid_count,
            })

    if not selected:
        return False

    pids = torch.as_tensor(selected, dtype=torch.long, device=pose_frame.device)
    base = raw_pose_all[pids].detach()
    shape = pred['shape'][frame_id][pids].detach()
    transl = pred['transl'][frame_id][pids].detach()
    expr = pred['expression'][frame_id][pids].detach()
    target_rel = torch.stack(targets, 0).detach()
    x = torch.from_numpy(np.stack(features).astype(np.float32)).to(device)

    for p in smpl_layer.parameters():
        p.requires_grad_(False)

    cap = _j2s_env_float('WAO_J2S_MAX_JOINT_DELTA', 0.06)
    min_gain = _j2s_env_float('WAO_J2S_MIN_FIT_IMPROVEMENT', 0.05)
    Kb = K.to(base.device).unsqueeze(0).expand(len(selected), -1, -1)

    def forward_with_delta(delta):
        pose = torch.cat([
            base[:, :1],
            base[:, 1:22] + delta,
            base[:, 22:],
        ], 1)
        out = smpl_layer(
            pose,
            shape,
            transl,
            None,
            None,
            K=Kb,
            expression=expr,
        )
        verts = smpl_model.smplx2smpl @ out['smpl_v3d']
        joints = smpl_model.j_regressor @ verts
        return (
            pose,
            _j2s_rel(joints, pelvis_idx),
            _j2s_root(joints, pelvis_idx),
        )

    raw_rel = _j2s_rel(raw_pred_j3d_c[pids].detach(), pelvis_idx)
    raw_root = _j2s_root(raw_pred_j3d_c[pids].detach(), pelvis_idx)

    with torch.no_grad():
        zero = torch.zeros((len(selected), 21, 3), dtype=base.dtype, device=device)
        _, zero_rel, zero_root = forward_with_delta(zero)
        zero_recon = _j2s_person_error(zero_rel, raw_rel)
        init_err = _j2s_person_error(raw_rel, target_rel)

        raw_delta = proposal(x)
        delta = _j2s_bound_pose_delta(raw_delta, cap)
        cand_pose, cand_rel, cand_root = forward_with_delta(delta)
        cand_err = _j2s_person_error(cand_rel, target_rel)
        gain = (init_err - cand_err) / init_err.clamp_min(1e-8)
        accepted = (
            (zero_recon * 1000.0 <= 1.0)
            & (gain >= float(min_gain))
        )

        root_comp = raw_root - cand_root
        root_comp_mm = (
            torch.linalg.vector_norm(root_comp, dim=-1) * 1000.0
        )

        accepted_pids = []
        for j, pid in enumerate(selected):
            if bool(accepted[j].item()):
                pred['rotvec'][frame_id][pid, 1:22].copy_(cand_pose[j, 1:22])
                pred['transl'][frame_id][pid].add_(root_comp[j])
                accepted_pids.append(pid)

        compensated_root_drift_mm = torch.zeros_like(root_comp_mm)
        if accepted_pids:
            pose_after = pred['rotvec'][frame_id][pids].detach()
            transl_after = pred['transl'][frame_id][pids].detach()
            out_after = smpl_layer(
                pose_after,
                shape,
                transl_after,
                None,
                None,
                K=Kb,
                expression=expr,
            )
            verts_after = smpl_model.smplx2smpl @ out_after['smpl_v3d']
            joints_after = smpl_model.j_regressor @ verts_after
            root_after = _j2s_root(joints_after, pelvis_idx)
            compensated_root_drift_mm = (
                torch.linalg.vector_norm(root_after - raw_root, dim=-1)
                * 1000.0
            )

    def maxnorm(v):
        return (
            float(torch.linalg.vector_norm(
                v.reshape(v.shape[0], -1), dim=-1
            ).max().detach().cpu())
            if v.numel() else 0.0
        )

    rd = maxnorm(pred['rotvec'][frame_id][:, :1] - root_before[:, None])
    td = maxnorm(pred['transl'][frame_id] - transl_before)
    sd = maxnorm(pred['shape'][frame_id] - shape_before)
    ed = maxnorm(pred['expression'][frame_id] - expr_before)
    hd = maxnorm(pred['rotvec'][frame_id][:, 22:] - tail_before)

    if max(rd, sd, ed, hd) > 1e-8:
        raise RuntimeError('learned J2S frozen-state violation')

    if accepted_pids:
        max_post_root_drift = float(
            compensated_root_drift_mm[accepted].max().detach().cpu()
        )
        if max_post_root_drift > 0.01:
            raise RuntimeError(
                'learned J2S root compensation failed: '
                f'post_root_drift_mm={max_post_root_drift}'
            )
    else:
        max_post_root_drift = 0.0

    dn = torch.linalg.vector_norm(delta, dim=-1)
    _j2s_write_audit({
        'sequence': str(sequence),
        'frame_id': int(frame_id),
        'solver': 'learned_bounded',
        'matched': len(selected),
        'accepted': len(accepted_pids),
        'zero_reconstruction_mpjpe_mm': (
            zero_recon.cpu().numpy() * 1000.0
        ).astype(float).tolist(),
        'initial_target_mpjpe_mm': (
            init_err.cpu().numpy() * 1000.0
        ).astype(float).tolist(),
        'final_target_mpjpe_mm': (
            cand_err.cpu().numpy() * 1000.0
        ).astype(float).tolist(),
        'best_target_mpjpe_mm': (
            cand_err.cpu().numpy() * 1000.0
        ).astype(float).tolist(),
        'fit_improvement_ratio': gain.cpu().numpy().astype(float).tolist(),
        'precomp_root_drift_mm': root_comp_mm.cpu().numpy().astype(float).tolist(),
        'translation_compensation_mm': root_comp_mm.cpu().numpy().astype(float).tolist(),
        'postcomp_root_drift_mm': (
            compensated_root_drift_mm.cpu().numpy().astype(float).tolist()
        ),
        'best_step': [0 for _ in selected],
        'max_local_pose_delta_rad': (
            dn.max(dim=-1).values.cpu().numpy().astype(float).tolist()
        ),
        'root_delta_max': rd,
        'translation_compensation_param_delta_max': td,
        'shape_delta_max': sd,
        'expression_delta_max': ed,
        'hand_jaw_delta_max': hd,
        'max_postcomp_root_drift_mm': max_post_root_drift,
        'root_preservation_mode': 'analytic_translation_compensation',
        'steps_executed': 0,
        'lr': 0.0,
        'alpha': float(alpha),
    })

    if int(frame_id) == 0 or int(frame_id) % 100 == 0:
        print(
            '[WAO_J2S_LEARNED]',
            sequence,
            frame_id,
            'matched=', len(selected),
            'accepted=', len(accepted_pids),
            'zero_mm=', round(float(zero_recon.mean().cpu()) * 1000.0, 6),
            'target_mm=', round(float(init_err.mean().cpu()) * 1000.0, 3),
            'proposal_mm=', round(float(cand_err.mean().cpu()) * 1000.0, 3),
            'precomp_root_mm=', round(float(root_comp_mm.mean().cpu()), 3),
            'postcomp_root_mm=', round(float(compensated_root_drift_mm.mean().cpu()), 6),
            'steps=0',
            flush=True,
        )
    return bool(accepted_pids)


def refine_joint_to_smpl_frame(*, sequence, frame_id, image, pred,
        raw_pred_j3d_c, raw_pred_j2d, smpl_layer, smpl_model, K, pelvis_idx):
    solver = os.environ.get('WAO_J2S_SOLVER', 'learned_bounded').strip().lower()
    kwargs = dict(
        sequence=sequence,
        frame_id=frame_id,
        image=image,
        pred=pred,
        raw_pred_j3d_c=raw_pred_j3d_c,
        raw_pred_j2d=raw_pred_j2d,
        smpl_layer=smpl_layer,
        smpl_model=smpl_model,
        K=K,
        pelvis_idx=pelvis_idx,
    )
    if solver in {'iterative24', 'legacy24', 'iterative'}:
        return _refine_joint_to_smpl_frame_iterative24(**kwargs)
    if solver in {'learned_bounded', 'learned', 'proposal0'}:
        return _refine_joint_to_smpl_frame_learned(**kwargs)
    raise RuntimeError(f'unsupported WAO_J2S_SOLVER={solver!r}')
# ===== WAO_J2S_LEARNED_BOUNDED_END =====

