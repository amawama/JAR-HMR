"""Read-only official evaluation capture with nearest-valid-frame fallback for JAR-HMR.

The GPU evaluation still runs in a SINGLE pass. When an exact requested JAR-HMR
frame contains no mesh, its closest valid frame within the configured small
radius is used instead. The replacement is explicitly recorded in NPZ/JSON.
"""
from __future__ import annotations
import ast
import json
import os
import sys
from pathlib import Path
import numpy as np

H3R = Path('/root/data1/models/Human3R')
PRJ = Path('/root/data1/experiments/Human3R/wao_hmr')


def arr(v, dtype=None):
    if hasattr(v, 'detach'):
        v = v.detach().cpu()
    if hasattr(v, 'numpy'):
        v = v.numpy()
    return np.asarray(v, dtype=dtype)


def matrix(m, label):
    x = arr(m, np.float64)
    if x.ndim == 3 and x.shape[-2:] == (3, 3):
        if len(x) > 1 and not np.allclose(x, x[0], atol=1e-4):
            raise RuntimeError(label + '_DIFFERS_BY_PERSON')
        x = x[0]
    if x.shape != (3, 3) or not np.isfinite(x).all():
        raise RuntimeError(label + '_INVALID=' + str(x.shape))
    return x.astype(np.float32)


def build_candidate_mapping(image_paths, requested, radius):
    """Route nonoverlapping nearby real frames to the nearest planned index.

    The window is shortened for dense sampling to avoid having two planned
    frames compete for the same real candidate. No interpolation is permitted.
    """
    if radius < 0:
        raise ValueError('fallback-radius must be non-negative')
    if len(set(requested)) != len(requested):
        raise ValueError('requested frames must be unique')
    if requested[0] < 0 or requested[-1] >= len(image_paths):
        raise RuntimeError('OFFICIAL_INDEX_OUT_OF_BOUNDS=' + str(requested))
    min_gap = min(np.diff(requested)) if len(requested) > 1 else len(image_paths)
    effective = min(radius, max(0, int((min_gap - 1) // 2)))
    mapping = {}
    for center in requested:
        low, high = max(0, center - effective), min(len(image_paths) - 1, center + effective)
        for actual in range(low, high + 1):
            source = str(Path(image_paths[actual]).resolve())
            if source in mapping:
                prev_center, prev_actual = mapping[source]
                if abs(actual - center) >= abs(prev_actual - prev_center):
                    continue
            mapping[source] = (center, actual)
    return mapping, effective


def capture_official(args):
    os.chdir(H3R)
    for p in (str(H3R), str(H3R / 'src'), str(PRJ / 'method')):
        if p not in sys.path:
            sys.path.insert(0, p)
    from eval.global_human import launch as official
    seq = args.sequence
    frames = sorted(set(int(x) for x in args.frames.split(',') if x.strip()))
    if not frames:
        raise RuntimeError('EMPTY_FRAME_SELECTION')
    if args.mode == 'jar' and len(frames) < 1:
        raise RuntimeError('REQUIRES_AT_LEAST_ONE_SELECTED_FRAME')
    if args.mode == 'gt' and len(frames) != 1:
        raise RuntimeError('GT_REQUIRES_EXACTLY_ONE_FRAME')
    selected = {}
    found = set()
    resolved = {}
    original_get_seq = official.get_seq_list
    original_get_files = official.get_file_list
    effective_radius = 0

    def restricted_seqs(metadata, image_path):
        names, mapping, annotations = original_get_seq(metadata, image_path)
        if seq not in names:
            raise RuntimeError('OFFICIAL_SEQ_NOT_FOUND=' + seq)
        return [seq], mapping, annotations

    def restricted_files(metadata, image_path, name, mapping=None):
        nonlocal effective_radius
        image_paths, indices = original_get_files(metadata, image_path, name, mapping)
        radius = int(args.fallback_radius) if args.mode == 'jar' else 0
        candidate_map, effective_radius = build_candidate_mapping(image_paths, frames, radius)
        selected.update(candidate_map)
        for req in frames:
            source = Path(image_paths[req]).resolve()
            if not source.is_file():
                raise RuntimeError('OFFICIAL_FRAME_MISSING=' + str(source))
        if args.fast_window:
            low = max(0, frames[0] - effective_radius)
            high = min(len(image_paths), frames[-1] + effective_radius + 1)
        else:
            low, high = 0, min(len(image_paths), frames[-1] + effective_radius + 1)
        print('[OFFICIAL WINDOW]', seq, 'index', low, high, 'selection', frames,
              'fallback_radius', effective_radius, 'candidate_frames', len(selected), flush=True)
        return image_paths[low:high], indices[low:high]

    official.get_seq_list = restricted_seqs
    official.get_file_list = restricted_files
    target = Path(args.temp)
    target.mkdir(parents=True, exist_ok=True)
    exact_empty = set()

    def capture_visualize(save_dir, img_path, view, gt_v3d_c, pred_v3d_c,
                          K_to_proj, gt_K, bestMatch, smpl_face):
        source = str(Path(img_path).resolve())
        entry = selected.get(source)
        if entry is None:
            return
        planned, actual = entry
        # If the exact target already has at least two valid predicted meshes, no other candidate can improve it.
        if planned in resolved and abs(resolved[planned] - planned) <= abs(actual - planned):
            return
        if args.mode == 'jar':
            pred = arr(pred_v3d_c, np.float32)
            if pred.size == 0 or (pred.ndim == 3 and len(pred) < 2):
                if planned == actual:
                    exact_empty.add(planned)
                    print('[MISSING CENTER PAIR ON TARGET]', seq, 'frame', actual,
                          'predicted_people', len(pred) if pred.ndim else 0,
                          'searching nearby frames', flush=True)
                return
            if pred.ndim != 3 or pred.shape[-1] != 3:
                raise RuntimeError('PRED_MESH_INVALID=' + str(pred.shape))
            if not np.isfinite(pred).all():
                raise RuntimeError('PRED_MESH_NONFINITE_ON_FRAME=' + str(actual))
        else:
            pred = np.empty((0, 0, 3), np.float32)
        faces = arr(smpl_face, np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise RuntimeError('FACES_INVALID=' + str(faces.shape))
        if args.mode == 'jar' and faces.max() >= pred.shape[1]:
            raise RuntimeError('PRED_MESH_FACE_MISMATCH=' + str(actual))
        if args.mode == 'gt':
            gt = arr(gt_v3d_c, np.float32)
            if gt.ndim != 3 or gt.shape[-1] != 3 or len(gt) == 0 or faces.max() >= gt.shape[1]:
                raise RuntimeError('GT_MESH_MISSING_OR_INVALID=' + str(gt.shape))
        else:
            gt = np.empty((0, 0, 3), np.float32)
        if args.mode == 'gt':
            rgb = arr(view, np.float32)
            if rgb.ndim == 4 and rgb.shape[0] == 1:
                rgb = rgb[0]
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = rgb.transpose(1, 2, 0)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
                raise RuntimeError('PROCESSED_RGB_INVALID=' + str(rgb.shape))
            lo, hi = float(rgb.min()), float(rgb.max())
            if lo >= -1.1 and hi <= 1.1 and lo < -0.01:
                rgb = (rgb + 1.0) * 127.5
            elif lo >= 0.0 and hi <= 1.1:
                rgb *= 255.0
            elif lo >= 0.0 and hi <= 255.1:
                pass
            else:
                raise RuntimeError('PROCESSED_RGB_RANGE_INVALID')
            source_rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            source_rgb = np.empty((0, 0, 3), np.uint8)
        try:
            matches = arr(bestMatch).reshape(-1, 2).astype(np.int32) if args.mode == 'jar' else np.empty((0, 2), np.int32)
        except (TypeError, ValueError):
            matches = np.empty((0, 2), np.int32)
        np.savez_compressed(target / f'frame_{planned:05d}.npz',
                            pred_v3d=pred, source_path=source,
                            source_rgb=source_rgb, faces=faces, gt_v3d=gt,
                            best_match=matches,
                            planned_frame=np.asarray(planned, dtype=np.int32),
                            actual_frame=np.asarray(actual, dtype=np.int32),
                            K=matrix(K_to_proj, 'K'),
                            K_gt=matrix(gt_K, 'GT_K') if args.mode == 'gt' else np.eye(3, dtype=np.float32))
        found.add(planned)
        resolved[planned] = actual
        print('[CAPTURE]', args.mode, seq, 'planned', planned, 'actual', actual,
              'offset', actual - planned, 'JAR_PERSONS', len(pred),
              'GT_PERSONS', len(gt), flush=True)

    official.visualize = capture_visualize
    if official.eval_smpl_error.__globals__.get('visualize') is not capture_visualize:
        raise RuntimeError('OFFICIAL_VISUALIZE_HOOK_BINDING_CHANGED')
    source = Path(official.__file__).resolve()
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    entry = [node for node in tree.body if isinstance(node, ast.If)
             and isinstance(node.test, ast.Compare)
             and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__'
             and len(node.test.comparators) == 1
             and isinstance(node.test.comparators[0], ast.Constant)
             and node.test.comparators[0].value == '__main__']
    if len(entry) != 1:
        raise RuntimeError('OFFICIAL_MAIN_ENTRY_CHANGED')
    statement = ast.Module(body=entry[0].body, type_ignores=[])
    ast.fix_missing_locations(statement)
    sys.argv = [str(source), '--weights', './src/human3r_896L.pth',
                '--output_dir', str(target / 'official_temp_log'), '--eval_dataset', '3dpw',
                '--size', '512', '--reset_interval', '100', '--use_ttt3r', '--vis']
    print('[OFFICIAL METHOD] mode=', args.mode,
          'WAO_STAGE=', os.environ.get('WAO_HMR_STAGE', 'BASELINE_DISABLED'), flush=True)
    exec(compile(statement, str(source), 'exec'), official.__dict__)
    receipt = {'sequence': seq, 'mode': args.mode, 'requested': frames,
               'resolved': {str(k): resolved[k] for k in sorted(resolved)},
               'missing': sorted(set(frames) - found),
               'exact_target_empty': sorted(exact_empty),
               'fallback_radius_effective': effective_radius}
    (target / 'capture_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    missing = sorted(set(frames) - found)
    if missing:
        raise RuntimeError(f'NO_VALID_{args.mode.upper()}_MESH_WITHIN_RADIUS_'
                           f'{effective_radius}_FOR_REQUESTED_FRAMES={missing}')
    print('[OFFICIAL CAPTURE COMPLETE]', seq, args.mode,
          'resolved', [resolved[x] for x in frames], flush=True)
