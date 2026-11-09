"""
Unified WAO-HMR method entry.

Mainline stages integrated in this file:
- a2 / a2lite / a2a4: the verified body-only safe repair previously embedded
  in eval/global_human/launch.py.
- e4_window_legacy: the uploaded E4 V3 implementation, retained as a
  reproducibility stage.
- b0_candidate_audit: raw + A2 + E4-window candidate audit. It never changes
  official prediction and is the first B-group unification/parity stage.
- b1_visible_generate / b2_bbox_generate: generate compact body-pose
  hypotheses from two observation-guided image branches while keeping the
  official evaluator and model unchanged.
- b_multisource_audit: raw + A2 + E4-window + visible + bbox diversity audit.
- b3_multisource_precision_audit: GT-only offline diagnostic of candidate
  precision/oracle potential. It never changes official prediction.
- b6_consensus_fusion: no-GT conservative local fusion. A2 is the primary
  repair candidate; bbox and strict E4-window are independent support.
- b6_expanded_consensus: B6-v1 dev stage. It adds A2-lite, uses a higher-recall
  E4 candidate mode restricted to alpha 0.25/0.50, and ranks proposals by
  evidence strength rather than edit magnitude. The visible proxy remains
  excluded because B3 showed negative average 3D effect.
- b1_sam3_context_generate: build a real SAM3 mask-guided context candidate
  from existing 3DPW masks without changing coordinates or intrinsics.
- b3_sam3_precision_audit: GT-only candidate-space diagnosis for raw, A2,
  A2-lite, bbox, high-recall window and SAM3-context hypotheses.
- b5_person_mask_constraint_audit: GT-only diagnosis of whether per-person
  SAM3 mask alignment can select among raw/A2/A2-lite/bbox/window candidates.
  It never changes official prediction and never uses the failed SAM3-context
  input branch as a repair candidate.
- b6_mask_aware_consensus: no-GT hybrid. It starts from B6-v1 consensus,
  uses per-person SAM3 mask loss only to confirm/reject proposals, and permits
  conservative mask-supported fallback when consensus leaves a slot empty.
- g0_world_physics_audit: GT-only development audit of camera/world 3D
  geometry, causal world-motion consequences, joint-motion proxies and 2D/3D
  ambiguity for raw/A2/A2-lite/bbox/window candidates. It never modifies pred.
- h0_strong_local_oracle_audit: generates strong, chain-local SMPL hypotheses
  from robust 2D fitting, temporal/world-motion initialization, and a damped
  Jacobian direction that changes distal depth while minimally perturbing 2D.
  It is GT-only for oracle analysis and never modifies official prediction.
- h0v2_high_recall_oracle_audit: preserves the validated H0 optimizer but
  replaces the low-recall locator with a calibrated multi-evidence OR score,
  permits two person-chain proposals per frame, uses camera-depth hypotheses
  only for arms, and uses local knee-bend-plane hypotheses for legs. It also
  audits per-variant rejection reasons and camera/world rigid invariants.

Only pred["rotvec"][..., 1:22] may be changed by mainline repair stages.
Root orientation, translation, location, shape, expression, K, T_c2w and
camera state are frozen.

Legacy C/E/Z diagnostics are dispatched through compatibility imports, so
their old files can remain until the unified mainline has passed parity tests.
"""

from __future__ import annotations

import csv
import importlib
import itertools
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

METHOD_DIR = Path("/root/data1/experiments/Human3R/wao_hmr/method")
DEFAULT_UNIFIED_ROOT = Path(
    "/root/data1/experiments/Human3R/wao_hmr/results_safe/unified"
)

B_AUDIT_FIELDS = [
    "sequence", "frame_id", "stage", "n_persons",
    "candidate_names", "candidate_count",
    "a2_body_rms", "a2_body_max",
    "a2lite_body_rms", "a2lite_body_max", "a2_a2lite_rms",
    "window_available", "window_selected_pid",
    "window_selected_chain", "window_selected_alpha",
    "window_body_rms", "window_body_max",
    "a2_window_rms", "visible_available", "bbox_available", "sam3_available",
    "visible_body_rms", "bbox_body_rms", "sam3_body_rms",
    "visible_matched", "visible_branch_people",
    "bbox_matched", "bbox_branch_people",
    "sam3_matched", "sam3_branch_people",
    "max_root_delta", "frozen_contract_ok",
    "a2_frame_class",
    "window_rejection_reason", "window_reprojection_improvement",
    "window_temporal_acceleration_improvement",
    "b6_modified_persons", "b6_selected_sources",
    "b6_mean_body_rms", "b6_max_body_delta",
    "h0_proposals", "h0_candidates", "h0_variant_names",
]

B6_SELECTION_FIELDS = [
    "sequence", "frame_id", "pid", "a2_frame_class",
    "a2_variant", "evidence_score",
    "selected_sources", "modified", "body_rms", "body_max",
    "raw_reprojection", "a2_reprojection", "a2lite_reprojection",
    "bbox_reprojection",
    "window_reprojection", "a2_a2lite_agreement",
    "a2_bbox_agreement", "a2lite_bbox_agreement",
    "a2_window_agreement", "a2lite_window_agreement",
    "bbox_window_agreement",
    "a2_weight", "bbox_weight", "window_weight",
    "window_alpha", "window_reprojection_improvement",
    "window_temporal_acceleration_improvement",
    "mask_matched", "raw_mask_loss", "selected_mask_loss",
    "mask_loss_delta", "mask_decision", "mask_candidate",
    "frozen_contract_ok",
]

B_PRECISION_FIELDS = [
    "sequence", "frame_id", "pred_id", "gt_id", "candidate",
    "candidate_available", "candidate_changed",
    "a2_frame_class", "body_rms_from_raw", "body_max_from_raw",
    "consensus_rms", "external_reprojection_error",
    "window_selected_chain", "window_selected_alpha",
    "window_reprojection_improvement",
    "window_temporal_acceleration_improvement",
    "mpjpe", "pa_mpjpe", "pve", "pa_pve",
    "metric_mpjpe", "metric_pve",
    "delta_mpjpe_from_raw", "delta_pve_from_raw",
    "delta_pa_mpjpe_from_raw", "delta_pa_pve_from_raw",
    "frozen_contract_ok",
]


B5_MASK_FIELDS = [
    "sequence", "frame_id", "pred_id", "gt_id", "candidate",
    "candidate_changed", "mask_matched", "mask_file",
    "mask_fraction", "mask_bbox_iou", "mask_joint_inside_ratio",
    "mask_vertex_outside_ratio", "mask_vertex_outside_mean",
    "mask_vertex_outside_p90", "mask_loss", "mask_loss_delta_from_raw",
    "body_rms_from_raw", "body_max_from_raw", "consensus_rms",
    "external_reprojection_error", "a2_frame_class",
    "window_selected_chain", "window_selected_alpha",
    "window_reprojection_improvement",
    "window_temporal_acceleration_improvement",
    "mpjpe", "pa_mpjpe", "pve", "pa_pve",
    "metric_mpjpe", "metric_pve",
    "delta_mpjpe_from_raw", "delta_pve_from_raw",
    "delta_pa_mpjpe_from_raw", "delta_pa_pve_from_raw",
    "frozen_contract_ok",
]

G0_WORLD_FIELDS = [
    "sequence", "frame_id", "pred_id", "gt_id", "candidate",
    "candidate_changed", "a2_frame_class", "temporal_history",
    "body_rms_from_raw", "body_max_from_raw",
    "camera_joint_rms_from_raw_mm", "camera_joint_max_from_raw_mm",
    "pelvis_relative_joint_rms_from_raw_mm",
    "world_joint_rms_from_raw_mm", "world_endpoint_rms_from_raw_mm",
    "mean_depth_delta_from_raw_mm", "max_depth_delta_from_raw_mm",
    "pairwise_2d_rms_from_raw_px", "pairwise_2d_rms_from_raw_norm",
    "inserted_world_velocity_mm", "raw_world_velocity_mm",
    "world_velocity_excess_mm", "inserted_world_acceleration_mm",
    "raw_world_acceleration_mm", "world_acceleration_excess_mm",
    "inserted_world_jerk_mm", "raw_world_jerk_mm",
    "world_jerk_excess_mm", "inserted_endpoint_velocity_mm",
    "raw_endpoint_velocity_mm", "endpoint_velocity_excess_mm",
    "inserted_endpoint_acceleration_mm", "raw_endpoint_acceleration_mm",
    "endpoint_acceleration_excess_mm", "depth_jump_excess_mm",
    "limb_angle_jump_deg", "raw_limb_angle_jump_deg",
    "limb_angle_jump_excess_deg", "limb_flip_ratio",
    "joint_rotvec_norm_mean", "joint_rotvec_norm_max",
    "joint_limit_excess_ratio", "physics_penalty_proxy",
    "external_reprojection_error", "consensus_rms",
    "window_selected_chain", "window_selected_alpha",
    "window_reprojection_improvement",
    "window_temporal_acceleration_improvement",
    "mpjpe", "pa_mpjpe", "pve", "pa_pve",
    "metric_mpjpe", "metric_pve", "world_mpjpe", "world_pve",
    "world_root_error", "delta_mpjpe_from_raw", "delta_pve_from_raw",
    "delta_pa_mpjpe_from_raw", "delta_pa_pve_from_raw",
    "delta_world_mpjpe_from_raw", "delta_world_pve_from_raw",
    "frozen_contract_ok",
]

G0_PAIRWISE_FIELDS = [
    "sequence", "frame_id", "pred_id", "gt_id",
    "candidate_a", "candidate_b", "pair_2d_rms_px", "pair_2d_rms_norm",
    "pair_camera_pelvis_relative_rms_mm", "pair_world_joint_rms_mm",
    "pair_world_endpoint_rms_mm", "near_2d_0025", "near_2d_005",
    "near_2d_01", "different_3d_20mm", "different_3d_40mm",
    "different_3d_60mm", "candidate_a_delta_mpjpe",
    "candidate_b_delta_mpjpe", "candidate_a_delta_pve",
    "candidate_b_delta_pve", "frozen_contract_ok",
]

_SEQUENCE_STATES: Dict[str, Dict[str, Any]] = {}
_B_FRAME_STATS: Dict[Tuple[str, int], Dict[str, Any]] = {}
_B_FRAME_CANDIDATES: Dict[Tuple[str, int], Dict[str, Any]] = {}
_B6_FRAME_SELECTIONS: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
_EXTERNAL_BRANCH_CACHE: Dict[Tuple[str, str, str], Any] = {}
_SAM3_SEQUENCE_INDEX: Dict[Tuple[str, str], Dict[int, List[Path]]] = {}
_SAM3_ROOT_PNG_CACHE: Dict[str, List[Path]] = {}
_SAM3_MASK_CACHE: Dict[Tuple[str, str, int, int, int], Any] = {}
_SAM3_INSTANCE_CACHE: Dict[Tuple[str, str, int, int, int], Any] = {}
_G0_RAW_HISTORY: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}


def _stage() -> str:
    return os.environ.get("WAO_HMR_STAGE", "").strip().lower()


def _run_root() -> Path:
    root = Path(os.environ.get(
        "WAO_HMR_UNIFIED_OUT_ROOT",
        str(DEFAULT_UNIFIED_ROOT),
    ))
    tag = os.environ.get("WAO_HMR_RUN_TAG", "unified_run")
    return root / tag


def _parse_seq_name(path_like: Any) -> str:
    s = str(path_like)
    parts = s.split(os.sep)
    if "imageFiles" in parts:
        idx = parts.index("imageFiles")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(s).parent.name


def _clone_value(x: Any):
    if torch.is_tensor(x):
        return x.detach().clone()
    return np.array(x, copy=True)


def _clone_rotvec_list(values: Iterable[Any]) -> List[Any]:
    return [_clone_value(x) for x in values]


def _safe_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _value_norm(x: Any) -> float:
    if torch.is_tensor(x):
        return float(torch.linalg.norm(x).detach().cpu())
    return float(np.linalg.norm(np.asarray(x)))


def _body_delta_stats(candidate: Any, raw: Any) -> Tuple[float, float]:
    if candidate is None or raw is None:
        return float("nan"), float("nan")
    c = _safe_numpy(candidate)
    r = _safe_numpy(raw)
    if c.ndim != 3 or r.ndim != 3 or c.shape != r.shape:
        return float("nan"), float("nan")
    delta = c[:, 1:22] - r[:, 1:22]
    if delta.size == 0:
        return 0.0, 0.0
    joint_norm = np.linalg.norm(delta, axis=-1)
    return float(np.sqrt(np.mean(delta ** 2))), float(joint_norm.max())


def _root_delta_max(candidate: Any, raw: Any) -> float:
    if candidate is None or raw is None:
        return float("nan")
    c = _safe_numpy(candidate)
    r = _safe_numpy(raw)
    if c.ndim != 3 or r.ndim != 3 or c.shape != r.shape or c.shape[1] < 1:
        return float("nan")
    if c.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(c[:, 0] - r[:, 0], axis=-1).max())


def _copy_pred_with_frame_pose(pred: Dict[str, Any], frame_id: int) -> Dict[str, Any]:
    candidate = dict(pred)
    candidate["rotvec"] = list(pred["rotvec"])
    candidate["rotvec"][frame_id] = _clone_value(pred["rotvec"][frame_id])
    return candidate

from collections import Counter
import csv
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

_UNIFIED_DEFAULT_ROOT = Path(
    "/root/data1/experiments/Human3R/wao_hmr/results_safe/unified"
)
_UNIFIED_RUN_TAG = os.environ.get("WAO_HMR_RUN_TAG", "unified_run")
OUT_DIR = Path(os.environ.get(
    "WAO_HMR_UNIFIED_OUT_ROOT",
    str(_UNIFIED_DEFAULT_ROOT),
)) / _UNIFIED_RUN_TAG
OBS_PATH = Path(os.environ.get(
    "WAO_HMR_E4_OBS_PATH",
    "/root/data1/experiments/Human3R/wao_hmr/assets/e4_2d_observations.pkl",
))
CSV_PATH = OUT_DIR / "e4_legacy_audit.csv" 

VIS_THR = 0.30
DET_SCORE_THR = 0.75
MATCH_IOU_THR = 0.55
MATCH_CENTER_THR = 0.04
TRACK_IOU_THR = 0.20
TRACK_CENTER_THR = 0.08

OPT_STEPS = 4
OPT_LR = 0.05
POSE_PRIOR_WEIGHT = 0.50
AMPLITUDES = (0.25, 0.50, 0.75, 1.00)
MAX_MODIFIED_RATIO = 0.03
MIN_MODIFIED_GAP = 2
RAW_COST = 1.0
MIN_COST_MARGIN = 0.05

# V3 acceptance gates: temporal acceleration must still improve, but a
# candidate may pass through either a relative or an absolute improvement
# margin. Small/medium amplitudes use slightly lower 2D and cost margins;
# velocity, exact projection and raw-pose fallback remain hard safeguards.
MIN_HISTORY_DEPTH = 2
MIN_RAW_ACCEL_FOR_REPAIR = 0.020
ACCEL_MIN_REL_DROP = 0.02
ACCEL_MIN_ABS_DROP = 0.0015
VELOCITY_REL_LIMIT = 1.20
VELOCITY_ABS_MARGIN = 0.012
MIN_APPLIED_DELTA = 0.025
DIRECTION_WORSEN_TOL = 0.005
ALPHA_REPROJECTION_MIN = {
    0.25: 0.10,
    0.50: 0.13,
    0.75: 0.17,
    1.00: 0.22,
}
ALPHA_COST_MARGIN = {
    0.25: 0.05,
    0.50: 0.07,
    0.75: 0.10,
    1.00: 0.15,
}

# Official J14 order:
# 0 RAnkle, 1 RKnee, 2 RHip, 3 LHip, 4 LKnee, 5 LAnkle,
# 6 RWrist, 7 RElbow, 8 RShoulder,
# 9 LShoulder, 10 LElbow, 11 LWrist, 12 Neck, 13 Head
CHAINS = {
    'left_arm': {
        'coco': [5, 7, 9], 'j14': [9, 10, 11], 'pose': [16, 18],
        'conf': 0.68, 'err_min': 0.060, 'err_max': 0.220,
        'improve': 0.10, 'delta_max': 0.150, 'risk': 1.20,
        'kind': 'arm',
    },
    'right_arm': {
        'coco': [6, 8, 10], 'j14': [8, 7, 6], 'pose': [17, 19],
        'conf': 0.68, 'err_min': 0.060, 'err_max': 0.220,
        'improve': 0.10, 'delta_max': 0.150, 'risk': 1.20,
        'kind': 'arm',
    },
    'left_leg': {
        'coco': [11, 13, 15], 'j14': [3, 4, 5], 'pose': [1, 4],
        'conf': 0.78, 'err_min': 0.085, 'err_max': 0.220,
        'improve': 0.18, 'delta_max': 0.100, 'risk': 0.75,
        'kind': 'leg',
    },
    'right_leg': {
        'coco': [12, 14, 16], 'j14': [2, 1, 0], 'pose': [2, 5],
        'conf': 0.78, 'err_min': 0.085, 'err_max': 0.220,
        'improve': 0.18, 'delta_max': 0.100, 'risk': 0.75,
        'kind': 'leg',
    },
}

FIELDS = [
    'sequence', 'frame_id', 'n_human3r', 'n_external', 'n_matched',
    'candidate_chains', 'selected_pid', 'selected_chain', 'selected_alpha', 'track_id',
    'history_depth', 'raw_reprojection_error', 'final_reprojection_error',
    'relative_reprojection_improvement',
    'required_reprojection_improvement',
    'raw_direction_error', 'final_direction_error',
    'raw_temporal_velocity', 'recent_temporal_velocity',
    'final_temporal_velocity', 'velocity_limit',
    'raw_temporal_acceleration', 'final_temporal_acceleration',
    'temporal_acceleration_improvement', 'candidate_cost',
    'rejection_reason', 'base_delta_max', 'applied_delta_max',
    'clamp_ratio', 'amplitude_costs', 'amplitude_gates',
    'amplitude_failure_counts', 'amplitude_pass_count',
    'sequence_modified_frames', 'pred_modified',
]

_OBS = None
_FRAME_STATS = {}
_MODIFIED = {}
_LAST_MODIFIED = {}
_TRACKS = {}
_NEXT_TRACK = {}


def _np(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _load_obs():
    global _OBS
    if _OBS is None:
        with OBS_PATH.open('rb') as f:
            _OBS = pickle.load(f)['sequences']
        print(f'[E4-T-V3] observations loaded: {len(_OBS)} sequences', flush=True)
    return _OBS


def _bbox(points, conf=None):
    points = np.asarray(points)
    valid = np.isfinite(points[:, :2]).all(axis=1)
    if conf is not None:
        valid &= np.asarray(conf) >= VIS_THR
    if valid.sum() < 3:
        return None
    xy = points[valid, :2]
    x1, y1 = xy.min(axis=0)
    x2, y2 = xy.max(axis=0)
    w, h = max(float(x2-x1), 1.0), max(float(y2-y1), 1.0)
    return np.array([x1-.08*w, y1-.08*h, x2+.08*w, y2+.08*h], np.float32)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    ab = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return float(inter / (aa + ab - inter)) if aa + ab - inter > 0 else 0.0


def _center(a, b, width, height):
    ca = np.array([(a[0]+a[2])/2, (a[1]+a[3])/2])
    cb = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
    return float(np.linalg.norm(ca-cb) / max(np.hypot(width, height), 1.0))


def _chain_error(points, target, conf, scale):
    weights = conf / max(float(conf.sum()), 1e-6)
    return float(np.sum(weights * np.linalg.norm(points-target, axis=1)) / max(scale, 1.0))


def _direction_error(points, target):
    values = []
    for a, b in ((0, 1), (1, 2)):
        u, v = points[b]-points[a], target[b]-target[a]
        denom = max(float(np.linalg.norm(u)*np.linalg.norm(v)), 1e-6)
        cosine = float(np.clip(np.dot(u, v)/denom, -1.0, 1.0))
        values.append(1.0-cosine)
    return float(np.mean(values))


def _rms(a, b):
    return float(np.sqrt(np.mean(np.square(np.asarray(a)-np.asarray(b)))))


def _temporal(chain_pose, history):
    if history['depth'] == 0:
        return 0.0, 0.0, 0.0
    velocity = _rms(chain_pose, history['last'])
    if history['depth'] < 2:
        return velocity, 0.0, 0.0
    recent_velocity = _rms(history['last'], history['prev'])
    acceleration = _rms(
        chain_pose-history['last'],
        history['last']-history['prev'],
    )
    return velocity, acceleration, recent_velocity


def _detections(sequence, frame_id, height, width):
    output = []
    for item in _load_obs().get(sequence, {}).get(int(frame_id), []):
        kp = np.asarray(item['kp2d'], np.float32).copy()
        res = np.asarray(item['res'], np.float32).reshape(-1)
        if len(res) < 2:
            continue
        kp[:, 0] *= width / max(float(res[0]), 1.0)
        kp[:, 1] *= height / max(float(res[1]), 1.0)
        score = float(item.get('score', 0.0))
        box = _bbox(kp, kp[:, 2])
        if score >= DET_SCORE_THR and box is not None:
            output.append({'kp': kp, 'bbox': box, 'score': score})
    return output


def _match_external(pred_j2d, detections, height, width):
    pboxes = [_bbox(x) for x in pred_j2d]
    pids = [i for i, box in enumerate(pboxes) if box is not None]
    dids = list(range(len(detections)))
    if not pids or not dids:
        return []
    cost = np.full((len(pids), len(dids)), 100.0, np.float32)
    meta = {}
    for r, pid in enumerate(pids):
        for c, did in enumerate(dids):
            iou = _iou(pboxes[pid], detections[did]['bbox'])
            center = _center(pboxes[pid], detections[did]['bbox'], width, height)
            cost[r, c] = 1.0-iou + 2.5*center
            meta[r, c] = iou, center
    rows, cols = linear_sum_assignment(cost)
    result = []
    for r, c in zip(rows, cols):
        iou, center = meta[r, c]
        if iou >= MATCH_IOU_THR and center <= MATCH_CENTER_THR:
            result.append({'pid': pids[r], 'did': dids[c], 'iou': iou})
    return result


def _associate_tracks(sequence, frame_id, pred_j2d, width, height):
    boxes = [_bbox(x) for x in pred_j2d]
    tracks = _TRACKS.setdefault(sequence, {})
    assignments = [None] * len(boxes)
    active = [tid for tid, state in tracks.items() if int(frame_id)-state['frame'] == 1]
    current = [pid for pid, box in enumerate(boxes) if box is not None]
    if active and current:
        cost = np.full((len(current), len(active)), 100.0, np.float32)
        meta = {}
        for r, pid in enumerate(current):
            for c, tid in enumerate(active):
                iou = _iou(boxes[pid], tracks[tid]['bbox'])
                center = _center(boxes[pid], tracks[tid]['bbox'], width, height)
                cost[r, c] = 1.0-iou + 2.0*center
                meta[r, c] = iou, center
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            iou, center = meta[r, c]
            if iou >= TRACK_IOU_THR and center <= TRACK_CENTER_THR:
                assignments[current[r]] = active[c]
    next_id = _NEXT_TRACK.setdefault(sequence, 0)
    for pid, box in enumerate(boxes):
        if box is not None and assignments[pid] is None:
            assignments[pid] = next_id
            next_id += 1
    _NEXT_TRACK[sequence] = next_id
    return assignments, boxes


def _history(sequence, track_id, frame_id, pose_ids):
    state = _TRACKS.get(sequence, {}).get(track_id)
    if state is None or state['frame'] != int(frame_id)-1:
        return {'depth': 0, 'last': None, 'prev': None}
    depth = 1
    prev = None
    if state.get('prev_pose') is not None and state.get('prev_frame') == int(frame_id)-2:
        depth = 2
        prev = state['prev_pose'][pose_ids]
    return {'depth': depth, 'last': state['pose'][pose_ids], 'prev': prev}


def _update_tracks(sequence, frame_id, assignments, boxes, pose):
    tracks = _TRACKS.setdefault(sequence, {})
    pose = _np(pose)
    for pid, tid in enumerate(assignments):
        if tid is None or boxes[pid] is None:
            continue
        old = tracks.get(tid)
        tracks[tid] = {
            'bbox': boxes[pid].copy(),
            'pose': pose[pid].copy(),
            'frame': int(frame_id),
            'prev_pose': None if old is None else old['pose'],
            'prev_frame': None if old is None else old['frame'],
        }


def _project(pose, pred, frame_id, smpl_layer, smpl_model, K):
    count = pose.shape[0]
    Kb = K.unsqueeze(0).expand(count, -1, -1)
    out = smpl_layer(
        pose, pred['shape'][frame_id], pred['transl'][frame_id],
        None, None, K=Kb, expression=pred['expression'][frame_id],
    )
    vertices = smpl_model.smplx2smpl.to(out['smpl_v3d']) @ out['smpl_v3d']
    joints = smpl_model.j_regressor.to(vertices) @ vertices
    normalized = joints / joints[..., 2:3].clamp_min(1e-4)
    return torch.einsum('bij,bkj->bki', Kb, normalized)[..., :2]


def _blank(n):
    return {
        'n_human3r': int(n), 'n_external': 0, 'n_matched': 0,
        'candidate_chains': 0, 'selected_pid': '', 'selected_chain': '', 'selected_alpha': '',
        'track_id': '', 'history_depth': 0, 'raw_reprojection_error': '',
        'final_reprojection_error': '', 'relative_reprojection_improvement': '',
        'raw_direction_error': '', 'final_direction_error': '',
        'raw_temporal_velocity': '', 'recent_temporal_velocity': '',
        'final_temporal_velocity': '', 'velocity_limit': '',
        'raw_temporal_acceleration': '', 'final_temporal_acceleration': '',
        'temporal_acceleration_improvement': '',
        'required_reprojection_improvement': '', 'candidate_cost': '',
        'rejection_reason': '', 'amplitude_gates': '',
        'amplitude_failure_counts': '', 'amplitude_pass_count': 0,
        'base_delta_max': 0.0, 'applied_delta_max': 0.0,
        'clamp_ratio': 0.0, 'amplitude_costs': '',
        'sequence_modified_frames': 0, 'pred_modified': False,
    }


def _e4_refine_online_frame(
    *, sequence, frame_id, image, pred, raw_pred_j2d,
    smpl_layer, smpl_model, K, candidate_mode=False,
):
    key = (sequence, int(frame_id))
    raw_pose = pred['rotvec'][frame_id].detach()
    raw_j2d = _np(raw_pred_j2d)
    height, width = image.shape[-2:]
    stats = _blank(raw_pose.shape[0])

    detections = _detections(sequence, frame_id, height, width)
    matches = _match_external(raw_j2d, detections, height, width)
    assignments, boxes = _associate_tracks(sequence, frame_id, raw_j2d, width, height)
    stats['n_external'], stats['n_matched'] = len(detections), len(matches)

    seq_len = max(len(_load_obs().get(sequence, {})), 1)
    max_modified = max(1, int(math.ceil(seq_len*MAX_MODIFIED_RATIO)))
    modified = _MODIFIED.get(sequence, 0)
    last_modified = _LAST_MODIFIED.get(sequence, -100000)
    if (not candidate_mode) and (
        modified >= max_modified
        or int(frame_id)-last_modified < MIN_MODIFIED_GAP
    ):
        stats['rejection_reason'] = 'budget_or_min_gap'
        _update_tracks(sequence, frame_id, assignments, boxes, raw_pose)
        stats['sequence_modified_frames'] = modified
        _FRAME_STATS[key] = stats
        return False

    candidates = []
    for match in matches:
        pid, det = match['pid'], detections[match['did']]
        diagonal = max(float(np.hypot(det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1])), 1.0)
        for name, chain in CHAINS.items():
            conf = det['kp'][chain['coco'], 2]
            if np.any(conf < chain['conf']):
                continue
            target = det['kp'][chain['coco'], :2]
            points = raw_j2d[pid, chain['j14']]
            error = _chain_error(points, target, conf, diagonal)
            if not chain['err_min'] <= error <= chain['err_max']:
                continue
            direction = _direction_error(points, target)
            risk = error * float(conf.mean()) * match['iou'] * det['score'] * chain['risk']
            candidates.append({
                'pid': pid, 'name': name, 'chain': chain, 'target': target,
                'conf': conf, 'scale': diagonal, 'raw_error': error,
                'raw_direction': direction, 'risk': risk,
            })
    stats['candidate_chains'] = len(candidates)
    if not candidates:
        stats['rejection_reason'] = 'no_reliable_chain_candidate'
        _update_tracks(sequence, frame_id, assignments, boxes, raw_pose)
        stats['sequence_modified_frames'] = modified
        _FRAME_STATS[key] = stats
        return False

    item = max(candidates, key=lambda x: x['risk'])
    pid, chain = item['pid'], item['chain']
    pose_ids, joint_ids = chain['pose'], chain['j14']
    track_id = assignments[pid]
    history = _history(sequence, track_id, frame_id, pose_ids)
    stats.update({
        'selected_pid': pid, 'selected_chain': item['name'], 'track_id': track_id,
        'history_depth': history['depth'],
        'raw_reprojection_error': item['raw_error'],
        'raw_direction_error': item['raw_direction'],
    })

    # V2: every accepted modification needs two causal history frames.
    # Without them, temporal plausibility cannot be evaluated reliably.
    if history['depth'] < MIN_HISTORY_DEPTH:
        stats['rejection_reason'] = 'insufficient_causal_history'
        _update_tracks(sequence, frame_id, assignments, boxes, raw_pose)
        stats['sequence_modified_frames'] = modified
        _FRAME_STATS[key] = stats
        return False

    raw_chain = _np(raw_pose[pid, pose_ids])
    raw_vel, raw_acc, recent_vel = _temporal(raw_chain, history)
    stats['raw_temporal_velocity'] = raw_vel
    stats['recent_temporal_velocity'] = recent_vel
    stats['raw_temporal_acceleration'] = raw_acc

    # Do not perturb a chain that is already temporally stable.
    min_raw_acc = (
        float(os.environ.get('WAO_HMR_B6V1_WINDOW_MIN_RAW_ACC', '0.012'))
        if candidate_mode else MIN_RAW_ACCEL_FOR_REPAIR
    )
    if raw_acc < min_raw_acc:
        stats['rejection_reason'] = 'raw_chain_temporally_stable'
        _update_tracks(sequence, frame_id, assignments, boxes, raw_pose)
        stats['sequence_modified_frames'] = modified
        _FRAME_STATS[key] = stats
        return False

    delta = torch.zeros((len(pose_ids), 3), dtype=raw_pose.dtype, device=raw_pose.device, requires_grad=True)
    target = torch.as_tensor(item['target'], dtype=raw_pose.dtype, device=raw_pose.device)
    weights = torch.as_tensor(item['conf'], dtype=raw_pose.dtype, device=raw_pose.device)
    weights = weights / weights.sum().clamp_min(1e-6)
    optimizer = torch.optim.Adam([delta], lr=OPT_LR)

    for _ in range(OPT_STEPS):
        optimizer.zero_grad()
        full_delta = torch.zeros_like(raw_pose)
        full_delta[pid, pose_ids] = delta
        projected = _project(raw_pose+full_delta, pred, frame_id, smpl_layer, smpl_model, K)
        residual = (projected[pid, joint_ids]-target) / item['scale']
        joint_error = torch.linalg.norm(residual, dim=-1)
        reproj = torch.sum(weights * F.smooth_l1_loss(
            joint_error, torch.zeros_like(joint_error), reduction='none', beta=0.02,
        ))
        loss = reproj + POSE_PRIOR_WEIGHT*torch.mean(delta.square())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-chain['delta_max'], chain['delta_max'])

    base_delta = delta.detach()
    base_max = float(base_delta.abs().max().item())
    stats['base_delta_max'] = base_max
    stats['clamp_ratio'] = base_max / max(chain['delta_max'], 1e-6)
    min_base_delta = (
        float(os.environ.get('WAO_HMR_B6V1_WINDOW_MIN_BASE_DELTA', '0.008'))
        if candidate_mode else 0.012
    )
    if base_max < min_base_delta:
        stats['rejection_reason'] = 'refit_direction_too_small'
        _update_tracks(sequence, frame_id, assignments, boxes, raw_pose)
        stats['sequence_modified_frames'] = modified
        _FRAME_STATS[key] = stats
        return False

    with torch.no_grad():
        full_delta = torch.zeros_like(raw_pose)
        full_delta[pid, pose_ids] = base_delta
        full_points = _np(_project(raw_pose+full_delta, pred, frame_id, smpl_layer, smpl_model, K)[pid, joint_ids])

    raw_points = raw_j2d[pid, joint_ids]
    valid = []
    cost_text = []
    gate_text = []
    failure_counts = Counter()

    # V3: final acceleration must always be lower than raw. In addition,
    # either the relative drop or the absolute drop must reach its margin.
    acceleration_limit = raw_acc
    velocity_limit = max(
        raw_vel + VELOCITY_ABS_MARGIN,
        recent_vel * VELOCITY_REL_LIMIT + VELOCITY_ABS_MARGIN,
    )
    stats['velocity_limit'] = velocity_limit

    amplitudes = (0.25, 0.50) if candidate_mode else AMPLITUDES
    candidate_reproj_min = {0.25: 0.06, 0.50: 0.08}
    candidate_cost_margin = {0.25: 0.02, 0.50: 0.03}
    accel_rel_req = 0.01 if candidate_mode else ACCEL_MIN_REL_DROP
    accel_abs_req = 0.0008 if candidate_mode else ACCEL_MIN_ABS_DROP
    direction_tol = 0.010 if candidate_mode else DIRECTION_WORSEN_TOL
    min_applied_delta = 0.010 if candidate_mode else MIN_APPLIED_DELTA

    for alpha in amplitudes:
        points = raw_points + alpha*(full_points-raw_points)
        error = _chain_error(points, item['target'], item['conf'], item['scale'])
        improvement = (item['raw_error']-error) / max(item['raw_error'], 1e-6)
        direction = _direction_error(points, item['target'])
        candidate_chain = raw_chain + alpha*_np(base_delta)
        velocity, acceleration, _ = _temporal(candidate_chain, history)
        applied_delta = alpha * base_max

        if candidate_mode:
            required_improvement = max(
                chain['improve'] * 0.55,
                candidate_reproj_min[alpha],
            )
        else:
            required_improvement = max(
                chain['improve'],
                ALPHA_REPROJECTION_MIN[alpha],
            )
        reasons = []
        if improvement < required_improvement:
            reasons.append('reprojection')
        if direction > item['raw_direction'] + direction_tol:
            reasons.append('direction')
        acceleration_abs_drop = raw_acc - acceleration
        acceleration_rel_drop = acceleration_abs_drop / max(raw_acc, 1e-6)
        acceleration_pass = (
            acceleration < acceleration_limit
            and (
                acceleration_rel_drop >= accel_rel_req
                or acceleration_abs_drop >= accel_abs_req
            )
        )
        if not acceleration_pass:
            reasons.append('acceleration')
        if velocity > velocity_limit:
            reasons.append('velocity')
        if applied_delta < min_applied_delta:
            reasons.append('delta_small')

        temporal_term = float(np.clip(
            (acceleration-raw_acc) / max(raw_acc, 0.03),
            -1.0,
            2.0,
        ))
        velocity_term = min(
            max(0.0, (velocity-velocity_limit)/0.05),
            2.0,
        )
        cost = (
            error/max(item['raw_error'], 1e-6)
            + 0.45*temporal_term
            + 0.20*velocity_term
            + 0.08*(applied_delta/max(chain['delta_max'], 1e-6))
        )
        cost_text.append(f'{alpha:.2f}:{cost:.4f}')

        if candidate_mode:
            required_margin = candidate_cost_margin[alpha]
        else:
            required_margin = max(
                MIN_COST_MARGIN,
                ALPHA_COST_MARGIN[alpha],
            )
        if cost > RAW_COST-required_margin:
            reasons.append('cost_margin')

        if reasons:
            failure_counts.update(reasons)
            gate_text.append(
                f"{alpha:.2f}:reject[{','.join(reasons)}]"
            )
            continue

        gate_text.append(f'{alpha:.2f}:pass')
        valid.append({
            'alpha': alpha,
            'cost': cost,
            'error': error,
            'improvement': improvement,
            'direction': direction,
            'velocity': velocity,
            'acceleration': acceleration,
            'required_improvement': required_improvement,
            'applied_delta': applied_delta,
        })

    stats['amplitude_costs'] = ';'.join(cost_text)
    stats['amplitude_gates'] = ';'.join(gate_text)
    stats['amplitude_failure_counts'] = ';'.join(
        f'{name}={count}' for name, count in sorted(failure_counts.items())
    )
    stats['amplitude_pass_count'] = len(valid)
    if not valid:
        stats['rejection_reason'] = 'no_amplitude_passed_hard_gates'
        _update_tracks(sequence, frame_id, assignments, boxes, raw_pose)
        stats['sequence_modified_frames'] = modified
        _FRAME_STATS[key] = stats
        return False

    best = min(valid, key=lambda x: x['cost'])
    alpha = best['alpha']
    final_pose = raw_pose.clone()
    final_pose[pid, pose_ids] += alpha*base_delta
    with torch.no_grad():
        final_points = _np(_project(final_pose, pred, frame_id, smpl_layer, smpl_model, K)[pid, joint_ids])

    final_error = _chain_error(final_points, item['target'], item['conf'], item['scale'])
    final_direction = _direction_error(final_points, item['target'])
    improvement = (item['raw_error']-final_error) / max(item['raw_error'], 1e-6)
    final_chain = _np(final_pose[pid, pose_ids])
    final_vel, final_acc, _ = _temporal(final_chain, history)
    if candidate_mode:
        required_improvement = max(
            chain['improve'] * 0.55,
            candidate_reproj_min[alpha],
        )
    else:
        required_improvement = max(
            chain['improve'],
            ALPHA_REPROJECTION_MIN[alpha],
        )
    final_applied_delta = alpha * base_max

    final_failures = []
    if improvement < required_improvement:
        final_failures.append('reprojection')
    if final_direction > item['raw_direction'] + direction_tol:
        final_failures.append('direction')
    final_acc_abs_drop = raw_acc - final_acc
    final_acc_rel_drop = final_acc_abs_drop / max(raw_acc, 1e-6)
    final_acceleration_pass = (
        final_acc < acceleration_limit
        and (
            final_acc_rel_drop >= accel_rel_req
            or final_acc_abs_drop >= accel_abs_req
        )
    )
    if not final_acceleration_pass:
        final_failures.append('acceleration')
    if final_vel > velocity_limit:
        final_failures.append('velocity')
    if final_applied_delta < min_applied_delta:
        final_failures.append('delta_small')

    if final_failures:
        stats['rejection_reason'] = (
            'exact_projection_gate_failed:' + ','.join(final_failures)
        )
        _update_tracks(sequence, frame_id, assignments, boxes, raw_pose)
        stats['sequence_modified_frames'] = modified
        _FRAME_STATS[key] = stats
        return False

    pred['rotvec'][frame_id] = final_pose.detach()
    if candidate_mode:
        _update_tracks(sequence, frame_id, assignments, boxes, raw_pose)
    else:
        modified += 1
        _MODIFIED[sequence] = modified
        _LAST_MODIFIED[sequence] = int(frame_id)
        _update_tracks(sequence, frame_id, assignments, boxes, final_pose)
    stats.update({
        'selected_alpha': alpha,
        'final_reprojection_error': final_error,
        'relative_reprojection_improvement': improvement,
        'required_reprojection_improvement': required_improvement,
        'final_direction_error': final_direction,
        'final_temporal_velocity': final_vel,
        'final_temporal_acceleration': final_acc,
        'temporal_acceleration_improvement': (
            raw_acc-final_acc
        ) / max(raw_acc, 1e-6),
        'candidate_cost': best['cost'],
        'rejection_reason': ('accepted_candidate' if candidate_mode else 'accepted'),
        'applied_delta_max': final_applied_delta,
        'sequence_modified_frames': modified,
        'pred_modified': True,
    })
    _FRAME_STATS[key] = stats
    return True


def _e4_record_online_audit(*, sequence, frame_id, pred_j2d, **unused):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = _FRAME_STATS.pop((sequence, int(frame_id)), _blank(len(_np(pred_j2d))))
    stats['sequence_modified_frames'] = _MODIFIED.get(sequence, stats['sequence_modified_frames'])
    row = {'sequence': sequence, 'frame_id': int(frame_id), **stats}
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    if frame_id == 0:
        print(
            f"[E4-T-V3] {sequence}: external={stats['n_external']} "
            f"matched={stats['n_matched']} modified={stats['pred_modified']}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# A2 / A2-lite / A2+A4 candidate generation, migrated from launch.py.
# ---------------------------------------------------------------------------

def _build_track_rows(pred: Dict[str, Any]):
    rot_raw = pred.get("rotvec", [])
    transl = pred.get("transl", [])
    n_frames = len(rot_raw)

    active = []
    rows = []
    next_tid = 0
    max_inactive = 1
    root_thr = float(os.environ.get("WAO_HMR_A2_ROOT_THR", "0.35"))
    pose_weight = float(os.environ.get("WAO_HMR_A2_POSE_WEIGHT", "0.02"))

    for frame_id in range(n_frames):
        n_people = int(rot_raw[frame_id].shape[0])
        current = [
            {
                "frame": frame_id,
                "pid": pid,
                "transl": transl[frame_id][pid],
                "rot": rot_raw[frame_id][pid],
            }
            for pid in range(n_people)
        ]

        assigned_tracks = set()
        assigned_pids = set()
        candidates = []

        for active_idx, track in enumerate(active):
            if frame_id - track["last_frame"] > max_inactive + 1:
                continue
            for item in current:
                root_dist = _value_norm(item["transl"] - track["last_transl"])
                pose_dist = _value_norm(item["rot"] - track["last_rot"])
                cost = root_dist + pose_weight * pose_dist
                candidates.append(
                    (cost, root_dist, pose_dist, active_idx, item["pid"])
                )

        candidates.sort(key=lambda item: item[0])
        for _, root_dist, pose_dist, active_idx, pid in candidates:
            if active_idx in assigned_tracks or pid in assigned_pids:
                continue
            if root_dist > root_thr:
                continue

            track = active[active_idx]
            rows.append(
                {
                    "track_id": track["tid"],
                    "frame": frame_id,
                    "pid": pid,
                    "root_jump": root_dist,
                    "pose_jump": pose_dist,
                    "is_new": 0,
                }
            )
            track["last_frame"] = frame_id
            track["last_pid"] = pid
            track["last_transl"] = transl[frame_id][pid]
            track["last_rot"] = rot_raw[frame_id][pid]
            assigned_tracks.add(active_idx)
            assigned_pids.add(pid)

        for item in current:
            if item["pid"] in assigned_pids:
                continue
            tid = next_tid
            next_tid += 1
            active.append(
                {
                    "tid": tid,
                    "last_frame": frame_id,
                    "last_pid": item["pid"],
                    "last_transl": item["transl"],
                    "last_rot": item["rot"],
                }
            )
            rows.append(
                {
                    "track_id": tid,
                    "frame": frame_id,
                    "pid": item["pid"],
                    "root_jump": 0.0,
                    "pose_jump": 0.0,
                    "is_new": 1,
                }
            )

        active = [
            track
            for track in active
            if frame_id - track["last_frame"] <= max_inactive + 1
        ]

    return rows


def _a2_frame_sets(rows: List[Dict[str, Any]], n_frames: int):
    frame_stats = []
    for frame_id in range(n_frames):
        current = [row for row in rows if row["frame"] == frame_id]
        if not current:
            frame_stats.append(
                {
                    "root_jump": 0.0,
                    "pose_jump": 0.0,
                    "new_ratio": 0.0,
                }
            )
            continue
        frame_stats.append(
            {
                "root_jump": float(
                    np.mean([row["root_jump"] for row in current])
                ),
                "pose_jump": float(
                    np.mean([row["pose_jump"] for row in current])
                ),
                "new_ratio": float(
                    np.mean([row["is_new"] for row in current])
                ),
            }
        )

    root_values = np.asarray(
        [item["root_jump"] for item in frame_stats],
        dtype=np.float64,
    )
    pose_values = np.asarray(
        [item["pose_jump"] for item in frame_stats],
        dtype=np.float64,
    )
    new_values = np.asarray(
        [item["new_ratio"] for item in frame_stats],
        dtype=np.float64,
    )

    def quantile(values, q):
        return float(np.quantile(values, q)) if len(values) else 0.0

    root_p90 = max(quantile(root_values, 0.90), 1e-8)
    pose_p90 = max(quantile(pose_values, 0.90), 1e-8)
    new_p90 = max(quantile(new_values, 0.90), 1e-8)

    badness = np.asarray(
        [
            min(item["root_jump"] / root_p90, 3.0)
            + min(item["pose_jump"] / pose_p90, 3.0)
            + min(item["new_ratio"] / new_p90, 3.0)
            for item in frame_stats
        ],
        dtype=np.float64,
    )

    bad_threshold = quantile(badness, 0.80)
    high_threshold = quantile(badness, 0.90)
    bad_frames = {
        index for index, value in enumerate(badness)
        if value >= bad_threshold
    }
    high_frames = {
        index for index, value in enumerate(badness)
        if value >= high_threshold
    }
    neighbor_frames = set()
    for frame_id in high_frames:
        if frame_id - 1 >= 0:
            neighbor_frames.add(frame_id - 1)
        if frame_id + 1 < n_frames:
            neighbor_frames.add(frame_id + 1)

    return bad_frames, high_frames, neighbor_frames


def _rows_by_track(rows: List[Dict[str, Any]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["track_id"]].append(row)
    for track_rows in grouped.values():
        track_rows.sort(key=lambda item: item["frame"])
    return grouped


def _apply_a2_smooth(
    source: List[Any],
    output: List[Any],
    grouped: Dict[int, List[Dict[str, Any]]],
    bad_frames: set,
    high_frames: set,
    neighbor_frames: set,
    *,
    gate_high: float,
    gate_bad: float,
    gate_neighbor: float,
    max_gap: int,
):
    updated_instances = 0
    for track_rows in grouped.values():
        for index in range(1, len(track_rows) - 1):
            previous = track_rows[index - 1]
            current = track_rows[index]
            following = track_rows[index + 1]

            frame_id = current["frame"]
            frame_previous = previous["frame"]
            frame_following = following["frame"]
            pid = current["pid"]
            pid_previous = previous["pid"]
            pid_following = following["pid"]

            if frame_following - frame_previous > max_gap:
                continue

            if frame_id in high_frames:
                gate = gate_high
            elif frame_id in bad_frames:
                gate = gate_bad
            elif frame_id in neighbor_frames:
                gate = gate_neighbor
            else:
                continue

            if gate <= 0.0:
                continue

            alpha = float(frame_id - frame_previous) / float(
                max(1, frame_following - frame_previous)
            )
            target = (
                (1.0 - alpha) * source[frame_previous][pid_previous]
                + alpha * source[frame_following][pid_following]
            )
            updated = (1.0 - gate) * output[frame_id][pid] + gate * target

            # Body-only contract.
            output[frame_id][pid][1:22] = updated[1:22]
            updated_instances += 1

    return updated_instances


def _build_a2_candidate(pred: Dict[str, Any], variant: str = "a2"):
    if "rotvec" not in pred or "transl" not in pred:
        return None, {"reason": "missing_rotvec_or_transl"}

    rot_raw = pred["rotvec"]
    n_frames = len(rot_raw)
    if n_frames == 0:
        return _clone_rotvec_list(rot_raw), {"reason": "empty"}

    rows = _build_track_rows(pred)
    if not rows:
        return _clone_rotvec_list(rot_raw), {"reason": "no_tracks"}

    bad_frames, high_frames, neighbor_frames = _a2_frame_sets(
        rows, n_frames
    )
    grouped = _rows_by_track(rows)

    if variant in {"a2lite", "a2_lite", "a2-lite"}:
        output = _clone_rotvec_list(rot_raw)
        updates = _apply_a2_smooth(
            rot_raw,
            output,
            grouped,
            bad_frames,
            high_frames,
            neighbor_frames,
            gate_high=0.45,
            gate_bad=0.25,
            gate_neighbor=0.00,
            max_gap=3,
        )
        a4_updates = 0
    else:
        output = _clone_rotvec_list(rot_raw)
        updates = _apply_a2_smooth(
            rot_raw,
            output,
            grouped,
            bad_frames,
            high_frames,
            neighbor_frames,
            gate_high=0.80,
            gate_bad=0.55,
            gate_neighbor=0.25,
            max_gap=4,
        )
        a4_updates = 0

        if variant in {"a2a4", "a2+a4", "a2_a4"}:
            second = _clone_rotvec_list(output)
            a4_updates = _apply_a2_smooth(
                output,
                second,
                grouped,
                bad_frames,
                high_frames,
                neighbor_frames,
                gate_high=0.30,
                gate_bad=0.22,
                gate_neighbor=0.10,
                max_gap=4,
            )
            output = second

    max_root_delta = max(
        [_root_delta_max(cand, raw) for cand, raw in zip(output, rot_raw)]
        or [0.0]
    )
    return output, {
        "tracks": len(grouped),
        "bad_frames": len(bad_frames),
        "high_frames": len(high_frames),
        "neighbor_frames": len(neighbor_frames),
        "bad_frame_ids": sorted(bad_frames),
        "high_frame_ids": sorted(high_frames),
        "neighbor_frame_ids": sorted(neighbor_frames),
        "updates": updates,
        "a4_updates": a4_updates,
        "max_root_delta": max_root_delta,
    }


# ---------------------------------------------------------------------------
# Compatibility dispatcher for old diagnostics / negative experiments.
# ---------------------------------------------------------------------------

_LEGACY_DISPATCH = {
    "c1_online_dryrun": ("wao_c1_online_refit", "apply_online_refit"),
    "c1_online_score": ("wao_c1_online_refit", "apply_online_refit"),
    "c1_online_refit": ("wao_c1_online_refit", "apply_online_refit"),
    "c2_body_candidate_score": (
        "wao_c2_body_candidate", "c2_body_candidate_score"
    ),
    "c3_internal_candidate_score": (
        "wao_c3_internal_candidates", "c3_internal_candidate_score"
    ),
    "c4_internal_fusion": (
        "wao_c4_internal_fusion", "c4_internal_fusion"
    ),
    "e1_bad_joint_probe": ("wao_e1_bad_joint_probe", "e1_bad_joint_probe"),
    "e2_joint_limb_repair": (
        "wao_e2_joint_limb_repair", "e2_joint_limb_repair"
    ),
    "e2b_adaptive_joint_repair": (
        "wao_e2b_adaptive_joint_repair", "e2b_adaptive_joint_repair"
    ),
    "e3_dense_limb_posterior": (
        "wao_e3_dense_limb_posterior", "e3_dense_limb_posterior"
    ),
    "z0_eval_path_smash": (
        "wao_z0_eval_path_smash", "z0_eval_path_smash"
    ),
    "z1_body_pose_smash": (
        "wao_z1_body_pose_smash", "z1_body_pose_smash"
    ),
}


def _dispatch_legacy(
    stage: str,
    pred: Dict[str, Any],
    filelist=None,
    sampled_indices=None,
    args=None,
):
    target = _LEGACY_DISPATCH.get(stage)
    if target is None:
        return None
    module_name, function_name = target
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function(
        pred,
        filelist=filelist,
        sampled_indices=sampled_indices,
        args=args,
    )


# ---------------------------------------------------------------------------
# B-group input branches and compact candidate contract.
#
# Generation stages keep image coordinates unchanged. They only suppress
# irrelevant background using the existing RT-DETRv2 + ViTPose observations:
#   visible: pose-guided soft visible-body focus mask;
#   bbox: detector-box focus mask.
#   sam3_context: real SAM3 instance-mask union with a preserved context ring.
#
# One compact torch file per sequence:
#   <root>/<branch>/<sequence>.pt
# Payload:
#   {"rotvec": list, "j2d": list, "branch": str, "sequence": str}
# ---------------------------------------------------------------------------

BRANCH_GENERATION_STAGES = {
    "b1_visible_generate": "visible",
    "b2_bbox_generate": "bbox",
    "b1_sam3_context_generate": "sam3_context",
}

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def _branch_candidate_root() -> Path:
    value = os.environ.get(
        "WAO_HMR_B_EXTERNAL_ROOT",
        "/root/data1/experiments/Human3R/wao_hmr/candidates/b_multisource",
    )
    return Path(value)

def _observation_items_for_view(sequence: str, frame_id: int):
    return _load_obs().get(sequence, {}).get(int(frame_id), [])

def _draw_branch_mask(sequence: str, frame_id: int, height: int, width: int, branch: str):
    from PIL import Image, ImageDraw, ImageFilter

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    items = _observation_items_for_view(sequence, frame_id)
    kept = 0
    for item in items:
        score = float(item.get("score", 0.0))
        if score < DET_SCORE_THR:
            continue
        kp = np.asarray(item.get("kp2d"), dtype=np.float32).copy()
        res = np.asarray(item.get("res", [width, height]), dtype=np.float32).reshape(-1)
        if kp.ndim != 2 or kp.shape[0] < 17 or len(res) < 2:
            continue
        kp[:, 0] *= width / max(float(res[0]), 1.0)
        kp[:, 1] *= height / max(float(res[1]), 1.0)
        conf = kp[:, 2]
        box = _bbox(kp, conf)
        if box is None:
            continue
        kept += 1
        x1, y1, x2, y2 = [float(x) for x in box]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        margin = 0.10 if branch == "bbox" else 0.05
        rect = (
            max(0, int(x1 - margin * bw)),
            max(0, int(y1 - margin * bh)),
            min(width - 1, int(x2 + margin * bw)),
            min(height - 1, int(y2 + margin * bh)),
        )
        if branch == "bbox":
            draw.rounded_rectangle(rect, radius=max(2, int(0.04 * min(bw, bh))), fill=255)
            continue

        radius = max(3, int(0.035 * max(bw, bh)))
        line_width = max(4, int(0.045 * max(bw, bh)))
        for j in range(17):
            if conf[j] < VIS_THR:
                continue
            x, y = float(kp[j, 0]), float(kp[j, 1])
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
        for a, b in COCO_EDGES:
            if conf[a] < VIS_THR or conf[b] < VIS_THR:
                continue
            draw.line(
                (float(kp[a, 0]), float(kp[a, 1]), float(kp[b, 0]), float(kp[b, 1])),
                fill=255, width=line_width,
            )
        torso = [idx for idx in (5, 6, 12, 11) if conf[idx] >= VIS_THR]
        if len(torso) >= 3:
            draw.polygon([(float(kp[i, 0]), float(kp[i, 1])) for i in torso], fill=255)

    if kept == 0:
        return None, 0
    blur_radius = max(2.0, 0.018 * max(height, width))
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
    return mask.clamp(0.0, 1.0), kept


def _sam3_mask_root() -> Path:
    return Path(os.environ.get(
        "WAO_HMR_SAM3_MASK_ROOT",
        "/root/data1/sam_pose/experiments/wham_v1/"
        "v4_singletrack_sam3_instance_masks_all3dpw_clean",
    ))


def _numeric_frame_id(path: Path):
    # Ignore short sequence suffixes such as _00; use a 4-8 digit frame token.
    tokens = re.findall(r"(?<!\d)(\d{4,8})(?!\d)", path.stem)
    if not tokens:
        return None
    try:
        return int(tokens[-1])
    except Exception:
        return None


def _sam3_sequence_index(sequence: str):
    root = _sam3_mask_root()
    key = (str(root), sequence)
    if key in _SAM3_SEQUENCE_INDEX:
        return _SAM3_SEQUENCE_INDEX[key]

    index = defaultdict(list)
    if not root.exists():
        print("[WAO_SAM3] missing mask root:", root, flush=True)
        _SAM3_SEQUENCE_INDEX[key] = index
        return index

    # M0 established that sequence names are present in paths. Scan the mask
    # root once per process, then filter the cached paths by sequence.
    root_key = str(root)
    if root_key not in _SAM3_ROOT_PNG_CACHE:
        _SAM3_ROOT_PNG_CACHE[root_key] = list(root.rglob("*.png"))
        print(
            "[WAO_SAM3_ROOT_INDEX]",
            "pngs=", len(_SAM3_ROOT_PNG_CACHE[root_key]),
            "root=", root_key,
            flush=True,
        )
    for path in _SAM3_ROOT_PNG_CACHE[root_key]:
        if sequence not in path.as_posix():
            continue
        frame_id = _numeric_frame_id(path)
        if frame_id is not None:
            index[int(frame_id)].append(path)

    _SAM3_SEQUENCE_INDEX[key] = index
    print(
        "[WAO_SAM3_INDEX]",
        "sequence=", sequence,
        "frames=", len(index),
        "pngs=", sum(len(v) for v in index.values()),
        "root=", str(root),
        flush=True,
    )
    return index


def _mask_name_score(path: Path) -> float:
    name = path.name.lower()
    score = 0.0
    for token, value in [
        ("mask", 5.0), ("seg", 3.0), ("sam", 2.0),
        ("person", 1.5), ("instance", 1.5),
        ("overlay", -5.0), ("visual", -4.0), ("vis", -2.0),
        ("render", -3.0), ("rgb", -2.0),
    ]:
        if token in name:
            score += value
    return score


def _read_mask_like_png(path: Path):
    from PIL import Image

    try:
        image = Image.open(path)
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None, -1e9

    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    # Prefer a meaningful transparent alpha channel when present.
    alpha_fg = float((alpha > 0.5).mean())
    if 0.001 <= alpha_fg <= 0.95 and float(alpha.std()) > 0.05:
        return (alpha > 0.5).astype(np.float32), 10.0 + _mask_name_score(path)

    gray = rgb.mean(axis=-1)
    binary_fraction = float(((gray <= 0.04) | (gray >= 0.96)).mean())

    bright = gray >= 0.5
    dark = gray < 0.5
    bright_fraction = float(bright.mean())
    dark_fraction = float(dark.mean())

    # Masks may encode the foreground as white or black. Prefer the minority
    # region so a white-background black mask is also handled correctly.
    if 0.001 <= bright_fraction <= 0.60:
        foreground = bright
        fg_fraction = bright_fraction
    elif 0.001 <= dark_fraction <= 0.60:
        foreground = dark
        fg_fraction = dark_fraction
    else:
        return None, -1e9

    name_score = _mask_name_score(path)
    threshold = 0.72 if name_score >= 2.0 else 0.88
    if binary_fraction < threshold:
        return None, -1e9

    score = 4.0 * binary_fraction + name_score - abs(fg_fraction - 0.20)
    return foreground.astype(np.float32), score


def _sam3_union_mask(sequence: str, frame_id: int, height: int, width: int):
    from PIL import Image

    root = _sam3_mask_root()
    cache_key = (str(root), sequence, int(frame_id), int(height), int(width))
    if cache_key in _SAM3_MASK_CACHE:
        return _SAM3_MASK_CACHE[cache_key]

    index = _sam3_sequence_index(sequence)
    # Infer a global 0-based/1-based offset once from the available mask ids.
    # Do not borrow an adjacent frame when a mask is genuinely missing.
    frame_keys = sorted(index)
    one_based = bool(frame_keys and 0 not in index and frame_keys[0] == 1)
    used_frame_id = int(frame_id) + (1 if one_based else 0)
    paths = index.get(used_frame_id, [])

    accepted = []
    for path in sorted(paths, key=lambda p: _mask_name_score(p), reverse=True):
        mask, score = _read_mask_like_png(path)
        if mask is None:
            continue
        accepted.append((score, path, mask))

    if not accepted:
        result = (None, {"files": 0, "frame_id": used_frame_id, "fraction": 0.0})
        _SAM3_MASK_CACHE[cache_key] = result
        return result

    # Union all plausible person-instance masks. This keeps multi-person input
    # consistent and leaves person matching to the existing 2D branch matcher.
    accepted.sort(key=lambda item: item[0], reverse=True)
    union = None
    used = []
    for score, path, mask in accepted:
        if union is None:
            union = mask.astype(np.float32)
        else:
            if mask.shape != union.shape:
                mask = np.asarray(
                    Image.fromarray((mask * 255).astype(np.uint8)).resize(
                        (union.shape[1], union.shape[0]), resample=Image.NEAREST
                    ),
                    dtype=np.float32,
                ) / 255.0
            union = np.maximum(union, mask)
        used.append(path)

    union_image = Image.fromarray((union > 0.5).astype(np.uint8) * 255)
    if union_image.size != (width, height):
        union_image = union_image.resize((width, height), resample=Image.NEAREST)
    union = np.asarray(union_image, dtype=np.float32) / 255.0

    fraction = float((union > 0.5).mean())
    if not (0.001 <= fraction <= 0.95):
        result = (None, {"files": 0, "frame_id": used_frame_id, "fraction": fraction})
    else:
        result = (
            torch.from_numpy(union).clamp(0.0, 1.0),
            {"files": len(used), "frame_id": used_frame_id, "fraction": fraction},
        )
    _SAM3_MASK_CACHE[cache_key] = result
    return result



def _mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(np.asarray(mask) > 0.5)
    if len(xs) < 8:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _bbox_center_distance(a, b, width, height):
    ca = np.asarray([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float32)
    cb = np.asarray([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)
    return float(np.linalg.norm(ca - cb) / max(float(np.hypot(width, height)), 1.0))


def _sam3_instance_masks(sequence: str, frame_id: int, height: int, width: int):
    """Return deduplicated person-instance masks for one frame.

    This deliberately does not union masks. The earlier SAM3-context input
    branch used the frame-level union and was nearly neutral on 3D metrics.
    B5 instead matches each raw Human3R person to one SAM3 instance mask.
    """
    from PIL import Image

    root = _sam3_mask_root()
    cache_key = (str(root), sequence, int(frame_id), int(height), int(width))
    if cache_key in _SAM3_INSTANCE_CACHE:
        return _SAM3_INSTANCE_CACHE[cache_key]

    index = _sam3_sequence_index(sequence)
    frame_keys = sorted(index)
    one_based = bool(frame_keys and 0 not in index and frame_keys[0] == 1)
    used_frame_id = int(frame_id) + (1 if one_based else 0)
    paths = index.get(used_frame_id, [])

    items = []
    for path in sorted(paths, key=lambda x: _mask_name_score(x), reverse=True):
        mask, score = _read_mask_like_png(path)
        if mask is None:
            continue
        image = Image.fromarray((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.NEAREST)
        arr = (np.asarray(image, dtype=np.uint8) > 127).astype(np.uint8)
        fraction = float(arr.mean())
        if not (0.001 <= fraction <= 0.80):
            continue
        bbox = _mask_bbox(arr)
        if bbox is None:
            continue

        duplicate = False
        for old in items:
            iou = _iou(bbox, old["bbox"])
            area_ratio = min(fraction, old["fraction"]) / max(fraction, old["fraction"], 1e-8)
            if iou >= 0.96 and area_ratio >= 0.94:
                overlap = float(np.logical_and(arr > 0, old["mask"] > 0).sum())
                union = float(np.logical_or(arr > 0, old["mask"] > 0).sum())
                if union > 0 and overlap / union >= 0.94:
                    duplicate = True
                    break
        if duplicate:
            continue

        items.append({
            "mask": arr,
            "bbox": bbox,
            "fraction": fraction,
            "score": float(score),
            "path": str(path),
        })

    _SAM3_INSTANCE_CACHE[cache_key] = items
    return items


def _match_raw_people_to_masks(raw_j2d, masks, height: int, width: int):
    raw = _safe_numpy(raw_j2d)
    if raw.ndim != 3 or not masks:
        return {}

    pboxes = [_bbox(points) for points in raw]
    pids = [i for i, box in enumerate(pboxes) if box is not None]
    mids = list(range(len(masks)))
    if not pids or not mids:
        return {}

    cost = np.full((len(pids), len(mids)), 100.0, dtype=np.float32)
    meta = {}
    for r, pid in enumerate(pids):
        for c, mid in enumerate(mids):
            iou = _iou(pboxes[pid], masks[mid]["bbox"])
            center = _bbox_center_distance(
                pboxes[pid], masks[mid]["bbox"], width, height
            )
            area_p = max(float((pboxes[pid][2]-pboxes[pid][0]) * (pboxes[pid][3]-pboxes[pid][1])), 1.0)
            area_m = max(float((masks[mid]["bbox"][2]-masks[mid]["bbox"][0]) * (masks[mid]["bbox"][3]-masks[mid]["bbox"][1])), 1.0)
            area_penalty = abs(math.log(area_p / area_m))
            cost[r, c] = (1.0 - iou) + 2.2 * center + 0.12 * area_penalty
            meta[(r, c)] = (iou, center)

    rows, cols = linear_sum_assignment(cost)
    output = {}
    for r, c in zip(rows, cols):
        iou, center = meta[(r, c)]
        if iou >= 0.15 and center <= 0.10:
            output[pids[r]] = masks[mids[c]]
    return output


def _project_vertices_2d(vertices, K):
    if vertices is None:
        return None
    if torch.is_tensor(vertices):
        n = int(vertices.shape[0])
        Kb = K.unsqueeze(0).expand(n, -1, -1)
        return torch.einsum(
            "bij,bkj->bki", Kb,
            vertices / vertices[..., 2:3].clamp_min(1e-4),
        )[..., :2]
    arr = np.asarray(vertices)
    k = _safe_numpy(K)
    norm = arr / np.maximum(arr[..., 2:3], 1e-4)
    return np.einsum("ij,nkj->nki", k, norm)[..., :2]


def _mask_candidate_features(vertex_2d, joint_2d, mask_item, height: int, width: int):
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask_item["mask"], dtype=np.uint8) > 0
    if mask.shape != (height, width):
        return None

    points = np.asarray(vertex_2d, dtype=np.float32)
    stride = max(1, int(os.environ.get("WAO_HMR_B5_VERTEX_STRIDE", "8")))
    points = points[::stride]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 32:
        return None

    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    in_frame = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    outside = np.ones(len(points), dtype=bool)
    outside[in_frame] = ~mask[y[in_frame], x[in_frame]]
    outside_ratio = float(outside.mean())

    distance = distance_transform_edt(~mask)
    diag = max(float(np.hypot(
        mask_item["bbox"][2] - mask_item["bbox"][0],
        mask_item["bbox"][3] - mask_item["bbox"][1],
    )), 1.0)
    distances = np.full(len(points), diag, dtype=np.float32)
    valid = in_frame
    distances[valid] = distance[y[valid], x[valid]]
    normalized = distances / diag

    pbox = _bbox(points)
    bbox_iou = _iou(pbox, mask_item["bbox"]) if pbox is not None else 0.0

    joints = np.asarray(joint_2d, dtype=np.float32)
    jfinite = np.isfinite(joints).all(axis=1)
    jx = np.rint(joints[:, 0]).astype(np.int64)
    jy = np.rint(joints[:, 1]).astype(np.int64)
    jin = jfinite & (jx >= 0) & (jx < width) & (jy >= 0) & (jy < height)
    joint_inside = np.zeros(len(joints), dtype=bool)
    joint_inside[jin] = mask[jy[jin], jx[jin]]
    joint_inside_ratio = float(joint_inside[jfinite].mean()) if jfinite.any() else 0.0

    outside_mean = float(normalized.mean())
    outside_p90 = float(np.quantile(normalized, 0.90))
    loss = (
        1.00 * outside_ratio
        + 1.60 * outside_mean
        + 0.25 * outside_p90
        + 0.35 * (1.0 - bbox_iou)
        + 0.25 * (1.0 - joint_inside_ratio)
    )

    return {
        "mask_bbox_iou": bbox_iou,
        "mask_joint_inside_ratio": joint_inside_ratio,
        "mask_vertex_outside_ratio": outside_ratio,
        "mask_vertex_outside_mean": outside_mean,
        "mask_vertex_outside_p90": outside_p90,
        "mask_loss": float(loss),
    }


def _apply_sam3_context(image: torch.Tensor, mask: torch.Tensor):
    height, width = int(image.shape[-2]), int(image.shape[-1])
    image4 = image.unsqueeze(0) if image.ndim == 3 else image
    mask4 = mask.to(device=image.device, dtype=image.dtype).view(1, 1, height, width)

    # Small context ring around the visible person mask. No crop or resize is
    # introduced, so K and image coordinates remain unchanged.
    dilate_kernel = max(3, int(round(0.045 * max(height, width))) | 1)
    ring = torch.nn.functional.max_pool2d(
        mask4, kernel_size=dilate_kernel, stride=1, padding=dilate_kernel // 2
    ).clamp(0.0, 1.0)

    soft_kernel = max(3, int(round(0.012 * max(height, width))) | 1)
    core = torch.nn.functional.avg_pool2d(
        mask4, kernel_size=soft_kernel, stride=1, padding=soft_kernel // 2
    ).clamp(0.0, 1.0)
    ring_soft = torch.maximum(
        core,
        torch.nn.functional.avg_pool2d(
            ring, kernel_size=soft_kernel, stride=1, padding=soft_kernel // 2
        ).clamp(0.0, 1.0),
    )

    blur_kernel = max(3, int(round(0.035 * max(height, width))) | 1)
    blurred = torch.nn.functional.avg_pool2d(
        image4, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2
    )

    outside_keep = float(os.environ.get("WAO_HMR_SAM3_OUTSIDE_KEEP", "0.55"))
    ring_keep = float(os.environ.get("WAO_HMR_SAM3_RING_KEEP", "0.82"))
    outside = outside_keep * image4 + (1.0 - outside_keep) * blurred
    ring_image = ring_keep * image4 + (1.0 - ring_keep) * blurred

    focused = (
        core * image4
        + (ring_soft - core).clamp_min(0.0) * ring_image
        + (1.0 - ring_soft).clamp_min(0.0) * outside
    )
    return focused if image.ndim == 4 else focused.squeeze(0)


def prepare_input_branch(views, filelist=None, sampled_indices=None, args=None):
    if os.environ.get("WAO_HMR_SAFE", "0") != "1":
        return views
    stage = _stage()
    branch = BRANCH_GENERATION_STAGES.get(stage)
    if branch is None or not filelist:
        return views

    sequence = _parse_seq_name(filelist[0])
    changed = 0
    no_observation = 0
    for view in views:
        image = view.get("img")
        if image is None or not torch.is_tensor(image):
            continue
        frame_id = int(view.get("idx", 0))
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if branch == "sam3_context":
            mask, mask_meta = _sam3_union_mask(
                sequence, frame_id, height, width
            )
            count = int(mask_meta.get("files", 0))
            if mask is None:
                no_observation += 1
                continue
            view["img"] = _apply_sam3_context(image, mask)
        else:
            mask, count = _draw_branch_mask(
                sequence, frame_id, height, width, branch
            )
            if mask is None:
                no_observation += 1
                continue
            mask = mask.to(device=image.device, dtype=image.dtype).view(
                1, 1, height, width
            )
            image4 = image.unsqueeze(0) if image.ndim == 3 else image
            blur_kernel = max(
                3, int(round(0.035 * max(height, width))) | 1
            )
            blurred = torch.nn.functional.avg_pool2d(
                image4,
                kernel_size=blur_kernel,
                stride=1,
                padding=blur_kernel // 2,
            )
            outside_mix = 0.30 if branch == "visible" else 0.45
            background = outside_mix * image4 + (1.0 - outside_mix) * blurred
            focused = mask * image4 + (1.0 - mask) * background
            view["img"] = focused if image.ndim == 4 else focused.squeeze(0)
        changed += 1

    print(
        "[WAO_B_INPUT_BRANCH]",
        "sequence=", sequence,
        "stage=", stage,
        "branch=", branch,
        "changed_views=", changed,
        "no_observation_views=", no_observation,
        "coordinates_unchanged=1",
        flush=True,
    )
    return views

def _external_branch_root() -> Optional[Path]:
    value = os.environ.get("WAO_HMR_B_EXTERNAL_ROOT", "").strip()
    return Path(value) if value else None

def _load_external_branch(branch: str, sequence: str):
    root = _external_branch_root()
    if root is None:
        return None
    key = (str(root), branch, sequence)
    if key in _EXTERNAL_BRANCH_CACHE:
        return _EXTERNAL_BRANCH_CACHE[key]
    path = root / branch / f"{sequence}.pt"
    if not path.exists():
        _EXTERNAL_BRANCH_CACHE[key] = None
        return None
    try:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            payload = {"rotvec": payload}
        rotvec = payload.get("rotvec")
        j2d = payload.get("j2d")
        if not isinstance(rotvec, (list, tuple)):
            raise TypeError("rotvec payload must be a list")
        payload = dict(payload)
        payload["rotvec"] = list(rotvec)
        payload["j2d"] = list(j2d) if isinstance(j2d, (list, tuple)) else None
    except Exception as exc:
        print("[WAO_UNIFIED] external branch load failed:", branch, sequence, repr(exc), flush=True)
        payload = None
    _EXTERNAL_BRANCH_CACHE[key] = payload
    return payload

def _match_branch_people(raw_j2d, branch_j2d):
    raw = _safe_numpy(raw_j2d)
    branch = _safe_numpy(branch_j2d)
    if raw.ndim != 3 or branch.ndim != 3 or len(raw) == 0 or len(branch) == 0:
        return []
    cost = np.full((len(raw), len(branch)), 100.0, dtype=np.float32)
    for i in range(len(raw)):
        rb = _bbox(raw[i])
        for j in range(len(branch)):
            bb = _bbox(branch[j])
            if rb is None or bb is None:
                continue
            scale = max(float(np.hypot(rb[2]-rb[0], rb[3]-rb[1])), 1.0)
            n = min(raw.shape[1], branch.shape[1])
            joint = float(np.linalg.norm(raw[i, :n] - branch[j, :n], axis=-1).mean() / scale)
            iou = _iou(rb, bb)
            cost[i, j] = joint + 0.5 * (1.0 - iou)
    rows, cols = linear_sum_assignment(cost)
    out = []
    for i, j in zip(rows, cols):
        if float(cost[i, j]) <= 1.20:
            out.append((int(i), int(j), float(cost[i, j])))
    return out

def _external_frame_pose(branch: str, sequence: str, frame_id: int, raw_pose: Any, raw_j2d: Any):
    payload = _load_external_branch(branch, sequence)
    if payload is None:
        return None, {"matched": 0, "branch_people": 0}
    values = payload.get("rotvec")
    j2d_values = payload.get("j2d")
    if frame_id < 0 or frame_id >= len(values):
        return None, {"matched": 0, "branch_people": 0}
    branch_pose = values[frame_id]
    branch_j2d = None if j2d_values is None or frame_id >= len(j2d_values) else j2d_values[frame_id]
    bp = _safe_numpy(branch_pose)
    rp = _safe_numpy(raw_pose)
    branch_people = int(bp.shape[0]) if bp.ndim == 3 else 0
    if bp.ndim != 3 or rp.ndim != 3 or bp.shape[1] < 22 or branch_j2d is None:
        return None, {"matched": 0, "branch_people": branch_people}
    matches = _match_branch_people(raw_j2d, branch_j2d)
    if not matches:
        return None, {"matched": 0, "branch_people": branch_people}
    candidate = _clone_value(raw_pose)
    for raw_pid, branch_pid, _ in matches:
        if raw_pid >= rp.shape[0] or branch_pid >= bp.shape[0]:
            continue
        if torch.is_tensor(candidate):
            candidate[raw_pid, 1:22] = torch.as_tensor(
                bp[branch_pid, 1:22], dtype=candidate.dtype, device=candidate.device
            )
        else:
            candidate[raw_pid, 1:22] = bp[branch_pid, 1:22].astype(candidate.dtype, copy=False)
    return candidate, {"matched": len(matches), "branch_people": branch_people}

# ---------------------------------------------------------------------------
# Unified sequence / frame / audit hooks used by launch.py.
# ---------------------------------------------------------------------------

B_STAGES = {
    "b0_candidate_audit",
    "b_multisource_audit",
    "b3_multisource_precision_audit",
    "b3_sam3_precision_audit",
    "b5_person_mask_constraint_audit",
    "b6_consensus_fusion",
    "b6_expanded_consensus",
    "b6_mask_aware_consensus",
    "g0_world_physics_audit",
    "h0_strong_local_oracle_audit",
    "h0v2_high_recall_oracle_audit",
    "h0v3_segment_oracle_audit",
    "h0v3_fast_h1b_oracle_audit",
}


def apply_sequence_stage(
    pred: Dict[str, Any],
    filelist=None,
    sampled_indices=None,
    args=None,
):
    if os.environ.get("WAO_HMR_SAFE", "0") != "1":
        return pred

    stage = _stage()
    if not stage:
        return pred

    # ===== GT_RESIDUAL_V2_SEQUENCE_DISPATCH =====
    if stage == "gt_residual_v2_test":
        from wao_gt_residual_v2_online import apply_sequence
        return apply_sequence(
            pred, filelist=filelist, sampled_indices=sampled_indices, args=args
        )
    # ===== GT_RESIDUAL_V2_SEQUENCE_DISPATCH_END =====

    legacy_result = _dispatch_legacy(
        stage,
        pred,
        filelist=filelist,
        sampled_indices=sampled_indices,
        args=args,
    )
    if legacy_result is not None:
        return legacy_result

    sequence = (
        _parse_seq_name(filelist[0])
        if filelist
        else "unknown"
    )

    if stage in {"a2", "a2_only", "a2-only"}:
        candidate, stats = _build_a2_candidate(pred, "a2")
        if candidate is not None:
            pred["rotvec"] = candidate
        print(
            "[WAO_UNIFIED_A2]",
            "sequence=", sequence,
            "variant=a2",
            "tracks=", stats.get("tracks", 0),
            "bad=", stats.get("bad_frames", 0),
            "high=", stats.get("high_frames", 0),
            "neighbor=", stats.get("neighbor_frames", 0),
            "updates=", stats.get("updates", 0),
            "max_root_delta=", stats.get("max_root_delta", 0.0),
            flush=True,
        )
        return pred

    if stage in {"a2lite", "a2_lite", "a2-lite"}:
        candidate, stats = _build_a2_candidate(pred, "a2lite")
        if candidate is not None:
            pred["rotvec"] = candidate
        print(
            "[WAO_UNIFIED_A2]",
            "sequence=", sequence,
            "variant=a2lite",
            "tracks=", stats.get("tracks", 0),
            "bad=", stats.get("bad_frames", 0),
            "high=", stats.get("high_frames", 0),
            "neighbor=", stats.get("neighbor_frames", 0),
            "updates=", stats.get("updates", 0),
            "max_root_delta=", stats.get("max_root_delta", 0.0),
            flush=True,
        )
        return pred

    if stage in {"a2a4", "a2+a4", "a2_a4"}:
        candidate, stats = _build_a2_candidate(pred, "a2a4")
        if candidate is not None:
            pred["rotvec"] = candidate
        print(
            "[WAO_UNIFIED_A2]",
            "sequence=", sequence,
            "variant=a2a4",
            "tracks=", stats.get("tracks", 0),
            "bad=", stats.get("bad_frames", 0),
            "high=", stats.get("high_frames", 0),
            "neighbor=", stats.get("neighbor_frames", 0),
            "updates=", stats.get("updates", 0),
            "a4_updates=", stats.get("a4_updates", 0),
            "max_root_delta=", stats.get("max_root_delta", 0.0),
            flush=True,
        )
        return pred

    if stage in BRANCH_GENERATION_STAGES:
        branch = BRANCH_GENERATION_STAGES[stage]
        _SEQUENCE_STATES[sequence] = {
            "stage": stage,
            "branch": branch,
            "rotvec": [_clone_value(x).detach().cpu() if torch.is_tensor(x) else np.array(x, copy=True) for x in pred.get("rotvec", [])],
            "j2d": {},
            "n_frames": len(pred.get("rotvec", [])),
        }
        print(
            "[WAO_B_BRANCH_BEGIN]",
            "sequence=", sequence,
            "branch=", branch,
            "frames=", len(pred.get("rotvec", [])),
            "will_modify_pred=0",
            flush=True,
        )
        return pred

    if stage in B_STAGES:
        raw_rotvec = _clone_rotvec_list(pred.get("rotvec", []))
        a2_rotvec, a2_stats = _build_a2_candidate(pred, "a2")
        a2lite_rotvec, a2lite_stats = _build_a2_candidate(pred, "a2lite")
        _SEQUENCE_STATES[sequence] = {
            "raw_rotvec": raw_rotvec,
            "a2_rotvec": a2_rotvec,
            "a2lite_rotvec": a2lite_rotvec,
            "a2_stats": a2_stats,
            "a2lite_stats": a2lite_stats,
            "a2_bad_ids": set(a2_stats.get("bad_frame_ids", [])),
            "a2_high_ids": set(a2_stats.get("high_frame_ids", [])),
            "a2_neighbor_ids": set(a2_stats.get("neighbor_frame_ids", [])),
            "n_frames": len(raw_rotvec),
            "stage": stage,
            "h0_proposals": 0,
            "h0_candidates": 0,
        }
        _reset_e4_sequence(sequence)
        for _key in list(_G0_RAW_HISTORY):
            if _key[0] == sequence:
                _G0_RAW_HISTORY.pop(_key, None)
        for _key in list(_H0_GT_HISTORY):
            if _key[0] == sequence:
                _H0_GT_HISTORY.pop(_key, None)
        _H0_TRACKS.pop(sequence, None)
        _H0_NEXT_TRACK.pop(sequence, None)
        for _key in list(_H0V2_FEATURE_HISTORY):
            if _key[0] == sequence:
                _H0V2_FEATURE_HISTORY.pop(_key, None)
        for _key in list(_H0V2_LAST_CHAIN_PROPOSAL):
            if _key[0] == sequence:
                _H0V2_LAST_CHAIN_PROPOSAL.pop(_key, None)
        _H0V3_PRECOMPUTED.discard(sequence)
        for _key in list(_H0V3_FRAME_PROPOSALS):
            if _key[0] == sequence:
                _H0V3_FRAME_PROPOSALS.pop(_key, None)
        print(
            "[WAO_UNIFIED_B_BEGIN]",
            "sequence=", sequence,
            "stage=", stage,
            "frames=", len(raw_rotvec),
            "a2_updates=", a2_stats.get("updates", 0),
            "will_modify_pred=", int(stage in {"b6_consensus_fusion", "b6_expanded_consensus", "b6_mask_aware_consensus"}),
            "frozen=root/transl/loc/shape/expression/K/T_c2w/camera",
            flush=True,
        )
        return pred

    # E4 is intentionally frame-level.
    if stage in {"e4_window_legacy", "e4_online"}:
        _reset_e4_sequence(sequence)
        print(
            "[WAO_UNIFIED_E4_BEGIN]",
            "sequence=", sequence,
            "stage=", stage,
            "frozen=root/transl/loc/shape/expression/K/T_c2w/camera",
            flush=True,
        )
        return pred

    print(
        "[WAO_UNIFIED] unknown stage, returning raw:",
        stage,
        flush=True,
    )
    return pred


def _reset_e4_sequence(sequence: str):
    _MODIFIED.pop(sequence, None)
    _LAST_MODIFIED.pop(sequence, None)
    _TRACKS.pop(sequence, None)
    _NEXT_TRACK.pop(sequence, None)
    for key in list(_FRAME_STATS):
        if key[0] == sequence:
            _FRAME_STATS.pop(key, None)


def _pairwise_body_rms(a: Any, b: Any) -> float:
    if a is None or b is None:
        return float("nan")
    aa = _safe_numpy(a)
    bb = _safe_numpy(b)
    if aa.shape != bb.shape or aa.ndim != 3:
        return float("nan")
    return float(np.sqrt(np.mean((aa[:, 1:22] - bb[:, 1:22]) ** 2)))



# ---------------------------------------------------------------------------
# B6 conservative no-GT consensus fusion.
#
# B4 LOSO is used only as diagnosis; no GT-trained model is loaded here.
# The current visible proxy is intentionally excluded. A2 is the main repair
# source, while bbox and E4-window may confirm/attenuate it. Raw is fallback.
# ---------------------------------------------------------------------------

B6_JOINT_CAPS = np.asarray([
    0.060, 0.060, 0.040,  # hips, spine1
    0.090, 0.090, 0.040,  # knees, spine2
    0.080, 0.080, 0.040,  # ankles, spine3
    0.065, 0.065, 0.040,  # feet, neck
    0.055, 0.055, 0.045,  # collars, head
    0.070, 0.070,          # shoulders
    0.100, 0.100,          # elbows
    0.090, 0.090,          # wrists
], dtype=np.float32)


def _finite_float(value, default=float("nan")):
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _delta_for_person(candidate, raw, pid):
    if candidate is None or raw is None:
        return None
    c = _safe_numpy(candidate)
    r = _safe_numpy(raw)
    if c.ndim != 3 or r.ndim != 3 or pid >= len(c) or pid >= len(r):
        return None
    return np.asarray(c[pid, 1:22] - r[pid, 1:22], dtype=np.float32)


def _delta_rms_max(delta):
    if delta is None or delta.size == 0:
        return 0.0, 0.0
    return (
        float(np.sqrt(np.mean(np.square(delta)))),
        float(np.linalg.norm(delta, axis=-1).max()),
    )


def _delta_agreement(first, second, min_delta=0.003):
    if first is None or second is None:
        return float("nan")
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2:
        return float("nan")
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    valid = (na >= min_delta) & (nb >= min_delta)
    if not np.any(valid):
        return float("nan")
    cosine = np.sum(a[valid] * b[valid], axis=-1) / np.maximum(
        na[valid] * nb[valid], 1e-8
    )
    return float(np.mean(np.clip(cosine, -1.0, 1.0)))


def _relative_drop(raw_value, candidate_value):
    raw_value = _finite_float(raw_value)
    candidate_value = _finite_float(candidate_value)
    if not np.isfinite(raw_value) or not np.isfinite(candidate_value):
        return float("nan")
    return float((raw_value - candidate_value) / max(abs(raw_value), 1e-8))


def _project_candidate_j2d(candidate_pose, pred, frame_id, smpl_layer, smpl_model, K):
    if candidate_pose is None:
        return None
    _, _, j2d = _candidate_geometry(
        candidate_pose, pred, frame_id, smpl_layer, smpl_model, K
    )
    return j2d


def _apply_frame_pose(pred, frame_id, pose):
    cur = pred["rotvec"][frame_id]
    if torch.is_tensor(cur):
        pred["rotvec"][frame_id] = pose.to(device=cur.device, dtype=cur.dtype)
    else:
        pred["rotvec"][frame_id] = np.asarray(pose, dtype=cur.dtype)


def _b6_fuse_frame(
    *, sequence, frame_id, image, pred, raw_pred_j2d,
    raw_pose, a2_pose, bbox_pose, window_pose, a2_frame_class,
    e4_stats, smpl_layer, smpl_model, K,
):
    min_delta = float(os.environ.get("WAO_HMR_B6_MIN_DELTA", "0.004"))
    agree_thr = float(os.environ.get("WAO_HMR_B6_AGREE_COS", "0.15"))
    max_person_rms = float(os.environ.get("WAO_HMR_B6_MAX_PERSON_RMS", "0.055"))
    max_persons = int(os.environ.get("WAO_HMR_B6_MAX_PERSONS_PER_FRAME", "2"))
    bbox_only_drop = float(os.environ.get("WAO_HMR_B6_BBOX_ONLY_REPROJ_DROP", "0.08"))
    window_min_rep = float(os.environ.get("WAO_HMR_B6_WINDOW_MIN_REPROJ", "0.10"))
    window_min_temp = float(os.environ.get("WAO_HMR_B6_WINDOW_MIN_TEMP", "0.05"))

    raw_j2d = raw_pred_j2d
    a2_j2d = _project_candidate_j2d(a2_pose, pred, frame_id, smpl_layer, smpl_model, K)
    bbox_j2d = _project_candidate_j2d(bbox_pose, pred, frame_id, smpl_layer, smpl_model, K)
    window_j2d = _project_candidate_j2d(window_pose, pred, frame_id, smpl_layer, smpl_model, K)

    output = _clone_value(raw_pose)
    n_people = int(raw_pose.shape[0])
    modified = []
    selection_rows = []

    window_pid = e4_stats.get("selected_pid", "")
    try:
        window_pid = int(window_pid)
    except Exception:
        window_pid = -1
    window_alpha = _finite_float(e4_stats.get("selected_alpha", float("nan")))
    window_rep_gain = _finite_float(
        e4_stats.get("relative_reprojection_improvement", float("nan"))
    )
    window_temp_gain = _finite_float(
        e4_stats.get("temporal_acceleration_improvement", float("nan"))
    )

    class_weight = {
        "high": float(os.environ.get("WAO_HMR_B6_A2_HIGH_WEIGHT", "0.72")),
        "bad": float(os.environ.get("WAO_HMR_B6_A2_BAD_WEIGHT", "0.58")),
        "neighbor": float(os.environ.get("WAO_HMR_B6_A2_NEIGHBOR_WEIGHT", "0.30")),
        "normal": 0.0,
    }

    proposals = []
    for pid in range(n_people):
        d_a2 = _delta_for_person(a2_pose, raw_pose, pid)
        d_bbox = _delta_for_person(bbox_pose, raw_pose, pid)
        d_window = _delta_for_person(window_pose, raw_pose, pid)
        a2_rms, a2_max = _delta_rms_max(d_a2)
        bbox_rms, bbox_max = _delta_rms_max(d_bbox)
        window_rms, window_max = _delta_rms_max(d_window)

        raw_rep = _external_reprojection_for_person(
            raw_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        a2_rep = _external_reprojection_for_person(
            a2_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        bbox_rep = _external_reprojection_for_person(
            bbox_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        window_rep = _external_reprojection_for_person(
            window_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        a2_rep_drop = _relative_drop(raw_rep, a2_rep)
        bbox_rep_drop = _relative_drop(raw_rep, bbox_rep)

        ab = _delta_agreement(d_a2, d_bbox, min_delta)
        aw = _delta_agreement(d_a2, d_window, min_delta)
        bw = _delta_agreement(d_bbox, d_window, min_delta)

        a2_changed = a2_max >= min_delta
        bbox_changed = bbox_max >= min_delta
        window_changed = pid == window_pid and window_max >= min_delta

        a2_weight = class_weight.get(a2_frame_class, 0.0) if a2_changed else 0.0
        a2_supported = (
            (np.isfinite(ab) and ab >= agree_thr)
            or (window_changed and np.isfinite(aw) and aw >= agree_thr)
        )
        if a2_weight > 0.0:
            if a2_supported:
                a2_weight = min(0.88, a2_weight + 0.14)
            if np.isfinite(a2_rep_drop):
                if a2_rep_drop >= 0.03:
                    a2_weight = min(0.90, a2_weight + 0.08)
                elif a2_rep_drop <= -0.08:
                    a2_weight *= 0.55
            if not a2_supported and a2_frame_class == "neighbor":
                a2_weight = 0.0

        bbox_weight = 0.0
        bbox_supported = (
            a2_changed and np.isfinite(ab) and ab >= agree_thr
        ) or (
            window_changed and np.isfinite(bw) and bw >= agree_thr
        )
        bbox_safe_size = bbox_rms <= 0.030 and bbox_max <= 0.120
        if bbox_changed and bbox_safe_size:
            if bbox_supported:
                bbox_weight = 0.28
            elif np.isfinite(bbox_rep_drop) and bbox_rep_drop >= bbox_only_drop:
                bbox_weight = 0.22

        window_weight = 0.0
        window_supported = (
            a2_changed and np.isfinite(aw) and aw >= agree_thr
        ) or (
            bbox_changed and np.isfinite(bw) and bw >= agree_thr
        )
        window_gate = (
            window_changed
            and np.isfinite(window_rep_gain) and window_rep_gain >= window_min_rep
            and np.isfinite(window_temp_gain) and window_temp_gain >= window_min_temp
        )
        if window_gate:
            if window_alpha <= 0.50:
                window_weight = 0.45
            elif window_supported:
                window_weight = 0.35
            if window_supported and window_weight > 0.0:
                window_weight = min(0.62, window_weight + 0.15)

        source_items = []
        if a2_weight > 0.0 and d_a2 is not None:
            source_items.append(("a2", a2_weight, d_a2))
        if bbox_weight > 0.0 and d_bbox is not None:
            source_items.append(("bbox", bbox_weight, d_bbox))
        if window_weight > 0.0 and d_window is not None:
            source_items.append(("window", window_weight, d_window))

        fused_delta = np.zeros((21, 3), dtype=np.float32)
        used_sources = set()
        for joint in range(21):
            local = []
            for name, weight, delta in source_items:
                if float(np.linalg.norm(delta[joint])) >= min_delta:
                    local.append((name, weight, delta[joint]))
            if not local:
                continue
            if len(local) == 1:
                name, weight, delta = local[0]
                allow_single = (
                    (name == "a2" and a2_frame_class in {"high", "bad"})
                    or (name == "bbox" and np.isfinite(bbox_rep_drop) and bbox_rep_drop >= bbox_only_drop)
                    or (name == "window" and window_gate and window_alpha <= 0.50)
                )
                if not allow_single:
                    continue
                joint_delta = weight * delta
                used_sources.add(name)
            else:
                # Reject strongly conflicting local deltas. A2 has priority,
                # then window, then bbox.
                priority = {"a2": 3, "window": 2, "bbox": 1}
                local = sorted(local, key=lambda x: priority[x[0]], reverse=True)
                reference = local[0][2]
                filtered = [local[0]]
                nr = float(np.linalg.norm(reference))
                for item in local[1:]:
                    nv = float(np.linalg.norm(item[2]))
                    cosine = float(np.dot(reference, item[2]) / max(nr * nv, 1e-8))
                    if cosine >= 0.0:
                        filtered.append(item)
                denom = sum(item[1] for item in filtered)
                joint_delta = sum(item[1] * item[2] for item in filtered) / max(denom, 1e-8)
                for item in filtered:
                    used_sources.add(item[0])
            cap = float(B6_JOINT_CAPS[joint])
            norm = float(np.linalg.norm(joint_delta))
            if norm > cap:
                joint_delta = joint_delta * (cap / max(norm, 1e-8))
            fused_delta[joint] = joint_delta

        body_rms, body_max = _delta_rms_max(fused_delta)
        if body_rms > max_person_rms:
            fused_delta *= max_person_rms / max(body_rms, 1e-8)
            body_rms, body_max = _delta_rms_max(fused_delta)
        is_modified = int(body_max >= min_delta and bool(used_sources))
        proposals.append((body_rms, pid, fused_delta, sorted(used_sources), is_modified))

        selection_rows.append({
            "sequence": sequence, "frame_id": int(frame_id), "pid": pid,
            "a2_frame_class": a2_frame_class,
            "selected_sources": ",".join(sorted(used_sources)),
            "modified": is_modified, "body_rms": body_rms, "body_max": body_max,
            "raw_reprojection": raw_rep, "a2_reprojection": a2_rep,
            "bbox_reprojection": bbox_rep, "window_reprojection": window_rep,
            "a2_bbox_agreement": ab, "a2_window_agreement": aw,
            "bbox_window_agreement": bw,
            "a2_weight": a2_weight, "bbox_weight": bbox_weight,
            "window_weight": window_weight, "window_alpha": window_alpha,
            "window_reprojection_improvement": window_rep_gain,
            "window_temporal_acceleration_improvement": window_temp_gain,
            "frozen_contract_ok": 1,
        })

    # Highest-confidence local corrections first; cap persons modified per frame.
    selected = sorted(
        [item for item in proposals if item[4]],
        key=lambda item: item[0], reverse=True,
    )[:max_persons]
    selected_pids = {item[1] for item in selected}
    for _, pid, delta, sources, _ in selected:
        raw_np = _safe_numpy(raw_pose)
        body = raw_np[pid, 1:22] + delta
        if torch.is_tensor(output):
            output[pid, 1:22] = torch.as_tensor(
                body, dtype=output.dtype, device=output.device
            )
        else:
            output[pid, 1:22] = body.astype(output.dtype, copy=False)
        modified.append((pid, sources, *_delta_rms_max(delta)))

    for row in selection_rows:
        if int(row["pid"]) not in selected_pids:
            row["modified"] = 0
            row["selected_sources"] = ""
            row["body_rms"] = 0.0
            row["body_max"] = 0.0

    max_root = _root_delta_max(output, raw_pose)
    for row in selection_rows:
        row["frozen_contract_ok"] = int(max_root <= 1e-7)
    return output, modified, selection_rows, max_root


def process_eval_frame(
    *,
    sequence,
    frame_id,
    image,
    pred,
    raw_pred_j2d,
    smpl_layer,
    smpl_model,
    K,
):
    if os.environ.get("WAO_HMR_SAFE", "0") != "1":
        return False

    stage = _stage()
    if stage in {"e4_window_legacy", "e4_online"}:
        return _e4_refine_online_frame(
            sequence=sequence,
            frame_id=frame_id,
            image=image,
            pred=pred,
            raw_pred_j2d=raw_pred_j2d,
            smpl_layer=smpl_layer,
            smpl_model=smpl_model,
            K=K,
        )

    if stage not in B_STAGES:
        return False

    state = _SEQUENCE_STATES.get(sequence)
    if state is None:
        raise RuntimeError(
            f"missing unified B sequence state for {sequence}"
        )

    raw_pose = state["raw_rotvec"][frame_id]
    a2_pose = state["a2_rotvec"][frame_id]
    a2lite_pose = state["a2lite_rotvec"][frame_id]

    # E4 runs on a private candidate prediction, never on official pred.
    window_pred = _copy_pred_with_frame_pose(pred, frame_id)
    window_modified = _e4_refine_online_frame(
        sequence=sequence,
        frame_id=frame_id,
        image=image,
        pred=window_pred,
        raw_pred_j2d=raw_pred_j2d,
        smpl_layer=smpl_layer,
        smpl_model=smpl_model,
        K=K,
        candidate_mode=(stage in {"b6_expanded_consensus", "b6_mask_aware_consensus", "b3_sam3_precision_audit", "b5_person_mask_constraint_audit", "g0_world_physics_audit", "h0_strong_local_oracle_audit", "h0v2_high_recall_oracle_audit", "h0v3_segment_oracle_audit", "h0v3_fast_h1b_oracle_audit"}),
    )
    window_pose = (
        window_pred["rotvec"][frame_id]
        if window_modified
        else None
    )

    visible_pose, visible_match = _external_frame_pose(
        "visible", sequence, frame_id, raw_pose, raw_pred_j2d
    )
    bbox_pose, bbox_match = _external_frame_pose(
        "bbox", sequence, frame_id, raw_pose, raw_pred_j2d
    )
    sam3_pose, sam3_match = _external_frame_pose(
        "sam3_context", sequence, frame_id, raw_pose, raw_pred_j2d
    )

    e4_stats = dict(_FRAME_STATS.get(
        (sequence, int(frame_id)),
        {},
    ))
    candidate_names = ["raw", "a2"]
    if stage in {"b6_expanded_consensus", "b6_mask_aware_consensus", "b3_sam3_precision_audit", "b5_person_mask_constraint_audit", "g0_world_physics_audit", "h0_strong_local_oracle_audit", "h0v2_high_recall_oracle_audit", "h0v3_segment_oracle_audit", "h0v3_fast_h1b_oracle_audit"}:
        candidate_names.append("a2lite")
    if window_pose is not None:
        candidate_names.append("window")
    if visible_pose is not None:
        candidate_names.append("visible")
    if bbox_pose is not None:
        candidate_names.append("bbox")
    if sam3_pose is not None:
        candidate_names.append("sam3_context")

    a2_rms, a2_max = _body_delta_stats(a2_pose, raw_pose)
    a2lite_rms, a2lite_max = _body_delta_stats(a2lite_pose, raw_pose)
    window_rms, window_max = _body_delta_stats(window_pose, raw_pose)
    visible_rms, _ = _body_delta_stats(visible_pose, raw_pose)
    bbox_rms, _ = _body_delta_stats(bbox_pose, raw_pose)
    sam3_rms, _ = _body_delta_stats(sam3_pose, raw_pose)

    root_deltas = [
        _root_delta_max(candidate, raw_pose)
        for candidate in [a2_pose, a2lite_pose, window_pose, visible_pose, bbox_pose, sam3_pose]
        if candidate is not None
    ]
    max_root_delta = max(root_deltas or [0.0])
    if frame_id in state.get("a2_high_ids", set()):
        a2_frame_class = "high"
    elif frame_id in state.get("a2_bad_ids", set()):
        a2_frame_class = "bad"
    elif frame_id in state.get("a2_neighbor_ids", set()):
        a2_frame_class = "neighbor"
    else:
        a2_frame_class = "normal"

    h0_proposals = []
    if stage == "h0_strong_local_oracle_audit":
        h0_proposals = _h0_generate_frame_candidates(
            sequence=sequence, frame_id=frame_id, image=image, pred=pred,
            raw_pose=raw_pose, raw_pred_j2d=raw_pred_j2d,
            smpl_layer=smpl_layer, smpl_model=smpl_model, K=K,
            a2_frame_class=a2_frame_class,
        )
    elif stage == "h0v2_high_recall_oracle_audit":
        h0_proposals = _h0v2_generate_frame_candidates(
            sequence=sequence, frame_id=frame_id, image=image, pred=pred,
            raw_pose=raw_pose, raw_pred_j2d=raw_pred_j2d,
            smpl_layer=smpl_layer, smpl_model=smpl_model, K=K,
            a2_frame_class=a2_frame_class,
            candidate_poses={
                "a2": a2_pose, "a2lite": a2lite_pose,
                "bbox": bbox_pose, "window": window_pose,
            },
        )
    elif stage in {"h0v3_segment_oracle_audit", "h0v3_fast_h1b_oracle_audit"}:
        if sequence not in _H0V3_PRECOMPUTED:
            (_h0v3_fast_precompute_sequence if stage == "h0v3_fast_h1b_oracle_audit" else _h0v3_precompute_sequence)(
                sequence=sequence, image=image, pred=pred,
                smpl_layer=smpl_layer, smpl_model=smpl_model, K_fallback=K,
            )
        h0_proposals = list(_H0V3_FRAME_PROPOSALS.get((sequence, int(frame_id)), []))
    if stage in {"h0_strong_local_oracle_audit", "h0v2_high_recall_oracle_audit", "h0v3_segment_oracle_audit", "h0v3_fast_h1b_oracle_audit"}:
        if h0_proposals:
            candidate_names.extend([
                f"h0:{item['chain']}:{item['variant']}"
                for item in h0_proposals
            ])
        _H0_FRAME_PROPOSALS[(sequence, int(frame_id))] = h0_proposals

    _B_FRAME_CANDIDATES[(sequence, int(frame_id))] = {
        "raw": raw_pose,
        "a2": a2_pose,
        "a2lite": a2lite_pose,
        "window": window_pose,
        "visible": visible_pose,
        "bbox": bbox_pose,
        "sam3_context": sam3_pose,
        "a2_frame_class": a2_frame_class,
        "e4_stats": e4_stats,
    }

    b6_modified = []
    b6_rows = []
    b6_max_root = max_root_delta
    if stage == "b6_consensus_fusion":
        fused_pose, b6_modified, b6_rows, b6_max_root = _b6_fuse_frame(
            sequence=sequence, frame_id=frame_id, image=image, pred=pred,
            raw_pred_j2d=raw_pred_j2d, raw_pose=raw_pose,
            a2_pose=a2_pose, bbox_pose=bbox_pose, window_pose=window_pose,
            a2_frame_class=a2_frame_class, e4_stats=e4_stats,
            smpl_layer=smpl_layer, smpl_model=smpl_model, K=K,
        )
        _apply_frame_pose(pred, frame_id, fused_pose)
        _B6_FRAME_SELECTIONS[(sequence, int(frame_id))] = b6_rows
        max_root_delta = max(max_root_delta, b6_max_root)
    elif stage == "b6_expanded_consensus":
        fused_pose, b6_modified, b6_rows, b6_max_root = _b6_v1_fuse_frame(
            sequence=sequence, frame_id=frame_id, image=image, pred=pred,
            raw_pred_j2d=raw_pred_j2d, raw_pose=raw_pose,
            a2_pose=a2_pose, a2lite_pose=a2lite_pose,
            bbox_pose=bbox_pose, window_pose=window_pose,
            a2_frame_class=a2_frame_class, e4_stats=e4_stats,
            smpl_layer=smpl_layer, smpl_model=smpl_model, K=K,
        )
        _apply_frame_pose(pred, frame_id, fused_pose)
        _B6_FRAME_SELECTIONS[(sequence, int(frame_id))] = b6_rows
        max_root_delta = max(max_root_delta, b6_max_root)
    elif stage == "b6_mask_aware_consensus":
        fused_pose, b6_modified, b6_rows, b6_max_root = _b6_mask_aware_fuse_frame(
            sequence=sequence, frame_id=frame_id, image=image, pred=pred,
            raw_pred_j2d=raw_pred_j2d, raw_pose=raw_pose,
            a2_pose=a2_pose, a2lite_pose=a2lite_pose,
            bbox_pose=bbox_pose, window_pose=window_pose,
            a2_frame_class=a2_frame_class, e4_stats=e4_stats,
            smpl_layer=smpl_layer, smpl_model=smpl_model, K=K,
        )
        _apply_frame_pose(pred, frame_id, fused_pose)
        _B6_FRAME_SELECTIONS[(sequence, int(frame_id))] = b6_rows
        max_root_delta = max(max_root_delta, b6_max_root)

    _B_FRAME_STATS[(sequence, int(frame_id))] = {
        "sequence": sequence,
        "frame_id": int(frame_id),
        "stage": stage,
        "n_persons": int(raw_pose.shape[0]),
        "candidate_names": ",".join(candidate_names),
        "candidate_count": len(candidate_names),
        "a2_body_rms": a2_rms,
        "a2_body_max": a2_max,
        "a2lite_body_rms": a2lite_rms,
        "a2lite_body_max": a2lite_max,
        "a2_a2lite_rms": _pairwise_body_rms(a2_pose, a2lite_pose),
        "window_available": int(window_pose is not None),
        "window_selected_pid": e4_stats.get("selected_pid", ""),
        "window_selected_chain": e4_stats.get("selected_chain", ""),
        "window_selected_alpha": e4_stats.get("selected_alpha", ""),
        "window_body_rms": window_rms,
        "window_body_max": window_max,
        "a2_window_rms": _pairwise_body_rms(a2_pose, window_pose),
        "visible_available": int(visible_pose is not None),
        "bbox_available": int(bbox_pose is not None),
        "sam3_available": int(sam3_pose is not None),
        "visible_body_rms": visible_rms,
        "bbox_body_rms": bbox_rms,
        "sam3_body_rms": sam3_rms,
        "visible_matched": visible_match.get("matched", 0),
        "visible_branch_people": visible_match.get("branch_people", 0),
        "bbox_matched": bbox_match.get("matched", 0),
        "bbox_branch_people": bbox_match.get("branch_people", 0),
        "sam3_matched": sam3_match.get("matched", 0),
        "sam3_branch_people": sam3_match.get("branch_people", 0),
        "max_root_delta": max_root_delta,
        "a2_frame_class": a2_frame_class,
        "frozen_contract_ok": int(max_root_delta <= 1e-7),
        "window_rejection_reason": e4_stats.get(
            "rejection_reason", ""
        ),
        "window_reprojection_improvement": e4_stats.get(
            "relative_reprojection_improvement", ""
        ),
        "window_temporal_acceleration_improvement": e4_stats.get(
            "temporal_acceleration_improvement", ""
        ),
        "b6_modified_persons": len(b6_modified),
        "b6_selected_sources": ";".join(
            f"{pid}:{'+'.join(sources)}" for pid, sources, _, _ in b6_modified
        ),
        "b6_mean_body_rms": (
            float(np.mean([item[2] for item in b6_modified]))
            if b6_modified else 0.0
        ),
        "b6_max_body_delta": max([item[3] for item in b6_modified] or [0.0]),
        "h0_proposals": len({(item.get("proposal_rank"), item.get("pid"), item.get("chain")) for item in h0_proposals}),
        "h0_candidates": len(h0_proposals),
        "h0_variant_names": ";".join(
            f"{item.get('chain','')}:{item.get('variant','')}"
            for item in h0_proposals
        ),
    }

    return bool(stage in {"b6_consensus_fusion", "b6_expanded_consensus", "b6_mask_aware_consensus"} and b6_modified)



def _b6_v1_fuse_frame(
    *, sequence, frame_id, image, pred, raw_pred_j2d,
    raw_pose, a2_pose, a2lite_pose, bbox_pose, window_pose,
    a2_frame_class, e4_stats, smpl_layer, smpl_model, K,
):
    """Expanded but still raw-anchored no-GT consensus.

    Differences from B6-v0:
      - A2-lite is available as the safer primary variant;
      - the window candidate comes from high-recall alpha 0.25/0.50 mode;
      - proposal ranking uses evidence strength, never edit magnitude;
      - visible proxy remains excluded.
    """
    min_delta = float(os.environ.get("WAO_HMR_B6V1_MIN_DELTA", "0.004"))
    agree_thr = float(os.environ.get("WAO_HMR_B6V1_AGREE_COS", "0.10"))
    max_person_rms = float(os.environ.get("WAO_HMR_B6V1_MAX_PERSON_RMS", "0.060"))
    max_persons = int(os.environ.get("WAO_HMR_B6V1_MAX_PERSONS_PER_FRAME", "2"))
    bbox_only_drop = float(os.environ.get("WAO_HMR_B6V1_BBOX_ONLY_REPROJ_DROP", "0.10"))
    window_min_rep = float(os.environ.get("WAO_HMR_B6V1_WINDOW_MIN_REPROJ", "0.06"))
    window_min_temp = float(os.environ.get("WAO_HMR_B6V1_WINDOW_MIN_TEMP", "0.01"))

    raw_j2d = raw_pred_j2d
    a2_j2d = _project_candidate_j2d(a2_pose, pred, frame_id, smpl_layer, smpl_model, K)
    lite_j2d = _project_candidate_j2d(a2lite_pose, pred, frame_id, smpl_layer, smpl_model, K)
    bbox_j2d = _project_candidate_j2d(bbox_pose, pred, frame_id, smpl_layer, smpl_model, K)
    window_j2d = _project_candidate_j2d(window_pose, pred, frame_id, smpl_layer, smpl_model, K)

    output = _clone_value(raw_pose)
    n_people = int(raw_pose.shape[0])
    modified = []
    selection_rows = []

    window_pid = e4_stats.get("selected_pid", "")
    try:
        window_pid = int(window_pid)
    except Exception:
        window_pid = -1
    window_alpha = _finite_float(e4_stats.get("selected_alpha", float("nan")))
    window_rep_gain = _finite_float(
        e4_stats.get("relative_reprojection_improvement", float("nan"))
    )
    window_temp_gain = _finite_float(
        e4_stats.get("temporal_acceleration_improvement", float("nan"))
    )

    proposals = []
    for pid in range(n_people):
        d_a2 = _delta_for_person(a2_pose, raw_pose, pid)
        d_lite = _delta_for_person(a2lite_pose, raw_pose, pid)
        d_bbox = _delta_for_person(bbox_pose, raw_pose, pid)
        d_window = _delta_for_person(window_pose, raw_pose, pid)

        a2_rms, a2_max = _delta_rms_max(d_a2)
        lite_rms, lite_max = _delta_rms_max(d_lite)
        bbox_rms, bbox_max = _delta_rms_max(d_bbox)
        window_rms, window_max = _delta_rms_max(d_window)

        raw_rep = _external_reprojection_for_person(
            raw_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        a2_rep = _external_reprojection_for_person(
            a2_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        lite_rep = _external_reprojection_for_person(
            lite_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        bbox_rep = _external_reprojection_for_person(
            bbox_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        window_rep = _external_reprojection_for_person(
            window_j2d, raw_j2d, sequence, frame_id, pid, image
        )
        a2_rep_drop = _relative_drop(raw_rep, a2_rep)
        lite_rep_drop = _relative_drop(raw_rep, lite_rep)
        bbox_rep_drop = _relative_drop(raw_rep, bbox_rep)

        al = _delta_agreement(d_a2, d_lite, min_delta)
        ab = _delta_agreement(d_a2, d_bbox, min_delta)
        lb = _delta_agreement(d_lite, d_bbox, min_delta)
        aw = _delta_agreement(d_a2, d_window, min_delta)
        lw = _delta_agreement(d_lite, d_window, min_delta)
        bw = _delta_agreement(d_bbox, d_window, min_delta)

        a2_changed = a2_max >= min_delta
        lite_changed = lite_max >= min_delta
        bbox_changed = bbox_max >= min_delta
        window_changed = pid == window_pid and window_max >= min_delta

        def supported(agreements):
            return sum(int(np.isfinite(x) and x >= agree_thr) for x in agreements)

        a2_support = supported([ab, aw])
        lite_support = supported([lb, lw])

        primary_name = ""
        primary_delta = None
        primary_rep_drop = float("nan")
        primary_weight = 0.0

        # A2-lite is the default when independent support is weak.
        if a2_frame_class == "high":
            if (
                a2_changed
                and a2_support >= max(1, lite_support)
                and (not np.isfinite(a2_rep_drop) or a2_rep_drop > -0.06)
            ):
                primary_name, primary_delta = "a2", d_a2
                primary_rep_drop, primary_weight = a2_rep_drop, 0.70
            elif lite_changed:
                primary_name, primary_delta = "a2lite", d_lite
                primary_rep_drop, primary_weight = lite_rep_drop, 0.56
        elif a2_frame_class == "bad":
            if (
                a2_changed and a2_support >= 1
                and (not np.isfinite(a2_rep_drop) or a2_rep_drop > -0.04)
            ):
                primary_name, primary_delta = "a2", d_a2
                primary_rep_drop, primary_weight = a2_rep_drop, 0.56
            elif lite_changed:
                primary_name, primary_delta = "a2lite", d_lite
                primary_rep_drop, primary_weight = lite_rep_drop, 0.46
        elif a2_frame_class == "neighbor" and lite_changed:
            if lite_support >= 1 or (np.isfinite(lite_rep_drop) and lite_rep_drop >= 0.03):
                primary_name, primary_delta = "a2lite", d_lite
                primary_rep_drop, primary_weight = lite_rep_drop, 0.30

        if primary_name == "a2":
            primary_support = a2_support
            pb = ab
            pw = aw
        elif primary_name == "a2lite":
            primary_support = lite_support
            pb = lb
            pw = lw
        else:
            primary_support = 0
            pb = float("nan")
            pw = float("nan")

        if primary_weight > 0.0:
            primary_weight = min(0.90, primary_weight + 0.10 * primary_support)
            if np.isfinite(primary_rep_drop):
                if primary_rep_drop >= 0.03:
                    primary_weight = min(0.92, primary_weight + 0.06)
                elif primary_rep_drop <= -0.08:
                    primary_weight *= 0.55

        bbox_weight = 0.0
        bbox_supported = (
            primary_delta is not None and np.isfinite(pb) and pb >= agree_thr
        ) or (
            window_changed and np.isfinite(bw) and bw >= agree_thr
        )
        bbox_safe_size = bbox_rms <= 0.030 and bbox_max <= 0.120
        if bbox_changed and bbox_safe_size:
            if bbox_supported:
                bbox_weight = 0.28
            elif np.isfinite(bbox_rep_drop) and bbox_rep_drop >= bbox_only_drop:
                bbox_weight = 0.20

        window_supported = (
            primary_delta is not None and np.isfinite(pw) and pw >= agree_thr
        ) or (
            bbox_changed and np.isfinite(bw) and bw >= agree_thr
        )
        window_gate = (
            window_changed
            and np.isfinite(window_alpha) and window_alpha <= 0.50
            and np.isfinite(window_rep_gain) and window_rep_gain >= window_min_rep
            and np.isfinite(window_temp_gain) and window_temp_gain >= window_min_temp
        )
        window_weight = 0.0
        if window_gate:
            window_weight = 0.40
            if window_supported:
                window_weight = 0.58

        source_items = []
        if primary_weight > 0.0 and primary_delta is not None:
            source_items.append((primary_name, primary_weight, primary_delta))
        if bbox_weight > 0.0 and d_bbox is not None:
            source_items.append(("bbox", bbox_weight, d_bbox))
        if window_weight > 0.0 and d_window is not None:
            source_items.append(("window", window_weight, d_window))

        fused_delta = np.zeros((21, 3), dtype=np.float32)
        used_sources = set()
        for joint in range(21):
            local = []
            for name, weight, delta in source_items:
                if float(np.linalg.norm(delta[joint])) >= min_delta:
                    local.append((name, weight, delta[joint]))
            if not local:
                continue
            if len(local) == 1:
                name, weight, delta = local[0]
                allow_single = (
                    (name in {"a2", "a2lite"} and a2_frame_class in {"high", "bad"})
                    or (name == "bbox" and np.isfinite(bbox_rep_drop) and bbox_rep_drop >= bbox_only_drop)
                    or (name == "window" and window_gate and window_rep_gain >= 0.08)
                )
                if not allow_single:
                    continue
                joint_delta = weight * delta
                used_sources.add(name)
            else:
                priority = {"a2": 4, "a2lite": 3, "window": 2, "bbox": 1}
                local = sorted(local, key=lambda x: priority[x[0]], reverse=True)
                reference = local[0][2]
                nr = float(np.linalg.norm(reference))
                filtered = [local[0]]
                for item in local[1:]:
                    nv = float(np.linalg.norm(item[2]))
                    cosine = float(np.dot(reference, item[2]) / max(nr * nv, 1e-8))
                    if cosine >= 0.0:
                        filtered.append(item)
                denom = sum(item[1] for item in filtered)
                joint_delta = sum(item[1] * item[2] for item in filtered) / max(denom, 1e-8)
                used_sources.update(item[0] for item in filtered)
            cap = float(B6_JOINT_CAPS[joint])
            norm = float(np.linalg.norm(joint_delta))
            if norm > cap:
                joint_delta = joint_delta * (cap / max(norm, 1e-8))
            fused_delta[joint] = joint_delta

        body_rms, body_max = _delta_rms_max(fused_delta)
        if body_rms > max_person_rms:
            fused_delta *= max_person_rms / max(body_rms, 1e-8)
            body_rms, body_max = _delta_rms_max(fused_delta)

        evidence = primary_weight + bbox_weight + window_weight
        evidence += 0.12 * primary_support
        for value in [primary_rep_drop, bbox_rep_drop, window_rep_gain, window_temp_gain]:
            if np.isfinite(value) and value > 0:
                evidence += min(float(value), 0.30) * 0.20

        is_modified = int(body_max >= min_delta and bool(used_sources) and evidence >= 0.42)
        proposals.append((evidence, body_rms, pid, fused_delta, sorted(used_sources), is_modified))

        selection_rows.append({
            "sequence": sequence, "frame_id": int(frame_id), "pid": pid,
            "a2_frame_class": a2_frame_class,
            "a2_variant": primary_name,
            "evidence_score": evidence,
            "selected_sources": ",".join(sorted(used_sources)),
            "modified": is_modified, "body_rms": body_rms, "body_max": body_max,
            "raw_reprojection": raw_rep, "a2_reprojection": a2_rep,
            "a2lite_reprojection": lite_rep, "bbox_reprojection": bbox_rep,
            "window_reprojection": window_rep,
            "a2_a2lite_agreement": al,
            "a2_bbox_agreement": ab, "a2lite_bbox_agreement": lb,
            "a2_window_agreement": aw, "a2lite_window_agreement": lw,
            "bbox_window_agreement": bw,
            "a2_weight": (primary_weight if primary_name == "a2" else 0.0),
            "bbox_weight": bbox_weight, "window_weight": window_weight,
            "window_alpha": window_alpha,
            "window_reprojection_improvement": window_rep_gain,
            "window_temporal_acceleration_improvement": window_temp_gain,
            "frozen_contract_ok": 1,
        })

    # Confidence/evidence first, not edit magnitude.
    selected = sorted(
        [item for item in proposals if item[5]],
        key=lambda item: (item[0], item[1]), reverse=True,
    )[:max_persons]
    selected_pids = {item[2] for item in selected}
    for evidence, _, pid, delta, sources, _ in selected:
        raw_np = _safe_numpy(raw_pose)
        body = raw_np[pid, 1:22] + delta
        if torch.is_tensor(output):
            output[pid, 1:22] = torch.as_tensor(body, dtype=output.dtype, device=output.device)
        else:
            output[pid, 1:22] = body.astype(output.dtype, copy=False)
        modified.append((pid, sources, *_delta_rms_max(delta)))

    for row in selection_rows:
        if int(row["pid"]) not in selected_pids:
            row["modified"] = 0
            row["selected_sources"] = ""
            row["body_rms"] = 0.0
            row["body_max"] = 0.0

    max_root = _root_delta_max(output, raw_pose)
    for row in selection_rows:
        row["frozen_contract_ok"] = int(max_root <= 1e-7)
    return output, modified, selection_rows, max_root


def _b6_mask_feature_table(
    *, sequence, frame_id, image, pred, K, smpl_layer, smpl_model,
    raw_pose, pose_map,
):
    """Return per-person, per-candidate mask losses without GT."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    masks = _sam3_instance_masks(sequence, frame_id, height, width)
    if not masks:
        return {}

    geometry = {}
    for name, pose in pose_map.items():
        if pose is None:
            continue
        try:
            geometry[name] = _candidate_geometry(
                pose, pred, frame_id, smpl_layer, smpl_model, K
            )
        except Exception:
            continue
    if "raw" not in geometry:
        return {}

    raw_vertices, raw_joints, raw_j2d, _ = geometry["raw"]
    matched = _match_raw_people_to_masks(raw_j2d, masks, height, width)
    if not matched:
        return {}

    projected_vertices = {
        name: _project_vertices_2d(values[0], K)
        for name, values in geometry.items()
    }
    table = {}
    for pid, mask_item in matched.items():
        per_name = {}
        for name, (vertices, joints, j2d, pose) in geometry.items():
            v2d = projected_vertices.get(name)
            if v2d is None or pid >= int(v2d.shape[0]) or pid >= int(j2d.shape[0]):
                continue
            features = _mask_candidate_features(
                _safe_numpy(v2d)[pid], _safe_numpy(j2d)[pid],
                mask_item, height, width,
            )
            if features is not None:
                per_name[name] = features
        if "raw" in per_name:
            table[int(pid)] = {
                "mask": mask_item,
                "features": per_name,
            }
    return table


def _scaled_capped_delta(delta, weight, max_person_rms):
    if delta is None:
        return None
    out = np.asarray(delta, dtype=np.float32).copy() * float(weight)
    for joint in range(min(21, out.shape[0])):
        cap = float(B6_JOINT_CAPS[joint])
        norm = float(np.linalg.norm(out[joint]))
        if norm > cap:
            out[joint] *= cap / max(norm, 1e-8)
    rms, _ = _delta_rms_max(out)
    if rms > max_person_rms:
        out *= max_person_rms / max(rms, 1e-8)
    return out


def _b6_mask_aware_fuse_frame(
    *, sequence, frame_id, image, pred, raw_pred_j2d,
    raw_pose, a2_pose, a2lite_pose, bbox_pose, window_pose,
    a2_frame_class, e4_stats, smpl_layer, smpl_model, K,
):
    """B6-v1 consensus with per-person SAM3 confirmation/rejection.

    The mask is never used as a 3D target. It can only:
      1) reject a weak consensus edit that worsens silhouette alignment;
      2) boost an already-supported edit that improves mask loss;
      3) fill an unused frame slot with one conservative candidate whose
         mask loss improves by a fixed margin.
    """
    mask_margin = float(os.environ.get("WAO_HMR_B6M_MASK_MARGIN", "0.005"))
    reject_margin = float(os.environ.get("WAO_HMR_B6M_MASK_REJECT", "0.0025"))
    max_persons = int(os.environ.get("WAO_HMR_B6M_MAX_PERSONS_PER_FRAME", "2"))
    max_person_rms = float(os.environ.get("WAO_HMR_B6M_MAX_PERSON_RMS", "0.055"))
    strong_override = float(os.environ.get("WAO_HMR_B6M_STRONG_EVIDENCE", "1.00"))
    min_delta = float(os.environ.get("WAO_HMR_B6V1_MIN_DELTA", "0.004"))

    base_pose, _, base_rows, _ = _b6_v1_fuse_frame(
        sequence=sequence, frame_id=frame_id, image=image, pred=pred,
        raw_pred_j2d=raw_pred_j2d, raw_pose=raw_pose,
        a2_pose=a2_pose, a2lite_pose=a2lite_pose,
        bbox_pose=bbox_pose, window_pose=window_pose,
        a2_frame_class=a2_frame_class, e4_stats=e4_stats,
        smpl_layer=smpl_layer, smpl_model=smpl_model, K=K,
    )

    pose_map = {
        "raw": raw_pose,
        "fused": base_pose,
        "a2": a2_pose,
        "a2lite": a2lite_pose,
        "bbox": bbox_pose,
        "window": window_pose,
    }
    mask_table = _b6_mask_feature_table(
        sequence=sequence, frame_id=frame_id, image=image, pred=pred, K=K,
        smpl_layer=smpl_layer, smpl_model=smpl_model,
        raw_pose=raw_pose, pose_map=pose_map,
    )

    row_by_pid = {int(row["pid"]): row for row in base_rows}
    proposals = []
    n_people = int(raw_pose.shape[0])

    window_pid = e4_stats.get("selected_pid", "")
    try:
        window_pid = int(window_pid)
    except Exception:
        window_pid = -1
    window_alpha = _finite_float(e4_stats.get("selected_alpha", float("nan")))
    window_rep_gain = _finite_float(
        e4_stats.get("relative_reprojection_improvement", float("nan"))
    )
    window_temp_gain = _finite_float(
        e4_stats.get("temporal_acceleration_improvement", float("nan"))
    )

    for pid in range(n_people):
        row = row_by_pid.get(pid)
        if row is None:
            continue
        item = mask_table.get(pid)
        features = item.get("features", {}) if item else {}
        raw_loss = _finite_float(features.get("raw", {}).get("mask_loss", float("nan")))
        fused_loss = _finite_float(features.get("fused", {}).get("mask_loss", float("nan")))
        fused_mask_delta = (
            fused_loss - raw_loss
            if np.isfinite(raw_loss) and np.isfinite(fused_loss)
            else float("nan")
        )

        row.update({
            "mask_matched": int(item is not None),
            "raw_mask_loss": raw_loss,
            "selected_mask_loss": fused_loss,
            "mask_loss_delta": fused_mask_delta,
            "mask_decision": "unmatched" if item is None else "neutral",
            "mask_candidate": "",
        })

        base_modified = int(row.get("modified", 0)) == 1
        if base_modified:
            delta = _delta_for_person(base_pose, raw_pose, pid)
            evidence = _finite_float(row.get("evidence_score", 0.0), 0.0)
            keep = True
            decision = "unmatched" if item is None else "neutral"
            if item is not None and np.isfinite(fused_mask_delta):
                if fused_mask_delta <= -mask_margin:
                    evidence += min(0.30, (-fused_mask_delta / max(mask_margin, 1e-8)) * 0.08)
                    decision = "confirm"
                elif fused_mask_delta > reject_margin and evidence < strong_override:
                    keep = False
                    decision = "reject"
            if keep and delta is not None:
                proposals.append({
                    "pid": pid, "evidence": evidence, "delta": delta,
                    "sources": [x for x in str(row.get("selected_sources", "")).split(",") if x],
                    "row": row, "mask_decision": decision,
                    "mask_candidate": "fused",
                    "mask_loss": fused_loss, "mask_delta": fused_mask_delta,
                })
            else:
                row["modified"] = 0
                row["selected_sources"] = ""
                row["body_rms"] = 0.0
                row["body_max"] = 0.0
                row["mask_decision"] = decision

        # Conservative mask-supported fallback only when B6-v1 did not edit.
        if base_modified or item is None or not np.isfinite(raw_loss):
            continue

        candidate_specs = []
        if a2_frame_class in {"high", "bad"}:
            candidate_specs.append(("a2", a2_pose, 0.56 if a2_frame_class == "high" else 0.46, 0.12))
            candidate_specs.append(("a2lite", a2lite_pose, 0.50 if a2_frame_class == "high" else 0.42, 0.10))
        elif a2_frame_class == "neighbor":
            candidate_specs.append(("a2lite", a2lite_pose, 0.28, 0.06))
        candidate_specs.append(("bbox", bbox_pose, 0.20, 0.04))
        if (
            pid == window_pid and np.isfinite(window_alpha) and window_alpha <= 0.50
            and np.isfinite(window_rep_gain) and window_rep_gain >= 0.06
            and np.isfinite(window_temp_gain) and window_temp_gain >= 0.01
        ):
            candidate_specs.append(("window", window_pose, 0.40, 0.16))

        best = None
        for name, pose, weight, prior in candidate_specs:
            cand_features = features.get(name)
            if pose is None or cand_features is None:
                continue
            cand_loss = _finite_float(cand_features.get("mask_loss", float("nan")))
            if not np.isfinite(cand_loss):
                continue
            mask_delta = cand_loss - raw_loss
            if mask_delta > -mask_margin:
                continue
            raw_delta = _delta_for_person(pose, raw_pose, pid)
            rms, maxv = _delta_rms_max(raw_delta)
            if not np.isfinite(maxv) or maxv < min_delta:
                continue
            if name == "bbox" and (rms > 0.030 or maxv > 0.120):
                continue
            scaled = _scaled_capped_delta(raw_delta, weight, max_person_rms)
            if scaled is None:
                continue
            srms, smax = _delta_rms_max(scaled)
            if smax < min_delta:
                continue
            evidence = 0.42 + prior + min(0.42, (-mask_delta / max(mask_margin, 1e-8)) * 0.08)
            proposal = {
                "pid": pid, "evidence": evidence, "delta": scaled,
                "sources": [name], "row": row,
                "mask_decision": "fallback", "mask_candidate": name,
                "mask_loss": cand_loss, "mask_delta": mask_delta,
            }
            if best is None or proposal["evidence"] > best["evidence"]:
                best = proposal
        if best is not None:
            proposals.append(best)

    # One proposal per person, then evidence-first frame budget.
    best_by_pid = {}
    for proposal in proposals:
        pid = int(proposal["pid"])
        if pid not in best_by_pid or proposal["evidence"] > best_by_pid[pid]["evidence"]:
            best_by_pid[pid] = proposal
    selected = sorted(
        best_by_pid.values(), key=lambda x: x["evidence"], reverse=True
    )[:max_persons]
    selected_pids = {int(x["pid"]) for x in selected}

    output = _clone_value(raw_pose)
    modified = []
    raw_np = _safe_numpy(raw_pose)
    for proposal in selected:
        pid = int(proposal["pid"])
        delta = np.asarray(proposal["delta"], dtype=np.float32)
        body = raw_np[pid, 1:22] + delta
        if torch.is_tensor(output):
            output[pid, 1:22] = torch.as_tensor(body, dtype=output.dtype, device=output.device)
        else:
            output[pid, 1:22] = body.astype(output.dtype, copy=False)
        rms, maxv = _delta_rms_max(delta)
        modified.append((pid, proposal["sources"], rms, maxv))
        row = proposal["row"]
        row["modified"] = 1
        row["selected_sources"] = ",".join(proposal["sources"])
        row["evidence_score"] = proposal["evidence"]
        row["body_rms"] = rms
        row["body_max"] = maxv
        row["selected_mask_loss"] = proposal["mask_loss"]
        row["mask_loss_delta"] = proposal["mask_delta"]
        row["mask_decision"] = proposal["mask_decision"]
        row["mask_candidate"] = proposal["mask_candidate"]

    for pid, row in row_by_pid.items():
        if pid not in selected_pids:
            row["modified"] = 0
            row["selected_sources"] = ""
            row["body_rms"] = 0.0
            row["body_max"] = 0.0
            if not row.get("mask_decision"):
                row["mask_decision"] = "not_selected"

    max_root = _root_delta_max(output, raw_pose)
    for row in base_rows:
        row["frozen_contract_ok"] = int(max_root <= 1e-7)
    return output, modified, base_rows, max_root

def _similarity_transform(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    sm = source.mean(axis=0, keepdims=True)
    tm = target.mean(axis=0, keepdims=True)
    sc = source - sm
    tc = target - tm
    u, sv, vh = np.linalg.svd(sc.T @ tc)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    scale = float(np.sum(sv)) / max(float(np.sum(sc ** 2)), 1e-12)
    translation = tm.reshape(3) - scale * (sm.reshape(3) @ rotation.T)
    return scale, rotation, translation

def _metric_bundle(pred_joints, pred_vertices, target_joints, target_vertices, pelvis_index):
    pj = np.asarray(pred_joints, dtype=np.float64)
    pv = np.asarray(pred_vertices, dtype=np.float64)
    tj = np.asarray(target_joints, dtype=np.float64)
    tv = np.asarray(target_vertices, dtype=np.float64)
    pr = pj[int(pelvis_index)]
    tr = tj[int(pelvis_index)]
    mpjpe = float(np.linalg.norm((pj-pr) - (tj-tr), axis=1).mean() * 1000.0)
    pve = float(np.linalg.norm((pv-pr) - (tv-tr), axis=1).mean() * 1000.0)
    metric_mpjpe = float(np.linalg.norm(pj-tj, axis=1).mean() * 1000.0)
    metric_pve = float(np.linalg.norm(pv-tv, axis=1).mean() * 1000.0)
    scale, rotation, translation = _similarity_transform(pj, tj)
    pj_pa = scale * (pj @ rotation.T) + translation
    pv_pa = scale * (pv @ rotation.T) + translation
    pa_mpjpe = float(np.linalg.norm(pj_pa-tj, axis=1).mean() * 1000.0)
    pa_pve = float(np.linalg.norm(pv_pa-tv, axis=1).mean() * 1000.0)
    return {
        "mpjpe": mpjpe, "pa_mpjpe": pa_mpjpe,
        "pve": pve, "pa_pve": pa_pve,
        "metric_mpjpe": metric_mpjpe, "metric_pve": metric_pve,
    }

def _candidate_geometry(candidate_pose, pred, frame_id, smpl_layer, smpl_model, K):
    n = int(candidate_pose.shape[0])
    if n == 0:
        return None, None, None
    Kb = K.unsqueeze(0).expand(n, -1, -1)
    with torch.no_grad():
        out = smpl_layer(
            candidate_pose, pred["shape"][frame_id], pred["transl"][frame_id],
            None, None, K=Kb, expression=pred["expression"][frame_id],
        )
        vx = out["smpl_v3d"]
        vertices = smpl_model.smplx2smpl.to(vx) @ vx
        joints = smpl_model.j_regressor.to(vertices) @ vertices
        j2d = torch.einsum(
            "bij,bkj->bki", Kb, joints / joints[..., 2:3].clamp_min(1e-4)
        )[..., :2]
    return vertices, joints, j2d

def _person_body_delta(candidate, raw, pid):
    c = _safe_numpy(candidate)
    r = _safe_numpy(raw)
    if c.ndim != 3 or r.ndim != 3 or pid >= len(c) or pid >= len(r):
        return float("nan"), float("nan")
    d = c[pid, 1:22] - r[pid, 1:22]
    return float(np.sqrt(np.mean(d ** 2))), float(np.linalg.norm(d, axis=-1).max())

def _consensus_rms(candidates, name, pid):
    base = candidates.get(name)
    if base is None:
        return float("nan")
    vals = []
    for other_name, other in candidates.items():
        if other_name == name or other is None:
            continue
        a = _safe_numpy(base)
        b = _safe_numpy(other)
        if a.ndim == 3 and b.ndim == 3 and pid < len(a) and pid < len(b):
            vals.append(float(np.sqrt(np.mean((a[pid,1:22]-b[pid,1:22])**2))))
    return float(np.mean(vals)) if vals else float("nan")

def _external_reprojection_for_person(candidate_j2d, raw_j2d, sequence, frame_id, pid, image):
    if candidate_j2d is None or raw_j2d is None or pid >= len(candidate_j2d):
        return float("nan")
    height, width = int(image.shape[-2]), int(image.shape[-1])
    detections = _detections(sequence, frame_id, height, width)
    matches = _match_external(_safe_numpy(raw_j2d), detections, height, width)
    match = next((m for m in matches if int(m["pid"]) == int(pid)), None)
    if match is None:
        return float("nan")
    det = detections[match["did"]]
    mapping = [16,14,12,11,13,15,10,8,6,5,7,9]
    jids = list(range(12))
    conf = det["kp"][mapping, 2]
    valid = conf >= VIS_THR
    if valid.sum() < 3:
        return float("nan")
    target = det["kp"][mapping, :2]
    points = _safe_numpy(candidate_j2d)[pid, jids]
    box = det["bbox"]
    scale = max(float(np.hypot(box[2]-box[0], box[3]-box[1])), 1.0)
    return _chain_error(points[valid], target[valid], conf[valid], scale)


# ---------------------------------------------------------------------------
# G0 camera/world 3D and causal physical-consistency audit.
# All temporal quantities are per-frame proxies because dataset FPS is not
# injected into the evaluator hook. No prediction is modified.
# ---------------------------------------------------------------------------

_G0_BONES = [
    (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8),
    (6,9), (7,10), (8,11), (9,12), (12,13), (12,14), (12,15),
    (13,16), (14,17), (16,18), (17,19), (18,20), (19,21),
    (20,22), (21,23),
]
_G0_ENDPOINTS = [7, 8, 10, 11, 20, 21, 22, 23]


def _g0_transform_points(transform, points):
    pts = np.asarray(points, dtype=np.float64)
    mat = np.asarray(transform, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[-1] != 3:
        return None
    if mat.shape == (4, 4):
        homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        return (homo @ mat.T)[:, :3]
    if mat.shape == (3, 4):
        homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        return homo @ mat.T
    if mat.shape == (3, 3):
        return pts @ mat.T
    return None


def _g0_rms_mm(delta):
    arr = np.asarray(delta, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=-1))) * 1000.0)


def _g0_mean_norm_mm(delta):
    arr = np.asarray(delta, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.linalg.norm(arr, axis=-1).mean() * 1000.0)


def _g0_max_norm_mm(delta):
    arr = np.asarray(delta, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.linalg.norm(arr, axis=-1).max() * 1000.0)


def _g0_valid_indices(count, indices):
    return [int(i) for i in indices if 0 <= int(i) < int(count)]


def _g0_bone_angles_deg(current, previous):
    cur = np.asarray(current, dtype=np.float64)
    prev = np.asarray(previous, dtype=np.float64)
    values = []
    flips = 0
    used = 0
    for a, b in _G0_BONES:
        if max(a, b) >= len(cur) or max(a, b) >= len(prev):
            continue
        vc = cur[b] - cur[a]
        vp = prev[b] - prev[a]
        nc = float(np.linalg.norm(vc))
        npv = float(np.linalg.norm(vp))
        if nc < 1e-8 or npv < 1e-8:
            continue
        cosine = float(np.clip(np.dot(vc, vp) / (nc * npv), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        values.append(angle)
        flips += int(angle > 120.0)
        used += 1
    return (
        float(np.mean(values)) if values else float("nan"),
        float(flips / used) if used else float("nan"),
    )


def _g0_world_metric_bundle(pred_joints_w, pred_vertices_w, gt_joints_w, gt_vertices_w, pelvis):
    if pred_joints_w is None or pred_vertices_w is None or gt_joints_w is None or gt_vertices_w is None:
        return {"world_mpjpe": float("nan"), "world_pve": float("nan"), "world_root_error": float("nan")}
    pj = np.asarray(pred_joints_w, dtype=np.float64)
    pv = np.asarray(pred_vertices_w, dtype=np.float64)
    tj = np.asarray(gt_joints_w, dtype=np.float64)
    tv = np.asarray(gt_vertices_w, dtype=np.float64)
    n_j = min(len(pj), len(tj))
    n_v = min(len(pv), len(tv))
    if n_j == 0 or n_v == 0:
        return {"world_mpjpe": float("nan"), "world_pve": float("nan"), "world_root_error": float("nan")}
    pi = min(int(pelvis), n_j - 1)
    return {
        "world_mpjpe": float(np.linalg.norm(pj[:n_j] - tj[:n_j], axis=-1).mean() * 1000.0),
        "world_pve": float(np.linalg.norm(pv[:n_v] - tv[:n_v], axis=-1).mean() * 1000.0),
        "world_root_error": float(np.linalg.norm(pj[pi] - tj[pi]) * 1000.0),
    }


def _g0_temporal_features(frame_id, candidate_world, candidate_cam, raw_world, raw_cam, history):
    out = {
        "temporal_history": 0,
        "inserted_world_velocity_mm": float("nan"),
        "raw_world_velocity_mm": float("nan"),
        "world_velocity_excess_mm": float("nan"),
        "inserted_world_acceleration_mm": float("nan"),
        "raw_world_acceleration_mm": float("nan"),
        "world_acceleration_excess_mm": float("nan"),
        "inserted_world_jerk_mm": float("nan"),
        "raw_world_jerk_mm": float("nan"),
        "world_jerk_excess_mm": float("nan"),
        "inserted_endpoint_velocity_mm": float("nan"),
        "raw_endpoint_velocity_mm": float("nan"),
        "endpoint_velocity_excess_mm": float("nan"),
        "inserted_endpoint_acceleration_mm": float("nan"),
        "raw_endpoint_acceleration_mm": float("nan"),
        "endpoint_acceleration_excess_mm": float("nan"),
        "depth_jump_excess_mm": float("nan"),
        "limb_angle_jump_deg": float("nan"),
        "raw_limb_angle_jump_deg": float("nan"),
        "limb_angle_jump_excess_deg": float("nan"),
        "limb_flip_ratio": float("nan"),
    }
    consecutive = []
    expected = int(frame_id) - 1
    for item in reversed(history):
        if int(item["frame_id"]) != expected:
            break
        consecutive.append(item)
        expected -= 1
        if len(consecutive) == 3:
            break
    out["temporal_history"] = len(consecutive)
    if not consecutive:
        return out
    prev = consecutive[0]
    n = min(len(candidate_world), len(raw_world), len(prev["world"]))
    if n == 0:
        return out
    cand = candidate_world[:n]
    raw = raw_world[:n]
    prev_w = prev["world"][:n]
    cand_v = cand - prev_w
    raw_v = raw - prev_w
    out["inserted_world_velocity_mm"] = _g0_mean_norm_mm(cand_v)
    out["raw_world_velocity_mm"] = _g0_mean_norm_mm(raw_v)
    out["world_velocity_excess_mm"] = out["inserted_world_velocity_mm"] - out["raw_world_velocity_mm"]
    endpoints = _g0_valid_indices(n, _G0_ENDPOINTS)
    if endpoints:
        out["inserted_endpoint_velocity_mm"] = _g0_mean_norm_mm(cand_v[endpoints])
        out["raw_endpoint_velocity_mm"] = _g0_mean_norm_mm(raw_v[endpoints])
        out["endpoint_velocity_excess_mm"] = out["inserted_endpoint_velocity_mm"] - out["raw_endpoint_velocity_mm"]
    nc = min(len(candidate_cam), len(raw_cam), len(prev["cam"]))
    if nc:
        cand_depth = np.abs(candidate_cam[:nc, 2] - prev["cam"][:nc, 2])
        raw_depth = np.abs(raw_cam[:nc, 2] - prev["cam"][:nc, 2])
        out["depth_jump_excess_mm"] = float((cand_depth.mean() - raw_depth.mean()) * 1000.0)
    cand_ang, flip = _g0_bone_angles_deg(cand, prev_w)
    raw_ang, _ = _g0_bone_angles_deg(raw, prev_w)
    out["limb_angle_jump_deg"] = cand_ang
    out["raw_limb_angle_jump_deg"] = raw_ang
    out["limb_angle_jump_excess_deg"] = cand_ang - raw_ang if np.isfinite(cand_ang) and np.isfinite(raw_ang) else float("nan")
    out["limb_flip_ratio"] = flip
    if len(consecutive) >= 2:
        prev2_w = consecutive[1]["world"][:n]
        cand_a = cand - 2.0 * prev_w + prev2_w
        raw_a = raw - 2.0 * prev_w + prev2_w
        out["inserted_world_acceleration_mm"] = _g0_mean_norm_mm(cand_a)
        out["raw_world_acceleration_mm"] = _g0_mean_norm_mm(raw_a)
        out["world_acceleration_excess_mm"] = out["inserted_world_acceleration_mm"] - out["raw_world_acceleration_mm"]
        if endpoints:
            out["inserted_endpoint_acceleration_mm"] = _g0_mean_norm_mm(cand_a[endpoints])
            out["raw_endpoint_acceleration_mm"] = _g0_mean_norm_mm(raw_a[endpoints])
            out["endpoint_acceleration_excess_mm"] = out["inserted_endpoint_acceleration_mm"] - out["raw_endpoint_acceleration_mm"]
    if len(consecutive) >= 3:
        prev2_w = consecutive[1]["world"][:n]
        prev3_w = consecutive[2]["world"][:n]
        cand_j = cand - 3.0 * prev_w + 3.0 * prev2_w - prev3_w
        raw_j = raw - 3.0 * prev_w + 3.0 * prev2_w - prev3_w
        out["inserted_world_jerk_mm"] = _g0_mean_norm_mm(cand_j)
        out["raw_world_jerk_mm"] = _g0_mean_norm_mm(raw_j)
        out["world_jerk_excess_mm"] = out["inserted_world_jerk_mm"] - out["raw_world_jerk_mm"]
    return out


def _g0_physics_penalty(features, joint_limit_excess_ratio):
    terms = [
        max(float(features.get("world_acceleration_excess_mm", 0.0) or 0.0), 0.0) / 20.0,
        max(float(features.get("world_jerk_excess_mm", 0.0) or 0.0), 0.0) / 30.0,
        max(float(features.get("endpoint_acceleration_excess_mm", 0.0) or 0.0), 0.0) / 30.0,
        max(float(features.get("limb_angle_jump_excess_deg", 0.0) or 0.0), 0.0) / 10.0,
        max(float(features.get("depth_jump_excess_mm", 0.0) or 0.0), 0.0) / 20.0,
        max(float(joint_limit_excess_ratio), 0.0) * 2.0,
    ]
    return float(sum(x for x in terms if np.isfinite(x)))


def _g0_write_rows(output, rows, pair_rows):
    if rows:
        path = output / "g0_world_physics.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=G0_WORLD_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    if pair_rows:
        path = output / "g0_pairwise_ambiguity.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=G0_PAIRWISE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(pair_rows)


def _g0_record_world_physics(
    *, sequence, frame_id, image, pred, pred_j2d, gt_j3d_c, gt_v3d_c,
    gt_j3d_w, gt_v3d_w, best_match, pelvis_idx, K, smpl_layer, smpl_model,
    candidates,
):
    if pred is None or best_match is None or len(best_match) == 0:
        return
    raw_pose = candidates.get("raw")
    if raw_pose is None:
        return
    names = ["raw", "a2", "a2lite", "bbox", "window"]
    geometry = {}
    for name in names:
        pose = candidates.get(name)
        if pose is None:
            continue
        values = _candidate_geometry(pose, pred, frame_id, smpl_layer, smpl_model, K)
        if values[0] is not None:
            geometry[name] = (*values, pose)
    if "raw" not in geometry:
        return
    transform = pred.get("T_c2w", [None])[frame_id]
    transform_np = _safe_numpy(transform)
    height, width = int(image.shape[-2]), int(image.shape[-1])
    diag = max(float(np.hypot(width, height)), 1.0)
    pelvis = int(_safe_numpy(pelvis_idx).reshape(-1)[0]) if pelvis_idx is not None else 0
    output = _run_root()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    pair_rows = []
    for pair in np.asarray(best_match, dtype=np.int64):
        pid, gid = int(pair[0]), int(pair[1])
        raw_vertices, raw_joints, raw_j2d, _ = geometry["raw"]
        if pid >= len(raw_joints) or gid >= len(gt_j3d_c):
            continue
        raw_j_cam = _safe_numpy(raw_joints)[pid]
        raw_v_cam = _safe_numpy(raw_vertices)[pid]
        raw_j_world = _g0_transform_points(transform_np, raw_j_cam)
        raw_v_world = _g0_transform_points(transform_np, raw_v_cam)
        if raw_j_world is None or raw_v_world is None:
            continue
        gt_j_world = None if gt_j3d_w is None or gid >= len(gt_j3d_w) else _safe_numpy(gt_j3d_w)[gid]
        gt_v_world = None if gt_v3d_w is None or gid >= len(gt_v3d_w) else _safe_numpy(gt_v3d_w)[gid]
        raw_metrics = _metric_bundle(
            raw_j_cam, raw_v_cam, _safe_numpy(gt_j3d_c)[gid], _safe_numpy(gt_v3d_c)[gid], pelvis,
        )
        raw_world_metrics = _g0_world_metric_bundle(raw_j_world, raw_v_world, gt_j_world, gt_v_world, pelvis)
        history_key = (sequence, gid)
        history = _G0_RAW_HISTORY.setdefault(history_key, [])
        per_candidate = {}
        for name, (vertices, joints, cand_j2d, pose) in geometry.items():
            if pid >= len(joints) or pid >= len(pose):
                continue
            cand_j_cam = _safe_numpy(joints)[pid]
            cand_v_cam = _safe_numpy(vertices)[pid]
            cand_j_world = _g0_transform_points(transform_np, cand_j_cam)
            cand_v_world = _g0_transform_points(transform_np, cand_v_cam)
            if cand_j_world is None or cand_v_world is None:
                continue
            metrics = _metric_bundle(
                cand_j_cam, cand_v_cam, _safe_numpy(gt_j3d_c)[gid], _safe_numpy(gt_v3d_c)[gid], pelvis,
            )
            world_metrics = _g0_world_metric_bundle(cand_j_world, cand_v_world, gt_j_world, gt_v_world, pelvis)
            body_rms, body_max = _person_body_delta(pose, raw_pose, pid)
            changed = int(name != "raw" and np.isfinite(body_max) and body_max > 1e-7)
            if name != "raw" and not changed:
                continue
            n = min(len(cand_j_cam), len(raw_j_cam))
            pi = min(pelvis, n - 1)
            cam_delta = cand_j_cam[:n] - raw_j_cam[:n]
            cand_rel = cand_j_cam[:n] - cand_j_cam[pi]
            raw_rel = raw_j_cam[:n] - raw_j_cam[pi]
            world_delta = cand_j_world[:n] - raw_j_world[:n]
            endpoints = _g0_valid_indices(n, _G0_ENDPOINTS)
            depth_delta = np.abs(cand_j_cam[:n, 2] - raw_j_cam[:n, 2])
            j2d_a = _safe_numpy(cand_j2d)[pid]
            j2d_r = _safe_numpy(raw_j2d)[pid]
            nj2 = min(len(j2d_a), len(j2d_r))
            pair2d_px = float(np.sqrt(np.mean(np.sum((j2d_a[:nj2] - j2d_r[:nj2]) ** 2, axis=-1)))) if nj2 else float("nan")
            temporal = _g0_temporal_features(
                frame_id, cand_j_world, cand_j_cam, raw_j_world, raw_j_cam, history,
            )
            rot = _safe_numpy(pose)[pid, 1:22]
            rot_norm = np.linalg.norm(rot, axis=-1)
            limit_ratio = float(np.mean(rot_norm > 2.6)) if len(rot_norm) else 0.0
            penalty = _g0_physics_penalty(temporal, limit_ratio)
            row = {
                "sequence": sequence, "frame_id": int(frame_id), "pred_id": pid, "gt_id": gid,
                "candidate": name, "candidate_changed": changed,
                "a2_frame_class": candidates.get("a2_frame_class", ""),
                "body_rms_from_raw": body_rms, "body_max_from_raw": body_max,
                "camera_joint_rms_from_raw_mm": _g0_rms_mm(cam_delta),
                "camera_joint_max_from_raw_mm": _g0_max_norm_mm(cam_delta),
                "pelvis_relative_joint_rms_from_raw_mm": _g0_rms_mm(cand_rel - raw_rel),
                "world_joint_rms_from_raw_mm": _g0_rms_mm(world_delta),
                "world_endpoint_rms_from_raw_mm": _g0_rms_mm(world_delta[endpoints]) if endpoints else float("nan"),
                "mean_depth_delta_from_raw_mm": float(depth_delta.mean() * 1000.0) if len(depth_delta) else float("nan"),
                "max_depth_delta_from_raw_mm": float(depth_delta.max() * 1000.0) if len(depth_delta) else float("nan"),
                "pairwise_2d_rms_from_raw_px": pair2d_px,
                "pairwise_2d_rms_from_raw_norm": pair2d_px / diag if np.isfinite(pair2d_px) else float("nan"),
                **temporal,
                "joint_rotvec_norm_mean": float(rot_norm.mean()) if len(rot_norm) else float("nan"),
                "joint_rotvec_norm_max": float(rot_norm.max()) if len(rot_norm) else float("nan"),
                "joint_limit_excess_ratio": limit_ratio,
                "physics_penalty_proxy": penalty,
                "external_reprojection_error": _external_reprojection_for_person(cand_j2d, pred_j2d, sequence, frame_id, pid, image),
                "consensus_rms": _consensus_rms(candidates, name, pid),
                "window_selected_chain": candidates.get("e4_stats", {}).get("selected_chain", ""),
                "window_selected_alpha": candidates.get("e4_stats", {}).get("selected_alpha", ""),
                "window_reprojection_improvement": candidates.get("e4_stats", {}).get("relative_reprojection_improvement", ""),
                "window_temporal_acceleration_improvement": candidates.get("e4_stats", {}).get("temporal_acceleration_improvement", ""),
                **metrics, **world_metrics,
                "delta_mpjpe_from_raw": 0.0 if name == "raw" else metrics["mpjpe"] - raw_metrics["mpjpe"],
                "delta_pve_from_raw": 0.0 if name == "raw" else metrics["pve"] - raw_metrics["pve"],
                "delta_pa_mpjpe_from_raw": 0.0 if name == "raw" else metrics["pa_mpjpe"] - raw_metrics["pa_mpjpe"],
                "delta_pa_pve_from_raw": 0.0 if name == "raw" else metrics["pa_pve"] - raw_metrics["pa_pve"],
                "delta_world_mpjpe_from_raw": 0.0 if name == "raw" else world_metrics["world_mpjpe"] - raw_world_metrics["world_mpjpe"],
                "delta_world_pve_from_raw": 0.0 if name == "raw" else world_metrics["world_pve"] - raw_world_metrics["world_pve"],
                "frozen_contract_ok": int(_root_delta_max(pose, raw_pose) <= 1e-7),
            }
            rows.append(row)
            per_candidate[name] = {
                "j_cam": cand_j_cam, "j_world": cand_j_world, "j2d": j2d_a,
                "delta_mpjpe": row["delta_mpjpe_from_raw"], "delta_pve": row["delta_pve_from_raw"],
                "frozen": row["frozen_contract_ok"],
            }
        for name_a, name_b in itertools.combinations(sorted(per_candidate), 2):
            a = per_candidate[name_a]
            b = per_candidate[name_b]
            nj = min(len(a["j_cam"]), len(b["j_cam"]))
            n2 = min(len(a["j2d"]), len(b["j2d"]))
            if nj == 0 or n2 == 0:
                continue
            pi = min(pelvis, nj - 1)
            a_rel = a["j_cam"][:nj] - a["j_cam"][pi]
            b_rel = b["j_cam"][:nj] - b["j_cam"][pi]
            pair2d = float(np.sqrt(np.mean(np.sum((a["j2d"][:n2] - b["j2d"][:n2]) ** 2, axis=-1))))
            norm2d = pair2d / diag
            world_delta = a["j_world"][:nj] - b["j_world"][:nj]
            endpoints = _g0_valid_indices(nj, _G0_ENDPOINTS)
            pair3d = _g0_rms_mm(a_rel - b_rel)
            pair_rows.append({
                "sequence": sequence, "frame_id": int(frame_id), "pred_id": pid, "gt_id": gid,
                "candidate_a": name_a, "candidate_b": name_b,
                "pair_2d_rms_px": pair2d, "pair_2d_rms_norm": norm2d,
                "pair_camera_pelvis_relative_rms_mm": pair3d,
                "pair_world_joint_rms_mm": _g0_rms_mm(world_delta),
                "pair_world_endpoint_rms_mm": _g0_rms_mm(world_delta[endpoints]) if endpoints else float("nan"),
                "near_2d_0025": int(norm2d <= 0.0025),
                "near_2d_005": int(norm2d <= 0.005),
                "near_2d_01": int(norm2d <= 0.01),
                "different_3d_20mm": int(pair3d >= 20.0),
                "different_3d_40mm": int(pair3d >= 40.0),
                "different_3d_60mm": int(pair3d >= 60.0),
                "candidate_a_delta_mpjpe": a["delta_mpjpe"],
                "candidate_b_delta_mpjpe": b["delta_mpjpe"],
                "candidate_a_delta_pve": a["delta_pve"],
                "candidate_b_delta_pve": b["delta_pve"],
                "frozen_contract_ok": int(a["frozen"] and b["frozen"]),
            })
        history.append({
            "frame_id": int(frame_id),
            "world": np.array(raw_j_world, copy=True),
            "cam": np.array(raw_j_cam, copy=True),
        })
        if len(history) > 3:
            del history[:-3]
    _g0_write_rows(output, rows, pair_rows)


# ---------------------------------------------------------------------------
# H0 strong local multi-hypothesis generation and GT-only oracle audit.
# ---------------------------------------------------------------------------

H0_ORACLE_FIELDS = [
    "sequence", "frame_id", "pred_id", "gt_id", "candidate",
    "candidate_family", "chain", "variant", "proposal_rank", "track_id",
    "history_depth", "risk_score", "segment_id", "segment_start", "segment_end",
    "segment_center", "segment_length", "segment_offset", "segment_peak_probability",
    "temporal_basis_dim", "boundary_taper", "raw_chain_error", "final_chain_error",
    "reprojection_improvement", "raw_direction_error", "final_direction_error",
    "trust_radius_rad", "optimizer_steps", "depth_target_mm", "depth_actual_mm",
    "final_total_loss", "final_reprojection_loss", "final_direction_loss",
    "final_temporal_pose_loss", "final_world_motion_loss", "final_depth_loss",
    "final_anchor_loss", "body_rms_from_raw", "body_max_from_raw",
    "camera_joint_rms_from_raw_mm", "pelvis_relative_joint_rms_from_raw_mm",
    "world_joint_rms_from_raw_mm", "world_endpoint_rms_from_raw_mm",
    "mean_depth_delta_from_raw_mm", "max_depth_delta_from_raw_mm",
    "pairwise_2d_rms_from_raw_norm", "external_reprojection_error",
    "temporal_history", "world_velocity_excess_mm",
    "world_acceleration_excess_mm", "world_jerk_excess_mm",
    "endpoint_velocity_excess_mm", "endpoint_acceleration_excess_mm",
    "depth_jump_excess_mm", "limb_angle_jump_excess_deg",
    "joint_limit_excess_ratio", "physics_penalty_proxy",
    "mpjpe", "pa_mpjpe", "pve", "pa_pve", "metric_mpjpe", "metric_pve",
    "world_mpjpe", "world_pve", "world_root_error",
    "delta_mpjpe_from_raw", "delta_pve_from_raw",
    "delta_pa_mpjpe_from_raw", "delta_pa_pve_from_raw",
    "delta_world_mpjpe_from_raw", "delta_world_pve_from_raw",
    "proposal_probability", "p_2d", "p_temporal", "p_occlusion",
    "p_disagreement", "p_depth", "p_mask",
    "constraint_kind", "constraint_target_mm", "constraint_actual_mm",
    "rigid_transform_invariance_error_mm",
    "world_reconstruction_error_mm",
    "frozen_contract_ok",
]

H0_PAIR_FIELDS = [
    "sequence", "frame_id", "pred_id", "gt_id", "chain", "proposal_rank",
    "candidate_a", "candidate_b", "pair_2d_rms_norm",
    "pair_camera_pelvis_relative_rms_mm", "pair_world_joint_rms_mm",
    "pair_world_endpoint_rms_mm", "pair_distal_depth_difference_mm",
    "near_2d_0025", "near_2d_005", "near_2d_01",
    "different_3d_20mm", "different_3d_40mm", "different_3d_60mm",
    "candidate_a_delta_mpjpe", "candidate_b_delta_mpjpe",
    "candidate_a_delta_pve", "candidate_b_delta_pve",
    "frozen_contract_ok",
]

_H0_TRACKS: Dict[str, Dict[int, Dict[str, Any]]] = {}
_H0_NEXT_TRACK: Dict[str, int] = {}
_H0_FRAME_PROPOSALS: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
_H0_GT_HISTORY: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
_H0V2_FEATURE_HISTORY: Dict[Tuple[str, str, str], List[float]] = {}
_H0V2_LAST_CHAIN_PROPOSAL: Dict[Tuple[str, int, str], int] = {}

H0V2_PROPOSAL_FIELDS = [
    "sequence", "frame_id", "pid", "track_id", "chain",
    "raw_chain_error", "temporal_acc", "occlusion_score",
    "candidate_disagreement", "depth_jump", "mask_outside",
    "p_2d", "p_temporal", "p_occlusion", "p_disagreement",
    "p_depth", "p_mask", "proposal_probability",
    "selected", "proposal_rejection",
]

H0V2_REJECT_FIELDS = [
    "sequence", "frame_id", "pid", "track_id", "chain", "variant",
    "proposal_probability", "reason", "raw_chain_error",
]

H0_VARIANTS = {
    "fit_weak": {
        "trust": 0.10, "steps": 12, "lr": 0.035,
        "w_reproj": 1.00, "w_direction": 0.10,
        "w_temporal": 0.10, "w_world": 0.10,
        "w_depth": 0.00, "w_anchor": 0.55,
        "min_reproj_drop": 0.02, "allow_reproj_ratio": 0.995,
    },
    "fit_mid": {
        "trust": 0.22, "steps": 16, "lr": 0.040,
        "w_reproj": 1.00, "w_direction": 0.12,
        "w_temporal": 0.22, "w_world": 0.22,
        "w_depth": 0.00, "w_anchor": 0.16,
        "min_reproj_drop": 0.05, "allow_reproj_ratio": 0.98,
    },
    "motion": {
        "trust": 0.26, "steps": 16, "lr": 0.038,
        "w_reproj": 0.90, "w_direction": 0.12,
        "w_temporal": 0.55, "w_world": 0.70,
        "w_depth": 0.00, "w_anchor": 0.12,
        "min_reproj_drop": 0.00, "allow_reproj_ratio": 1.02,
    },
    "depth_near": {
        "trust": 0.32, "steps": 18, "lr": 0.035,
        "w_reproj": 1.00, "w_direction": 0.15,
        "w_temporal": 0.22, "w_world": 0.18,
        "w_depth": 0.65, "w_anchor": 0.10,
        "min_reproj_drop": -0.03, "allow_reproj_ratio": 1.035,
        "depth_sign": -1.0,
    },
    "depth_far": {
        "trust": 0.32, "steps": 18, "lr": 0.035,
        "w_reproj": 1.00, "w_direction": 0.15,
        "w_temporal": 0.22, "w_world": 0.18,
        "w_depth": 0.65, "w_anchor": 0.10,
        "min_reproj_drop": -0.03, "allow_reproj_ratio": 1.035,
        "depth_sign": 1.0,
    },
}

H0V2_VARIANTS = {
    "fit_mid": {
        "trust": 0.24, "steps": 18, "lr": 0.038,
        "w_reproj": 1.00, "w_direction": 0.12,
        "w_temporal": 0.24, "w_world": 0.24,
        "w_constraint": 0.00, "w_anchor": 0.13,
        "allow_reproj_ratio": 1.10,
    },
    "motion": {
        "trust": 0.28, "steps": 18, "lr": 0.036,
        "w_reproj": 0.88, "w_direction": 0.12,
        "w_temporal": 0.62, "w_world": 0.78,
        "w_constraint": 0.00, "w_anchor": 0.10,
        "allow_reproj_ratio": 1.12,
    },
    "arm_depth_near": {
        "trust": 0.34, "steps": 20, "lr": 0.033,
        "w_reproj": 1.00, "w_direction": 0.15,
        "w_temporal": 0.24, "w_world": 0.20,
        "w_constraint": 0.62, "w_anchor": 0.08,
        "allow_reproj_ratio": 1.15, "sign": -1.0,
    },
    "arm_depth_far": {
        "trust": 0.34, "steps": 20, "lr": 0.033,
        "w_reproj": 1.00, "w_direction": 0.15,
        "w_temporal": 0.24, "w_world": 0.20,
        "w_constraint": 0.62, "w_anchor": 0.08,
        "allow_reproj_ratio": 1.15, "sign": 1.0,
    },
    "leg_bend_positive": {
        "trust": 0.30, "steps": 20, "lr": 0.033,
        "w_reproj": 1.00, "w_direction": 0.16,
        "w_temporal": 0.30, "w_world": 0.28,
        "w_constraint": 0.58, "w_anchor": 0.09,
        "allow_reproj_ratio": 1.15, "sign": 1.0,
    },
    "leg_bend_negative": {
        "trust": 0.30, "steps": 20, "lr": 0.033,
        "w_reproj": 1.00, "w_direction": 0.16,
        "w_temporal": 0.30, "w_world": 0.28,
        "w_constraint": 0.58, "w_anchor": 0.09,
        "allow_reproj_ratio": 1.15, "sign": -1.0,
    },
}


def _h0_torch_transform_points(transform, points):
    if transform is None:
        return None
    mat = transform if torch.is_tensor(transform) else torch.as_tensor(
        transform, dtype=points.dtype, device=points.device
    )
    mat = mat.to(dtype=points.dtype, device=points.device)
    if mat.shape == (4, 4):
        ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
        homo = torch.cat([points, ones], dim=-1)
        return torch.einsum("ij,bkj->bki", mat, homo)[..., :3]
    if mat.shape == (3, 4):
        ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
        homo = torch.cat([points, ones], dim=-1)
        return torch.einsum("ij,bkj->bki", mat, homo)
    if mat.shape == (3, 3):
        return torch.einsum("ij,bkj->bki", mat, points)
    return None


def _h0_project_delta(delta, trust):
    with torch.no_grad():
        norms = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(float(trust) / norms, max=1.0)
        delta.mul_(scale)


def _h0_person_geometry(raw_pose, pid, pose_ids, delta, pred, frame_id, smpl_layer, smpl_model, K):
    person_pose = raw_pose[pid:pid + 1].clone()
    person_pose[:, pose_ids] = person_pose[:, pose_ids] + delta.unsqueeze(0)
    shape = pred["shape"][frame_id][pid:pid + 1]
    transl = pred["transl"][frame_id][pid:pid + 1]
    expression = pred["expression"][frame_id][pid:pid + 1]
    Kb = K.unsqueeze(0) if K.ndim == 2 else K[pid:pid + 1]
    out = smpl_layer(
        person_pose, shape, transl, None, None, K=Kb, expression=expression,
    )
    vx = out["smpl_v3d"]
    vertices = smpl_model.smplx2smpl.to(vx) @ vx
    joints = smpl_model.j_regressor.to(vertices) @ vertices
    j2d = torch.einsum(
        "bij,bkj->bki", Kb, joints / joints[..., 2:3].clamp_min(1e-4)
    )[..., :2]
    transform = pred.get("T_c2w", [None])[frame_id]
    world = _h0_torch_transform_points(transform, joints)
    return person_pose, vertices, joints, j2d, world


def _h0_direction_loss(points, target):
    losses = []
    for a, b in ((0, 1), (1, 2)):
        u = points[b] - points[a]
        v = target[b] - target[a]
        denom = torch.linalg.norm(u).clamp_min(1e-6) * torch.linalg.norm(v).clamp_min(1e-6)
        losses.append(1.0 - torch.clamp(torch.dot(u, v) / denom, -1.0, 1.0))
    return torch.stack(losses).mean()


def _h0_rotation_geodesic_loss(current, target):
    try:
        import roma
        rc = roma.rotvec_to_rotmat(current.reshape(-1, 3))
        rt = roma.rotvec_to_rotmat(target.reshape(-1, 3))
        rel = torch.matmul(rc.transpose(-1, -2), rt)
        cosine = ((torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(cosine).square().mean()
    except Exception:
        return F.smooth_l1_loss(current, target, beta=0.08)


def _h0_track_assignments(sequence, frame_id, raw_j2d, width, height):
    boxes = [_bbox(points) for points in _safe_numpy(raw_j2d)]
    tracks = _H0_TRACKS.setdefault(sequence, {})
    active = [tid for tid, state in tracks.items() if int(frame_id) - int(state.get("frame", -9)) == 1]
    current = [pid for pid, box in enumerate(boxes) if box is not None]
    assignments = [None] * len(boxes)
    if active and current:
        cost = np.full((len(current), len(active)), 100.0, dtype=np.float32)
        meta = {}
        for r, pid in enumerate(current):
            for c, tid in enumerate(active):
                iou = _iou(boxes[pid], tracks[tid]["bbox"])
                center = _center(boxes[pid], tracks[tid]["bbox"], width, height)
                cost[r, c] = 1.0 - iou + 2.0 * center
                meta[(r, c)] = (iou, center)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            iou, center = meta[(r, c)]
            if iou >= 0.18 and center <= 0.09:
                assignments[current[r]] = active[c]
    next_id = _H0_NEXT_TRACK.setdefault(sequence, 0)
    for pid in current:
        if assignments[pid] is None:
            assignments[pid] = next_id
            next_id += 1
    _H0_NEXT_TRACK[sequence] = next_id
    return assignments, boxes


def _h0_history_for_track(sequence, track_id):
    state = _H0_TRACKS.get(sequence, {}).get(track_id)
    if state is None:
        return {"depth": 0, "poses": [], "world": [], "cam": []}
    return {
        "depth": len(state.get("poses", [])),
        "poses": list(state.get("poses", [])),
        "world": list(state.get("world", [])),
        "cam": list(state.get("cam", [])),
    }


def _h0_update_track_history(sequence, frame_id, assignments, boxes, raw_pose, raw_joints, raw_world):
    tracks = _H0_TRACKS.setdefault(sequence, {})
    pose_np = _safe_numpy(raw_pose)
    joints_np = _safe_numpy(raw_joints)
    world_np = None if raw_world is None else _safe_numpy(raw_world)
    for pid, tid in enumerate(assignments):
        if tid is None or pid >= len(pose_np) or boxes[pid] is None:
            continue
        old = tracks.get(tid, {})
        poses = list(old.get("poses", []))
        cams = list(old.get("cam", []))
        worlds = list(old.get("world", []))
        poses.append(np.array(pose_np[pid], copy=True))
        cams.append(np.array(joints_np[pid], copy=True))
        if world_np is not None and pid < len(world_np):
            worlds.append(np.array(world_np[pid], copy=True))
        tracks[tid] = {
            "frame": int(frame_id), "bbox": np.array(boxes[pid], copy=True),
            "poses": poses[-3:], "cam": cams[-3:], "world": worlds[-3:],
            "last_proposal": old.get("last_proposal", -100000),
            "last_proposal_by_chain": dict(old.get("last_proposal_by_chain", {})),
        }


def _h0_temporal_pose_target(history, pose_ids, device, dtype):
    poses = history.get("poses", [])
    if not poses:
        return None
    last = torch.as_tensor(poses[-1][pose_ids], dtype=dtype, device=device)
    if len(poses) >= 2:
        prev = torch.as_tensor(poses[-2][pose_ids], dtype=dtype, device=device)
        return last + (last - prev)
    return last


def _h0_world_motion_target(history, joint_ids, device, dtype):
    worlds = history.get("world", [])
    if not worlds:
        return None
    last = torch.as_tensor(worlds[-1][joint_ids], dtype=dtype, device=device)
    if len(worlds) >= 2:
        prev = torch.as_tensor(worlds[-2][joint_ids], dtype=dtype, device=device)
        return last + (last - prev)
    return last


def _h0_depth_preserving_direction(raw_pose, pid, pose_ids, joint_ids, pred, frame_id, smpl_layer, smpl_model, K, scale):
    delta = torch.zeros((len(pose_ids), 3), dtype=raw_pose.dtype, device=raw_pose.device, requires_grad=True)
    _, _, joints, j2d, _ = _h0_person_geometry(
        raw_pose, pid, pose_ids, delta, pred, frame_id, smpl_layer, smpl_model, K
    )
    outputs = (j2d[0, joint_ids] / max(float(scale), 1.0)).reshape(-1)
    jac_rows = []
    for value in outputs:
        grad = torch.autograd.grad(value, delta, retain_graph=True, create_graph=False)[0]
        jac_rows.append(grad.reshape(-1))
    distal_z = joints[0, joint_ids[-1], 2]
    depth_grad = torch.autograd.grad(distal_z, delta, retain_graph=False, create_graph=False)[0].reshape(-1)
    J = torch.stack(jac_rows, dim=0)
    damping = float(os.environ.get("WAO_HMR_H0_DEPTH_DAMPING", "0.020"))
    system = J.T @ J + damping * torch.eye(J.shape[1], dtype=J.dtype, device=J.device)
    try:
        direction = torch.linalg.solve(system, depth_grad)
    except RuntimeError:
        direction = torch.linalg.pinv(system) @ depth_grad
    norm = torch.linalg.norm(direction)
    if not torch.isfinite(norm) or float(norm.detach().cpu()) < 1e-8:
        return None
    direction = direction / norm
    signed = torch.dot(depth_grad, direction)
    if float(signed.detach().cpu()) < 0:
        direction = -direction
    return direction.reshape(len(pose_ids), 3).detach()


def _h0_final_losses(delta, raw_pose, pid, pose_ids, joint_ids, target, weights, scale, pred, frame_id, smpl_layer, smpl_model, K, history, variant_cfg, depth_target, depth_reference):
    _, _, joints, j2d, world = _h0_person_geometry(
        raw_pose, pid, pose_ids, delta, pred, frame_id, smpl_layer, smpl_model, K
    )
    points = j2d[0, joint_ids]
    residual = (points - target) / max(float(scale), 1.0)
    dist = torch.linalg.norm(residual, dim=-1)
    reproj = torch.sum(weights * F.smooth_l1_loss(dist, torch.zeros_like(dist), reduction="none", beta=0.02))
    direction = _h0_direction_loss(points, target)
    local = raw_pose[pid, pose_ids] + delta
    temporal_target = _h0_temporal_pose_target(history, pose_ids, delta.device, delta.dtype)
    temporal = torch.zeros((), dtype=delta.dtype, device=delta.device)
    if temporal_target is not None:
        temporal = _h0_rotation_geodesic_loss(local, temporal_target)
    world_loss = torch.zeros((), dtype=delta.dtype, device=delta.device)
    world_target = _h0_world_motion_target(history, joint_ids, delta.device, delta.dtype)
    if world is not None and world_target is not None:
        cand_world = world[0, joint_ids]
        limb_scale = torch.linalg.norm(cand_world[-1] - cand_world[0]).detach().clamp_min(0.15)
        world_loss = F.smooth_l1_loss((cand_world - world_target) / limb_scale, torch.zeros_like(cand_world), beta=0.05)
    depth_loss = torch.zeros((), dtype=delta.dtype, device=delta.device)
    if depth_target is not None and depth_reference is not None:
        raw_z, limb_scale = depth_reference
        achieved = joints[0, joint_ids[-1], 2] - raw_z
        depth_loss = ((achieved - depth_target) / limb_scale).square()
    anchor = torch.mean((delta / max(float(variant_cfg["trust"]), 1e-4)) ** 2)
    local_norm = torch.linalg.norm(local, dim=-1)
    limit = torch.relu(local_norm - 2.6).square().mean()
    total = (
        variant_cfg["w_reproj"] * reproj
        + variant_cfg["w_direction"] * direction
        + variant_cfg["w_temporal"] * temporal
        + variant_cfg["w_world"] * world_loss
        + variant_cfg["w_depth"] * depth_loss
        + variant_cfg["w_anchor"] * anchor
        + 0.20 * limit
    )
    return total, {
        "reprojection": reproj, "direction": direction, "temporal": temporal,
        "world": world_loss, "depth": depth_loss, "anchor": anchor,
    }, joints, j2d, world


def _h0_optimize_variant(*, raw_pose, pid, chain_name, chain, target_np, conf_np, scale, pred, frame_id, smpl_layer, smpl_model, K, history, variant, depth_direction):
    cfg = H0_VARIANTS[variant]
    pose_ids = list(chain["pose"])
    joint_ids = list(chain["j14"])
    device, dtype = raw_pose.device, raw_pose.dtype
    target = torch.as_tensor(target_np, dtype=dtype, device=device)
    weights = torch.as_tensor(conf_np, dtype=dtype, device=device)
    weights = weights / weights.sum().clamp_min(1e-6)
    init = torch.zeros((len(pose_ids), 3), dtype=dtype, device=device)
    if variant == "motion" and history.get("depth", 0) > 0:
        temporal_target = _h0_temporal_pose_target(history, pose_ids, device, dtype)
        if temporal_target is not None:
            init = temporal_target - raw_pose[pid, pose_ids]
    depth_target = None
    depth_reference = None
    if variant in {"depth_near", "depth_far"}:
        if depth_direction is None:
            return None
        init = depth_direction * min(0.12, float(cfg["trust"]) * 0.55) * float(cfg["depth_sign"])
        with torch.no_grad():
            zero = torch.zeros_like(init)
            _, _, raw_joints, _, _ = _h0_person_geometry(
                raw_pose, pid, pose_ids, zero, pred, frame_id, smpl_layer, smpl_model, K
            )
            limb = float(torch.linalg.norm(raw_joints[0, joint_ids[-1]] - raw_joints[0, joint_ids[0]]).detach().cpu())
        depth_mag = min(max(0.12 * limb, 0.035), 0.080)
        depth_target = torch.tensor(float(cfg["depth_sign"]) * depth_mag, dtype=dtype, device=device)
        depth_reference = (
            raw_joints[0, joint_ids[-1], 2].detach(),
            torch.linalg.norm(raw_joints[0, joint_ids[-1]] - raw_joints[0, joint_ids[0]]).detach().clamp_min(0.15),
        )
    _h0_project_delta(init, cfg["trust"])
    delta = init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=float(cfg["lr"]))
    last_parts = None
    for _ in range(int(cfg["steps"])):
        optimizer.zero_grad(set_to_none=True)
        total, parts, _, _, _ = _h0_final_losses(
            delta, raw_pose, pid, pose_ids, joint_ids, target, weights, scale,
            pred, frame_id, smpl_layer, smpl_model, K, history, cfg, depth_target, depth_reference,
        )
        if not torch.isfinite(total):
            return None
        total.backward()
        torch.nn.utils.clip_grad_norm_([delta], 2.0)
        optimizer.step()
        _h0_project_delta(delta, cfg["trust"])
        last_parts = parts
    with torch.no_grad():
        total, parts, joints, j2d, world = _h0_final_losses(
            delta, raw_pose, pid, pose_ids, joint_ids, target, weights, scale,
            pred, frame_id, smpl_layer, smpl_model, K, history, cfg, depth_target, depth_reference,
        )
        candidate = raw_pose.detach().clone()
        candidate[pid, pose_ids] = candidate[pid, pose_ids] + delta.detach()
        raw_zero = torch.zeros_like(delta)
        _, _, raw_joints, raw_j2d, _ = _h0_person_geometry(
            raw_pose, pid, pose_ids, raw_zero, pred, frame_id, smpl_layer, smpl_model, K
        )
    raw_points = _safe_numpy(raw_j2d[0, joint_ids])
    cand_points = _safe_numpy(j2d[0, joint_ids])
    raw_error = _chain_error(raw_points, target_np, conf_np, scale)
    final_error = _chain_error(cand_points, target_np, conf_np, scale)
    raw_direction = _direction_error(raw_points, target_np)
    final_direction = _direction_error(cand_points, target_np)
    improvement = (raw_error - final_error) / max(raw_error, 1e-6)
    body_rms, body_max = _person_body_delta(candidate, raw_pose, pid)
    raw_z = float(raw_joints[0, joint_ids[-1], 2].detach().cpu())
    final_z = float(joints[0, joint_ids[-1], 2].detach().cpu())
    depth_actual_mm = (final_z - raw_z) * 1000.0
    depth_target_mm = float(depth_target.detach().cpu()) * 1000.0 if depth_target is not None else 0.0
    local_norm = torch.linalg.norm(candidate[pid, pose_ids], dim=-1)
    if not np.isfinite([body_rms, body_max, final_error, final_direction]).all():
        return None
    min_delta = 0.010 if variant == "fit_weak" else 0.018
    if body_max < min_delta or body_max > float(cfg["trust"]) + 1e-4:
        return None
    if final_error > raw_error * float(cfg["allow_reproj_ratio"]):
        return None
    if improvement < float(cfg["min_reproj_drop"]):
        return None
    if final_direction > raw_direction + 0.04:
        return None
    if float(local_norm.max().detach().cpu()) > 2.85:
        return None
    if variant.startswith("depth_") and abs(depth_actual_mm) < 10.0:
        return None
    return {
        "pose": candidate.detach(), "pid": int(pid), "chain": chain_name,
        "variant": variant, "trust_radius_rad": float(cfg["trust"]),
        "optimizer_steps": int(cfg["steps"]), "raw_chain_error": raw_error,
        "final_chain_error": final_error, "reprojection_improvement": improvement,
        "raw_direction_error": raw_direction, "final_direction_error": final_direction,
        "depth_target_mm": depth_target_mm, "depth_actual_mm": depth_actual_mm,
        "body_rms": body_rms, "body_max": body_max,
        "final_total_loss": float(total.detach().cpu()),
        "final_reprojection_loss": float(parts["reprojection"].detach().cpu()),
        "final_direction_loss": float(parts["direction"].detach().cpu()),
        "final_temporal_pose_loss": float(parts["temporal"].detach().cpu()),
        "final_world_motion_loss": float(parts["world"].detach().cpu()),
        "final_depth_loss": float(parts["depth"].detach().cpu()),
        "final_anchor_loss": float(parts["anchor"].detach().cpu()),
    }


def _h0_generate_frame_candidates(*, sequence, frame_id, image, pred, raw_pose, raw_pred_j2d, smpl_layer, smpl_model, K, a2_frame_class):
    height, width = int(image.shape[-2]), int(image.shape[-1])
    raw_j2d_np = _safe_numpy(raw_pred_j2d)
    assignments, boxes = _h0_track_assignments(sequence, frame_id, raw_j2d_np, width, height)
    raw_vertices, raw_joints, _ = _candidate_geometry(raw_pose, pred, frame_id, smpl_layer, smpl_model, K)
    if raw_joints is None:
        return []
    transform = pred.get("T_c2w", [None])[frame_id]
    raw_world = _h0_torch_transform_points(transform, raw_joints)
    detections = _detections(sequence, frame_id, height, width)
    matches = _match_external(raw_j2d_np, detections, height, width)
    proposals = []
    class_boost = {"high": 1.0, "bad": 0.65, "neighbor": 0.25, "normal": 0.0}.get(a2_frame_class, 0.0)
    for match in matches:
        pid = int(match["pid"])
        if pid >= len(assignments) or assignments[pid] is None:
            continue
        tid = assignments[pid]
        history = _h0_history_for_track(sequence, tid)
        det = detections[match["did"]]
        diag = max(float(np.hypot(det["bbox"][2] - det["bbox"][0], det["bbox"][3] - det["bbox"][1])), 1.0)
        for name, chain in CHAINS.items():
            conf = det["kp"][chain["coco"], 2]
            min_conf = 0.55 if chain["kind"] == "arm" else 0.65
            if np.any(conf < min_conf):
                continue
            target = det["kp"][chain["coco"], :2]
            points = raw_j2d_np[pid, chain["j14"]]
            error = _chain_error(points, target, conf, diag)
            if error < 0.045 or error > 0.35:
                continue
            raw_chain = _safe_numpy(raw_pose)[pid, chain["pose"]]
            temporal_acc = 0.0
            if history["depth"] >= 2:
                last = history["poses"][-1][chain["pose"]]
                prev = history["poses"][-2][chain["pose"]]
                temporal_acc = _rms(raw_chain - last, last - prev)
            error_term = float(np.clip((error - 0.045) / 0.12, 0.0, 2.0))
            temporal_term = float(np.clip(temporal_acc / 0.025, 0.0, 2.0))
            reliability = float(np.mean(conf)) * float(det["score"]) * math.sqrt(max(float(match["iou"]), 1e-6))
            risk = reliability * (1.35 * error_term + 0.75 * temporal_term + class_boost)
            if risk < float(os.environ.get("WAO_HMR_H0_MIN_RISK", "0.95")):
                continue
            if a2_frame_class == "normal" and error < 0.070 and temporal_acc < 0.012:
                continue
            proposals.append({
                "pid": pid, "track_id": tid, "chain": name, "config": chain,
                "target": target, "conf": conf, "scale": diag,
                "raw_error": error, "temporal_acc": temporal_acc,
                "risk": risk, "history": history,
            })
    proposals.sort(key=lambda item: item["risk"], reverse=True)
    state = _SEQUENCE_STATES.get(sequence, {})
    budget = max(24, int(math.ceil(float(state.get("n_frames", 1)) * float(os.environ.get("WAO_HMR_H0_SEQUENCE_RATIO", "0.06")))))
    used = int(state.get("h0_proposals", 0))
    max_frame = int(os.environ.get("WAO_HMR_H0_MAX_PROPOSALS_PER_FRAME", "1"))
    min_gap = int(os.environ.get("WAO_HMR_H0_MIN_TRACK_GAP", "3"))
    selected = []
    used_pids = set()
    for proposal in proposals:
        if used + len(selected) >= budget or len(selected) >= max_frame:
            break
        if proposal["pid"] in used_pids:
            continue
        track_state = _H0_TRACKS.get(sequence, {}).get(proposal["track_id"], {})
        if int(frame_id) - int(track_state.get("last_proposal", -100000)) < min_gap:
            continue
        selected.append(proposal)
        used_pids.add(proposal["pid"])
    generated = []
    for rank, proposal in enumerate(selected):
        chain = proposal["config"]
        depth_direction = _h0_depth_preserving_direction(
            raw_pose, proposal["pid"], chain["pose"], chain["j14"], pred, frame_id,
            smpl_layer, smpl_model, K, proposal["scale"],
        )
        variants = ["fit_weak", "fit_mid"]
        if proposal["history"]["depth"] >= 1:
            variants.append("motion")
        if depth_direction is not None:
            variants.extend(["depth_near", "depth_far"])
        for variant in variants:
            result = _h0_optimize_variant(
                raw_pose=raw_pose, pid=proposal["pid"], chain_name=proposal["chain"],
                chain=chain, target_np=proposal["target"], conf_np=proposal["conf"],
                scale=proposal["scale"], pred=pred, frame_id=frame_id,
                smpl_layer=smpl_layer, smpl_model=smpl_model, K=K,
                history=proposal["history"], variant=variant,
                depth_direction=depth_direction,
            )
            if result is None:
                continue
            result.update({
                "proposal_rank": int(rank), "track_id": int(proposal["track_id"]),
                "history_depth": int(proposal["history"]["depth"]),
                "risk_score": float(proposal["risk"]),
            })
            generated.append(result)
        track_state = _H0_TRACKS.get(sequence, {}).get(proposal["track_id"])
        if track_state is not None:
            track_state["last_proposal"] = int(frame_id)
    state["h0_proposals"] = used + len(selected)
    state["h0_candidates"] = int(state.get("h0_candidates", 0)) + len(generated)
    _h0_update_track_history(
        sequence, frame_id, assignments, boxes, raw_pose, raw_joints, raw_world,
    )
    return generated



def _h0v2_static_feature_score(name, value):
    value = float(value)
    if name == "error":
        return float(np.clip((value - 0.025) / 0.120, 0.0, 1.0))
    if name == "temporal":
        return float(np.clip(value / 0.035, 0.0, 1.0))
    if name == "occlusion":
        return float(np.clip(value, 0.0, 1.0))
    if name == "disagreement":
        return float(np.clip(value / 0.065, 0.0, 1.0))
    if name == "depth":
        return float(np.clip(value / 0.080, 0.0, 1.0))
    if name == "mask":
        return float(np.clip(value, 0.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def _h0v2_calibrated_score(sequence, chain, name, value):
    key = (sequence, chain, name)
    history = _H0V2_FEATURE_HISTORY.setdefault(key, [])
    value = float(value)
    if len(history) >= 20:
        arr = np.asarray(history, dtype=np.float64)
        empirical = float((np.sum(arr < value) + 0.5 * np.sum(arr == value)) / len(arr))
        score = 0.75 * empirical + 0.25 * _h0v2_static_feature_score(name, value)
    else:
        score = _h0v2_static_feature_score(name, value)
    history.append(value)
    if len(history) > 512:
        del history[:-512]
    return float(np.clip(score, 0.0, 1.0))


def _h0v2_mask_outside_score(pid, chain, raw_j2d_np, person_masks, height, width):
    item = person_masks.get(pid)
    if item is None:
        return 0.0
    mask = np.asarray(item["mask"], dtype=np.uint8) > 0
    points = np.asarray(raw_j2d_np[pid, chain["j14"]], dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return 0.0
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    valid = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[y[valid], x[valid]]
    return float(1.0 - inside[finite].mean())


def _h0v2_pose_disagreement(pid, chain, raw_pose, candidate_poses):
    raw = _safe_numpy(raw_pose)
    values = []
    for pose in candidate_poses.values():
        if pose is None:
            continue
        arr = _safe_numpy(pose)
        if arr.shape != raw.shape or pid >= len(arr):
            continue
        values.append(_rms(arr[pid, chain["pose"]], raw[pid, chain["pose"]]))
    return float(max(values or [0.0]))


def _h0v2_depth_jump(history, raw_joints_np, joint_ids):
    cams = history.get("cam", [])
    if len(cams) < 2:
        return 0.0
    current = np.asarray(raw_joints_np[joint_ids], dtype=np.float64)
    last = np.asarray(cams[-1][joint_ids], dtype=np.float64)
    prev = np.asarray(cams[-2][joint_ids], dtype=np.float64)
    predicted = last + (last - prev)
    limb = max(float(np.linalg.norm(current[-1] - current[0])), 0.15)
    return float(abs(current[-1, 2] - predicted[-1, 2]) / limb)


def _h0v2_bend_preserving_direction(raw_pose, pid, pose_ids, joint_ids, pred, frame_id, smpl_layer, smpl_model, K, scale):
    delta = torch.zeros((len(pose_ids), 3), dtype=raw_pose.dtype, device=raw_pose.device, requires_grad=True)
    _, _, joints, j2d, _ = _h0_person_geometry(
        raw_pose, pid, pose_ids, delta, pred, frame_id, smpl_layer, smpl_model, K
    )
    xyz = joints[0, joint_ids]
    hip, knee, ankle = xyz[0], xyz[1], xyz[2]
    u = knee - hip
    v = ankle - knee
    normal = torch.cross(u, v, dim=0)
    if float(torch.linalg.norm(normal).detach().cpu()) < 1e-5:
        camera_z = torch.tensor([0.0, 0.0, 1.0], dtype=xyz.dtype, device=xyz.device)
        normal = torch.cross(ankle - hip, camera_z, dim=0)
    normal = normal / torch.linalg.norm(normal).clamp_min(1e-6)
    chord = ankle - hip
    bend_axis = torch.cross(normal, chord, dim=0)
    bend_axis = bend_axis / torch.linalg.norm(bend_axis).clamp_min(1e-6)
    midpoint = 0.5 * (hip + ankle)
    if torch.dot(knee - midpoint, bend_axis) < 0:
        bend_axis = -bend_axis

    outputs = (j2d[0, joint_ids] / max(float(scale), 1.0)).reshape(-1)
    jac_rows = []
    for value in outputs:
        grad = torch.autograd.grad(value, delta, retain_graph=True, create_graph=False)[0]
        jac_rows.append(grad.reshape(-1))
    scalar = torch.dot(knee, bend_axis.detach())
    target_grad = torch.autograd.grad(scalar, delta, retain_graph=False, create_graph=False)[0].reshape(-1)
    J = torch.stack(jac_rows, dim=0)
    damping = float(os.environ.get("WAO_HMR_H0V2_BEND_DAMPING", "0.025"))
    system = J.T @ J + damping * torch.eye(J.shape[1], dtype=J.dtype, device=J.device)
    try:
        direction = torch.linalg.solve(system, target_grad)
    except RuntimeError:
        direction = torch.linalg.pinv(system) @ target_grad
    norm = torch.linalg.norm(direction)
    if not torch.isfinite(norm) or float(norm.detach().cpu()) < 1e-8:
        return None, None
    direction = direction / norm
    if torch.dot(target_grad, direction) < 0:
        direction = -direction
    return direction.reshape(len(pose_ids), 3).detach(), bend_axis.detach()


def _h0v2_final_losses(delta, raw_pose, pid, pose_ids, joint_ids, target, weights, scale,
                       pred, frame_id, smpl_layer, smpl_model, K, history, cfg,
                       constraint_spec):
    _, _, joints, j2d, world = _h0_person_geometry(
        raw_pose, pid, pose_ids, delta, pred, frame_id, smpl_layer, smpl_model, K
    )
    points = j2d[0, joint_ids]
    residual = (points - target) / max(float(scale), 1.0)
    dist = torch.linalg.norm(residual, dim=-1)
    reproj = torch.sum(weights * F.smooth_l1_loss(
        dist, torch.zeros_like(dist), reduction="none", beta=0.02
    ))
    direction = _h0_direction_loss(points, target)
    local = raw_pose[pid, pose_ids] + delta
    temporal_target = _h0_temporal_pose_target(history, pose_ids, delta.device, delta.dtype)
    temporal = torch.zeros((), dtype=delta.dtype, device=delta.device)
    if temporal_target is not None:
        temporal = _h0_rotation_geodesic_loss(local, temporal_target)
    world_loss = torch.zeros((), dtype=delta.dtype, device=delta.device)
    world_target = _h0_world_motion_target(history, joint_ids, delta.device, delta.dtype)
    if world is not None and world_target is not None:
        cand_world = world[0, joint_ids]
        limb_scale = torch.linalg.norm(cand_world[-1] - cand_world[0]).detach().clamp_min(0.15)
        world_loss = F.smooth_l1_loss(
            (cand_world - world_target) / limb_scale,
            torch.zeros_like(cand_world), beta=0.05,
        )
    constraint = torch.zeros((), dtype=delta.dtype, device=delta.device)
    if constraint_spec is not None:
        jlocal = int(constraint_spec["joint_local"])
        axis = constraint_spec["axis"].to(dtype=delta.dtype, device=delta.device)
        raw_point = constraint_spec["raw_point"].to(dtype=delta.dtype, device=delta.device)
        target_value = constraint_spec["target"].to(dtype=delta.dtype, device=delta.device)
        limb_scale = constraint_spec["scale"].to(dtype=delta.dtype, device=delta.device)
        achieved = torch.dot(joints[0, joint_ids[jlocal]] - raw_point, axis)
        constraint = ((achieved - target_value) / limb_scale).square()
    anchor = torch.mean((delta / max(float(cfg["trust"]), 1e-4)) ** 2)
    local_norm = torch.linalg.norm(local, dim=-1)
    limit = torch.relu(local_norm - 2.6).square().mean()
    total = (
        cfg["w_reproj"] * reproj + cfg["w_direction"] * direction
        + cfg["w_temporal"] * temporal + cfg["w_world"] * world_loss
        + cfg["w_constraint"] * constraint + cfg["w_anchor"] * anchor
        + 0.20 * limit
    )
    return total, {
        "reprojection": reproj, "direction": direction, "temporal": temporal,
        "world": world_loss, "constraint": constraint, "anchor": anchor,
    }, joints, j2d, world


def _h0v2_optimize_variant(*, raw_pose, pid, chain_name, chain, target_np, conf_np,
                            scale, pred, frame_id, smpl_layer, smpl_model, K,
                            history, variant, constraint_direction=None,
                            constraint_axis=None):
    cfg = H0V2_VARIANTS[variant]
    pose_ids = list(chain["pose"])
    joint_ids = list(chain["j14"])
    device, dtype = raw_pose.device, raw_pose.dtype
    target = torch.as_tensor(target_np, dtype=dtype, device=device)
    weights = torch.as_tensor(conf_np, dtype=dtype, device=device).clamp_min(0.05)
    weights = weights / weights.sum().clamp_min(1e-6)
    init = torch.zeros((len(pose_ids), 3), dtype=dtype, device=device)
    if variant == "motion" and history.get("depth", 0) > 0:
        temporal_target = _h0_temporal_pose_target(history, pose_ids, device, dtype)
        if temporal_target is not None:
            init = temporal_target - raw_pose[pid, pose_ids]

    constraint_spec = None
    if variant.startswith("arm_depth_") or variant.startswith("leg_bend_"):
        if constraint_direction is None or constraint_axis is None:
            return None, "missing_constraint_direction"
        sign = float(cfg["sign"])
        init = constraint_direction * min(0.14, float(cfg["trust"]) * 0.55) * sign
        with torch.no_grad():
            zero = torch.zeros_like(init)
            _, _, raw_joints, _, _ = _h0_person_geometry(
                raw_pose, pid, pose_ids, zero, pred, frame_id, smpl_layer, smpl_model, K
            )
            limb = torch.linalg.norm(
                raw_joints[0, joint_ids[-1]] - raw_joints[0, joint_ids[0]]
            ).detach().clamp_min(0.15)
        if variant.startswith("arm_depth_"):
            joint_local = 2
            magnitude = min(max(0.12 * float(limb.cpu()), 0.035), 0.080)
            kind = "arm_camera_depth"
        else:
            joint_local = 1
            magnitude = min(max(0.10 * float(limb.cpu()), 0.025), 0.060)
            kind = "leg_local_bend"
        constraint_spec = {
            "joint_local": joint_local,
            "axis": constraint_axis,
            "raw_point": raw_joints[0, joint_ids[joint_local]].detach(),
            "target": torch.tensor(sign * magnitude, dtype=dtype, device=device),
            "scale": limb,
            "kind": kind,
        }

    _h0_project_delta(init, cfg["trust"])
    delta = init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=float(cfg["lr"]))
    for _ in range(int(cfg["steps"])):
        optimizer.zero_grad(set_to_none=True)
        total, parts, _, _, _ = _h0v2_final_losses(
            delta, raw_pose, pid, pose_ids, joint_ids, target, weights, scale,
            pred, frame_id, smpl_layer, smpl_model, K, history, cfg,
            constraint_spec,
        )
        if not torch.isfinite(total):
            return None, "nonfinite_loss"
        total.backward()
        torch.nn.utils.clip_grad_norm_([delta], 2.0)
        optimizer.step()
        _h0_project_delta(delta, cfg["trust"])

    with torch.no_grad():
        total, parts, joints, j2d, world = _h0v2_final_losses(
            delta, raw_pose, pid, pose_ids, joint_ids, target, weights, scale,
            pred, frame_id, smpl_layer, smpl_model, K, history, cfg,
            constraint_spec,
        )
        candidate = raw_pose.detach().clone()
        candidate[pid, pose_ids] = candidate[pid, pose_ids] + delta.detach()
        zero = torch.zeros_like(delta)
        _, _, raw_joints, raw_j2d, _ = _h0_person_geometry(
            raw_pose, pid, pose_ids, zero, pred, frame_id, smpl_layer, smpl_model, K
        )

    raw_points = _safe_numpy(raw_j2d[0, joint_ids])
    cand_points = _safe_numpy(j2d[0, joint_ids])
    raw_error = _chain_error(raw_points, target_np, conf_np, scale)
    final_error = _chain_error(cand_points, target_np, conf_np, scale)
    raw_direction = _direction_error(raw_points, target_np)
    final_direction = _direction_error(cand_points, target_np)
    improvement = (raw_error - final_error) / max(raw_error, 1e-6)
    body_rms, body_max = _person_body_delta(candidate, raw_pose, pid)
    local_norm = torch.linalg.norm(candidate[pid, pose_ids], dim=-1)
    constraint_target_mm = 0.0
    constraint_actual_mm = 0.0
    constraint_kind = "none"
    if constraint_spec is not None:
        jlocal = int(constraint_spec["joint_local"])
        axis = constraint_spec["axis"]
        raw_point = constraint_spec["raw_point"]
        constraint_actual_mm = float(torch.dot(
            joints[0, joint_ids[jlocal]] - raw_point, axis
        ).detach().cpu()) * 1000.0
        constraint_target_mm = float(constraint_spec["target"].detach().cpu()) * 1000.0
        constraint_kind = str(constraint_spec["kind"])

    if not np.isfinite([body_rms, body_max, final_error, final_direction]).all():
        return None, "nonfinite_geometry"
    if body_max < 0.012:
        return None, "delta_too_small"
    if body_max > float(cfg["trust"]) + 1e-4:
        return None, "trust_region_violation"
    catastrophe = max(raw_error * float(cfg["allow_reproj_ratio"]), raw_error + 0.015)
    if final_error > catastrophe:
        return None, "catastrophic_reprojection"
    if final_direction > raw_direction + 0.12:
        return None, "catastrophic_direction"
    if float(local_norm.max().detach().cpu()) > 2.95:
        return None, "joint_limit"
    if constraint_spec is not None:
        if abs(constraint_actual_mm) < 8.0:
            return None, "constraint_too_small"
        if constraint_actual_mm * constraint_target_mm <= 0.0:
            return None, "constraint_wrong_sign"

    depth_target_mm = constraint_target_mm if variant.startswith("arm_depth_") else 0.0
    depth_actual_mm = constraint_actual_mm if variant.startswith("arm_depth_") else 0.0
    return {
        "pose": candidate.detach(), "pid": int(pid), "chain": chain_name,
        "variant": variant, "trust_radius_rad": float(cfg["trust"]),
        "optimizer_steps": int(cfg["steps"]), "raw_chain_error": raw_error,
        "final_chain_error": final_error, "reprojection_improvement": improvement,
        "raw_direction_error": raw_direction, "final_direction_error": final_direction,
        "depth_target_mm": depth_target_mm, "depth_actual_mm": depth_actual_mm,
        "constraint_kind": constraint_kind,
        "constraint_target_mm": constraint_target_mm,
        "constraint_actual_mm": constraint_actual_mm,
        "body_rms": body_rms, "body_max": body_max,
        "final_total_loss": float(total.detach().cpu()),
        "final_reprojection_loss": float(parts["reprojection"].detach().cpu()),
        "final_direction_loss": float(parts["direction"].detach().cpu()),
        "final_temporal_pose_loss": float(parts["temporal"].detach().cpu()),
        "final_world_motion_loss": float(parts["world"].detach().cpu()),
        "final_depth_loss": float(parts["constraint"].detach().cpu()),
        "final_anchor_loss": float(parts["anchor"].detach().cpu()),
    }, "accepted"


def _h0v2_write_rows(filename, fields, rows):
    if not rows:
        return
    output = _run_root()
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _h0v2_generate_frame_candidates(*, sequence, frame_id, image, pred, raw_pose,
                                     raw_pred_j2d, smpl_layer, smpl_model, K,
                                     a2_frame_class, candidate_poses):
    height, width = int(image.shape[-2]), int(image.shape[-1])
    raw_j2d_np = _safe_numpy(raw_pred_j2d)
    assignments, boxes = _h0_track_assignments(sequence, frame_id, raw_j2d_np, width, height)
    raw_vertices, raw_joints, _ = _candidate_geometry(
        raw_pose, pred, frame_id, smpl_layer, smpl_model, K
    )
    if raw_joints is None:
        return []
    raw_joints_np = _safe_numpy(raw_joints)
    transform = pred.get("T_c2w", [None])[frame_id]
    raw_world = _h0_torch_transform_points(transform, raw_joints)
    detections = _detections(sequence, frame_id, height, width)
    matches = _match_external(raw_j2d_np, detections, height, width)
    masks = _sam3_instance_masks(sequence, frame_id, height, width)
    person_masks = _match_raw_people_to_masks(raw_pred_j2d, masks, height, width)
    class_boost = {"high": 0.18, "bad": 0.12, "neighbor": 0.05, "normal": 0.0}.get(a2_frame_class, 0.0)
    proposal_rows = []
    proposals = []

    for match in matches:
        pid = int(match["pid"])
        if pid >= len(assignments) or assignments[pid] is None:
            continue
        tid = int(assignments[pid])
        history = _h0_history_for_track(sequence, tid)
        det = detections[match["did"]]
        diag = max(float(np.hypot(
            det["bbox"][2] - det["bbox"][0], det["bbox"][3] - det["bbox"][1]
        )), 1.0)
        for name, chain in CHAINS.items():
            conf = np.asarray(det["kp"][chain["coco"], 2], dtype=np.float32)
            if int(np.sum(conf >= 0.25)) < 2 or float(np.mean(conf)) < 0.35:
                continue
            target = det["kp"][chain["coco"], :2]
            points = raw_j2d_np[pid, chain["j14"]]
            error = _chain_error(points, target, conf, diag)
            if not np.isfinite(error) or error > 0.45:
                continue
            raw_chain = _safe_numpy(raw_pose)[pid, chain["pose"]]
            temporal_acc = 0.0
            if history["depth"] >= 2:
                last = history["poses"][-1][chain["pose"]]
                prev = history["poses"][-2][chain["pose"]]
                temporal_acc = _rms(raw_chain - last, last - prev)
            occlusion = float(np.clip(1.0 - np.mean(conf), 0.0, 1.0))
            disagreement = _h0v2_pose_disagreement(pid, chain, raw_pose, candidate_poses)
            depth_jump = _h0v2_depth_jump(history, raw_joints_np[pid], chain["j14"])
            mask_outside = _h0v2_mask_outside_score(
                pid, chain, raw_j2d_np, person_masks, height, width
            )
            p2d = _h0v2_calibrated_score(sequence, name, "error", error)
            ptemp = _h0v2_calibrated_score(sequence, name, "temporal", temporal_acc)
            pocc = _h0v2_calibrated_score(sequence, name, "occlusion", occlusion)
            pdis = _h0v2_calibrated_score(sequence, name, "disagreement", disagreement)
            pdepth = _h0v2_calibrated_score(sequence, name, "depth", depth_jump)
            pmask = _h0v2_calibrated_score(sequence, name, "mask", mask_outside)
            evidence = [
                0.90 * p2d, 0.72 * ptemp, 0.55 * pocc,
                0.78 * pdis, 0.72 * pdepth, 0.50 * pmask,
            ]
            probability = 1.0 - float(np.prod([1.0 - np.clip(x, 0.0, 0.95) for x in evidence]))
            probability = float(np.clip(probability + class_boost, 0.0, 1.0))
            reliability = float(det["score"]) * math.sqrt(max(float(match["iou"]), 1e-6))
            probability *= 0.55 + 0.45 * float(np.clip(reliability, 0.0, 1.0))
            proposal = {
                "pid": pid, "track_id": tid, "chain": name, "config": chain,
                "target": target, "conf": conf, "scale": diag,
                "raw_error": error, "temporal_acc": temporal_acc,
                "risk": probability, "history": history,
                "p_2d": p2d, "p_temporal": ptemp, "p_occlusion": pocc,
                "p_disagreement": pdis, "p_depth": pdepth, "p_mask": pmask,
                "occlusion_score": occlusion,
                "candidate_disagreement": disagreement,
                "depth_jump": depth_jump, "mask_outside": mask_outside,
            }
            proposals.append(proposal)
            proposal_rows.append({
                "sequence": sequence, "frame_id": int(frame_id), "pid": pid,
                "track_id": tid, "chain": name, "raw_chain_error": error,
                "temporal_acc": temporal_acc, "occlusion_score": occlusion,
                "candidate_disagreement": disagreement, "depth_jump": depth_jump,
                "mask_outside": mask_outside, "p_2d": p2d,
                "p_temporal": ptemp, "p_occlusion": pocc,
                "p_disagreement": pdis, "p_depth": pdepth, "p_mask": pmask,
                "proposal_probability": probability, "selected": 0,
                "proposal_rejection": "below_probability",
            })

    proposals.sort(key=lambda item: item["risk"], reverse=True)
    state = _SEQUENCE_STATES.get(sequence, {})
    budget = max(48, int(math.ceil(
        float(state.get("n_frames", 1))
        * float(os.environ.get("WAO_HMR_H0V2_SEQUENCE_RATIO", "0.18"))
    )))
    used = int(state.get("h0_proposals", 0))
    max_frame = int(os.environ.get("WAO_HMR_H0V2_MAX_PROPOSALS_PER_FRAME", "2"))
    max_person = int(os.environ.get("WAO_HMR_H0V2_MAX_CHAINS_PER_PERSON", "2"))
    min_gap = int(os.environ.get("WAO_HMR_H0V2_MIN_CHAIN_GAP", "2"))
    threshold = float(os.environ.get("WAO_HMR_H0V2_MIN_PROBABILITY", "0.52"))
    selected = []
    per_pid = Counter()
    selected_keys = set()
    for proposal in proposals:
        if used + len(selected) >= budget or len(selected) >= max_frame:
            break
        if proposal["risk"] < threshold:
            continue
        pid = int(proposal["pid"])
        if per_pid[pid] >= max_person:
            continue
        key = (sequence, int(proposal["track_id"]), proposal["chain"])
        last = _H0V2_LAST_CHAIN_PROPOSAL.get(key, -100000)
        if int(frame_id) - int(last) < min_gap:
            continue
        selected.append(proposal)
        per_pid[pid] += 1
        selected_keys.add((pid, proposal["chain"]))
        _H0V2_LAST_CHAIN_PROPOSAL[key] = int(frame_id)

    for row in proposal_rows:
        if (int(row["pid"]), row["chain"]) in selected_keys:
            row["selected"] = 1
            row["proposal_rejection"] = "selected"
        elif float(row["proposal_probability"]) >= threshold:
            row["proposal_rejection"] = "budget_or_temporal_nms"
    _h0v2_write_rows("h0v2_proposals.csv", H0V2_PROPOSAL_FIELDS, proposal_rows)

    generated = []
    reject_rows = []
    for rank, proposal in enumerate(selected):
        chain = proposal["config"]
        variants = ["fit_mid"]
        if proposal["history"]["depth"] >= 1:
            variants.append("motion")
        constraint_direction = None
        constraint_axis = None
        if chain["kind"] == "arm":
            constraint_direction = _h0_depth_preserving_direction(
                raw_pose, proposal["pid"], chain["pose"], chain["j14"], pred,
                frame_id, smpl_layer, smpl_model, K, proposal["scale"],
            )
            if constraint_direction is not None:
                constraint_axis = torch.tensor(
                    [0.0, 0.0, 1.0], dtype=raw_pose.dtype, device=raw_pose.device
                )
                variants.extend(["arm_depth_near", "arm_depth_far"])
        else:
            constraint_direction, constraint_axis = _h0v2_bend_preserving_direction(
                raw_pose, proposal["pid"], chain["pose"], chain["j14"], pred,
                frame_id, smpl_layer, smpl_model, K, proposal["scale"],
            )
            if constraint_direction is not None:
                variants.extend(["leg_bend_positive", "leg_bend_negative"])

        for variant in variants:
            result, reason = _h0v2_optimize_variant(
                raw_pose=raw_pose, pid=proposal["pid"], chain_name=proposal["chain"],
                chain=chain, target_np=proposal["target"], conf_np=proposal["conf"],
                scale=proposal["scale"], pred=pred, frame_id=frame_id,
                smpl_layer=smpl_layer, smpl_model=smpl_model, K=K,
                history=proposal["history"], variant=variant,
                constraint_direction=constraint_direction,
                constraint_axis=constraint_axis,
            )
            if result is None:
                reject_rows.append({
                    "sequence": sequence, "frame_id": int(frame_id),
                    "pid": int(proposal["pid"]), "track_id": int(proposal["track_id"]),
                    "chain": proposal["chain"], "variant": variant,
                    "proposal_probability": float(proposal["risk"]),
                    "reason": reason, "raw_chain_error": float(proposal["raw_error"]),
                })
                continue
            result.update({
                "proposal_rank": int(rank), "track_id": int(proposal["track_id"]),
                "history_depth": int(proposal["history"]["depth"]),
                "risk_score": float(proposal["risk"]),
                "proposal_probability": float(proposal["risk"]),
                "p_2d": float(proposal["p_2d"]),
                "p_temporal": float(proposal["p_temporal"]),
                "p_occlusion": float(proposal["p_occlusion"]),
                "p_disagreement": float(proposal["p_disagreement"]),
                "p_depth": float(proposal["p_depth"]),
                "p_mask": float(proposal["p_mask"]),
            })
            generated.append(result)
    _h0v2_write_rows("h0v2_candidate_rejections.csv", H0V2_REJECT_FIELDS, reject_rows)
    state["h0_proposals"] = used + len(selected)
    state["h0_candidates"] = int(state.get("h0_candidates", 0)) + len(generated)
    _h0_update_track_history(
        sequence, frame_id, assignments, boxes, raw_pose, raw_joints, raw_world,
    )
    return generated


# ---------------------------------------------------------------------------
# H0-v3 segment-level multi-hypothesis generation.
# ---------------------------------------------------------------------------

_H0V3_PRECOMPUTED = set()
_H0V3_FRAME_PROPOSALS: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

H0V3_SCORE_FIELDS = [
    "sequence", "frame_id", "pid", "track_id", "chain", "raw_chain_error",
    "temporal_acc", "occlusion_score", "candidate_disagreement", "depth_jump",
    "mask_outside", "p_2d", "p_temporal", "p_occlusion", "p_disagreement",
    "p_depth", "p_mask", "proposal_probability",
]
H0V3_SEGMENT_FIELDS = [
    "sequence", "segment_id", "track_id", "chain", "start_frame", "end_frame",
    "center_frame", "length", "peak_probability", "mean_probability",
    "selected", "rejection", "covered_instances",
]
H0V3_REJECT_FIELDS = [
    "sequence", "segment_id", "track_id", "chain", "variant", "reason",
    "start_frame", "end_frame", "center_frame", "peak_probability",
    "mean_reprojection_ratio", "p90_reprojection_ratio", "max_body_delta",
    "boundary_delta", "constraint_target_mm", "constraint_actual_mm",
    "world_acceleration_ratio",
]

H0V3_VARIANTS = {
    "segment_fit": {
        "trust": 0.28, "steps": 16, "lr": 0.035,
        "w_reproj": 1.00, "w_direction": 0.12, "w_anchor": 0.11,
        "w_pose_acc": 0.22, "w_world_acc": 0.26, "w_boundary": 0.34,
        "w_motion": 0.00, "w_constraint": 0.00,
    },
    "segment_motion": {
        "trust": 0.30, "steps": 18, "lr": 0.033,
        "w_reproj": 0.88, "w_direction": 0.12, "w_anchor": 0.08,
        "w_pose_acc": 0.48, "w_world_acc": 0.64, "w_boundary": 0.42,
        "w_motion": 0.34, "w_constraint": 0.00,
    },
    "segment_depth_near": {
        "trust": 0.34, "steps": 18, "lr": 0.031,
        "w_reproj": 1.00, "w_direction": 0.15, "w_anchor": 0.07,
        "w_pose_acc": 0.28, "w_world_acc": 0.32, "w_boundary": 0.40,
        "w_motion": 0.08, "w_constraint": 0.62, "sign": -1.0,
    },
    "segment_depth_far": {
        "trust": 0.34, "steps": 18, "lr": 0.031,
        "w_reproj": 1.00, "w_direction": 0.15, "w_anchor": 0.07,
        "w_pose_acc": 0.28, "w_world_acc": 0.32, "w_boundary": 0.40,
        "w_motion": 0.08, "w_constraint": 0.62, "sign": 1.0,
    },
    "segment_bend_positive": {
        "trust": 0.32, "steps": 18, "lr": 0.031,
        "w_reproj": 1.00, "w_direction": 0.16, "w_anchor": 0.08,
        "w_pose_acc": 0.34, "w_world_acc": 0.38, "w_boundary": 0.42,
        "w_motion": 0.10, "w_constraint": 0.58, "sign": 1.0,
    },
    "segment_bend_negative": {
        "trust": 0.32, "steps": 18, "lr": 0.031,
        "w_reproj": 1.00, "w_direction": 0.16, "w_anchor": 0.08,
        "w_pose_acc": 0.34, "w_world_acc": 0.38, "w_boundary": 0.42,
        "w_motion": 0.10, "w_constraint": 0.58, "sign": -1.0,
    },
}


def _h0v3_frame_K(pred, frame_id, fallback):
    values = pred.get("K")
    if values is not None and frame_id < len(values):
        value = values[frame_id]
        if torch.is_tensor(value):
            return value.to(device=fallback.device, dtype=fallback.dtype)
        return torch.as_tensor(value, device=fallback.device, dtype=fallback.dtype)
    return fallback


def _h0v3_basis(length, device, dtype):
    if length <= 1:
        return torch.ones((1, 3), device=device, dtype=dtype), torch.ones(1, device=device, dtype=dtype)
    u = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    basis = torch.stack([(1.0-u).square(), 2.0*u*(1.0-u), u.square()], dim=1)
    taper = 0.25 + 0.75 * torch.sin(torch.pi * (torch.arange(length, device=device, dtype=dtype)+1.0)/(length+1.0))
    return basis, taper


def _h0v3_delta_from_coeff(coeff, basis, taper):
    return torch.einsum("tm,mjc->tjc", basis, coeff) * taper[:, None, None]


def _h0v3_fit_coeff(target_delta, basis, taper):
    A = basis * taper[:, None]
    flat = target_delta.reshape(target_delta.shape[0], -1)
    try:
        coeff = torch.linalg.lstsq(A, flat).solution
    except RuntimeError:
        coeff = torch.linalg.pinv(A) @ flat
    return coeff.reshape(basis.shape[1], target_delta.shape[1], 3)


def _h0v3_project_coeff(coeff, basis, taper, trust):
    with torch.no_grad():
        delta = _h0v3_delta_from_coeff(coeff, basis, taper)
        max_norm = torch.linalg.norm(delta, dim=-1).max().clamp_min(1e-8)
        if float(max_norm.detach().cpu()) > float(trust):
            coeff.mul_(float(trust) / max_norm)


def _rms_value(a):
    arr = np.asarray(a)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _h0v3_scan_sequence(sequence, pred, smpl_layer, smpl_model, K_fallback, height, width):
    state = _SEQUENCE_STATES[sequence]
    raw_list = state["raw_rotvec"]
    observations = []
    total_instances = set()
    _H0_TRACKS.pop(sequence, None)
    _H0_NEXT_TRACK.pop(sequence, None)
    for key in list(_H0V2_FEATURE_HISTORY):
        if key[0] == sequence:
            _H0V2_FEATURE_HISTORY.pop(key, None)
    for frame_id, raw_pose in enumerate(raw_list):
        Kf = _h0v3_frame_K(pred, frame_id, K_fallback)
        raw_vertices, raw_joints, raw_j2d = _candidate_geometry(raw_pose, pred, frame_id, smpl_layer, smpl_model, Kf)
        if raw_joints is None:
            continue
        raw_j2d_np = _safe_numpy(raw_j2d)
        assignments, boxes = _h0_track_assignments(sequence, frame_id, raw_j2d_np, width, height)
        raw_world = _h0_torch_transform_points(pred.get("T_c2w", [None])[frame_id], raw_joints)
        detections = _detections(sequence, frame_id, height, width)
        matches = _match_external(raw_j2d_np, detections, height, width)
        masks = _sam3_instance_masks(sequence, frame_id, height, width)
        person_masks = _match_raw_people_to_masks(raw_j2d, masks, height, width)
        if frame_id in state.get("a2_high_ids", set()):
            frame_class = "high"
        elif frame_id in state.get("a2_bad_ids", set()):
            frame_class = "bad"
        elif frame_id in state.get("a2_neighbor_ids", set()):
            frame_class = "neighbor"
        else:
            frame_class = "normal"
        class_boost = {"high": 0.18, "bad": 0.12, "neighbor": 0.05, "normal": 0.0}[frame_class]
        a2 = state["a2_rotvec"][frame_id]
        a2lite = state["a2lite_rotvec"][frame_id]
        bbox_pose, _ = _external_frame_pose("bbox", sequence, frame_id, raw_pose, raw_j2d)
        cand = {"a2": a2, "a2lite": a2lite, "bbox": bbox_pose}
        raw_np = _safe_numpy(raw_pose)
        joints_np = _safe_numpy(raw_joints)
        for match in matches:
            pid = int(match["pid"])
            if pid >= len(assignments) or assignments[pid] is None:
                continue
            tid = int(assignments[pid]); total_instances.add((frame_id, tid))
            history = _h0_history_for_track(sequence, tid)
            det = detections[match["did"]]
            diag = max(float(np.hypot(det["bbox"][2]-det["bbox"][0], det["bbox"][3]-det["bbox"][1])), 1.0)
            for name, chain in CHAINS.items():
                conf = np.asarray(det["kp"][chain["coco"], 2], dtype=np.float32)
                if int(np.sum(conf >= 0.20)) < 2 or float(np.mean(conf)) < 0.30:
                    continue
                target = np.asarray(det["kp"][chain["coco"], :2], dtype=np.float32)
                points = raw_j2d_np[pid, chain["j14"]]
                error = _chain_error(points, target, conf, diag)
                if not np.isfinite(error) or error > 0.50:
                    continue
                temporal_acc = 0.0
                if history["depth"] >= 2:
                    last = history["poses"][-1][chain["pose"]]
                    prev = history["poses"][-2][chain["pose"]]
                    temporal_acc = _rms_value(raw_np[pid, chain["pose"]] - 2.0*last + prev)
                occlusion = float(np.clip(1.0 - np.mean(conf), 0.0, 1.0))
                disagreement = _h0v2_pose_disagreement(pid, chain, raw_pose, cand)
                depth_jump = _h0v2_depth_jump(history, joints_np[pid], chain["j14"])
                mask_outside = _h0v2_mask_outside_score(pid, chain, raw_j2d_np, person_masks, height, width)
                p2d = _h0v2_calibrated_score(sequence, name, "error", error)
                ptemp = _h0v2_calibrated_score(sequence, name, "temporal", temporal_acc)
                pocc = _h0v2_calibrated_score(sequence, name, "occlusion", occlusion)
                pdis = _h0v2_calibrated_score(sequence, name, "disagreement", disagreement)
                pdepth = _h0v2_calibrated_score(sequence, name, "depth", depth_jump)
                pmask = _h0v2_calibrated_score(sequence, name, "mask", mask_outside)
                evidence = [0.90*p2d, 0.72*ptemp, 0.55*pocc, 0.78*pdis, 0.72*pdepth, 0.50*pmask]
                probability = 1.0 - float(np.prod([1.0-np.clip(x,0.0,0.95) for x in evidence]))
                probability = float(np.clip(probability+class_boost,0.0,1.0))
                reliability = float(det["score"]) * math.sqrt(max(float(match["iou"]),1e-6))
                probability *= 0.55 + 0.45*float(np.clip(reliability,0.0,1.0))
                observations.append({
                    "sequence":sequence,"frame_id":frame_id,"pid":pid,"track_id":tid,
                    "chain":name,"config":chain,"target":target,"conf":conf,"scale":diag,
                    "raw_chain_error":error,"temporal_acc":temporal_acc,"occlusion_score":occlusion,
                    "candidate_disagreement":disagreement,"depth_jump":depth_jump,"mask_outside":mask_outside,
                    "p_2d":p2d,"p_temporal":ptemp,"p_occlusion":pocc,"p_disagreement":pdis,
                    "p_depth":pdepth,"p_mask":pmask,"proposal_probability":probability,
                })
        _h0_update_track_history(sequence, frame_id, assignments, boxes, raw_pose, raw_joints, raw_world)
        if frame_id % 100 == 0 or frame_id+1 == len(raw_list):
            print("[H0V3_SCAN]", sequence, f"{frame_id+1}/{len(raw_list)}", "observations=", len(observations), flush=True)
    return observations, total_instances


def _h0v3_extract_segments(sequence, observations, total_instances):
    start_thr = float(os.environ.get("WAO_HMR_H0V3_START_THRESHOLD", "0.78"))
    continue_thr = float(os.environ.get("WAO_HMR_H0V3_CONTINUE_THRESHOLD", "0.55"))
    radius = int(os.environ.get("WAO_HMR_H0V3_RADIUS", "3"))
    min_len = int(os.environ.get("WAO_HMR_H0V3_MIN_LENGTH", "3"))
    max_len = int(os.environ.get("WAO_HMR_H0V3_MAX_LENGTH", "7"))
    groups = defaultdict(list)
    for obs in observations:
        groups[(int(obs["track_id"]), obs["chain"])].append(obs)
    candidates=[]; segment_id=0
    for (tid, chain), values in groups.items():
        values=sorted(values,key=lambda x:x["frame_id"]); by_frame={int(x["frame_id"]):x for x in values}
        for i, peak in enumerate(values):
            p=float(peak["proposal_probability"])
            if p < start_thr: continue
            prev=float(values[i-1]["proposal_probability"]) if i>0 and values[i-1]["frame_id"]==peak["frame_id"]-1 else -1
            nxt=float(values[i+1]["proposal_probability"]) if i+1<len(values) and values[i+1]["frame_id"]==peak["frame_id"]+1 else -1
            if p < max(prev,nxt): continue
            frames=[]
            for f in range(max(0,peak["frame_id"]-radius), peak["frame_id"]+radius+1):
                item=by_frame.get(f)
                if item is not None and (float(item["proposal_probability"])>=continue_thr or f==peak["frame_id"]):
                    frames.append(item)
            frames=sorted(frames,key=lambda x:x["frame_id"])
            # Keep a contiguous run around the peak.
            run=[peak]
            for direction in (-1,1):
                f=peak["frame_id"]+direction
                while len(run)<max_len and abs(f-peak["frame_id"])<=radius:
                    item=by_frame.get(f)
                    if item is None or float(item["proposal_probability"])<continue_thr: break
                    run.append(item); f+=direction
            frames=sorted({x["frame_id"]:x for x in run}.values(), key=lambda x:x["frame_id"])
            if len(frames)<min_len: continue
            if len(frames)>max_len:
                frames=sorted(frames,key=lambda x:abs(x["frame_id"]-peak["frame_id"]))[:max_len]
                frames=sorted(frames,key=lambda x:x["frame_id"])
            candidates.append({"segment_id":segment_id,"track_id":tid,"chain":chain,"frames":frames,
                "center_frame":int(peak["frame_id"]),"peak_probability":p,
                "mean_probability":float(np.mean([x["proposal_probability"] for x in frames]))})
            segment_id+=1
    candidates.sort(key=lambda x:(x["peak_probability"]+0.15*x["mean_probability"]), reverse=True)
    target_cov=float(os.environ.get("WAO_HMR_H0V3_TARGET_COVERAGE", "0.15"))
    cap=max(1,int(math.ceil(len(total_instances)*target_cov)))
    max_segments=int(os.environ.get("WAO_HMR_H0V3_MAX_SEGMENTS", "100000"))
    selected=[]; covered=set(); frame_track_count=Counter(); rows=[]
    for seg in candidates:
        keys={(int(x["frame_id"]), int(seg["track_id"])) for x in seg["frames"]}
        overlap=False
        for old in selected:
            if old["track_id"]==seg["track_id"] and old["chain"]==seg["chain"]:
                old_frames={x["frame_id"] for x in old["frames"]}; new_frames={x["frame_id"] for x in seg["frames"]}
                if len(old_frames & new_frames) >= max(1,min(len(old_frames),len(new_frames))//2): overlap=True; break
        reason="selected"
        if overlap: reason="segment_nms"
        elif any(frame_track_count[k]>=2 for k in keys): reason="max_chains_per_instance"
        elif len(selected)>=max_segments: reason="segment_budget"
        elif len(covered | keys)>cap and len(covered)>0: reason="coverage_budget"
        if reason=="selected":
            selected.append(seg); covered |= keys
            for k in keys: frame_track_count[k]+=1
        rows.append({"sequence":sequence,"segment_id":seg["segment_id"],"track_id":seg["track_id"],"chain":seg["chain"],
            "start_frame":seg["frames"][0]["frame_id"],"end_frame":seg["frames"][-1]["frame_id"],
            "center_frame":seg["center_frame"],"length":len(seg["frames"]),"peak_probability":seg["peak_probability"],
            "mean_probability":seg["mean_probability"],"selected":int(reason=="selected"),"rejection":reason,
            "covered_instances":len(keys)})
    return selected, rows


def _h0v3_segment_loss(coeff, segment, variant, pred, smpl_layer, smpl_model, K_fallback, constraint_data):
    cfg=H0V3_VARIANTS[variant]; frames=segment["frames"]
    raw_list=_SEQUENCE_STATES[segment["sequence"]]["raw_rotvec"]
    pose_ids=list(CHAINS[segment["chain"]]["pose"]); joint_ids=list(CHAINS[segment["chain"]]["j14"])
    basis,taper=_h0v3_basis(len(frames),coeff.device,coeff.dtype); deltas=_h0v3_delta_from_coeff(coeff,basis,taper)
    reproj=[]; direction=[]; anchor=[]; local_seq=[]; raw_local=[]; world_seq=[]; raw_world_seq=[]; constraint=[]
    for t,obs in enumerate(frames):
        f=int(obs["frame_id"]); pid=int(obs["pid"]); raw_pose=raw_list[f]
        Kf=_h0v3_frame_K(pred,f,K_fallback)
        _,_,joints,j2d,world=_h0_person_geometry(raw_pose,pid,pose_ids,deltas[t],pred,f,smpl_layer,smpl_model,Kf)
        target=torch.as_tensor(obs["target"],dtype=coeff.dtype,device=coeff.device)
        weights=torch.as_tensor(obs["conf"],dtype=coeff.dtype,device=coeff.device).clamp_min(0.05); weights/=weights.sum().clamp_min(1e-6)
        dist=torch.linalg.norm((j2d[0,joint_ids]-target)/max(float(obs["scale"]),1.0),dim=-1)
        reproj.append(torch.sum(weights*F.smooth_l1_loss(dist,torch.zeros_like(dist),reduction="none",beta=0.02)))
        direction.append(_h0_direction_loss(j2d[0,joint_ids],target))
        anchor.append(torch.mean((deltas[t]/max(float(cfg["trust"]),1e-4))**2))
        local_seq.append(raw_pose[pid,pose_ids]+deltas[t]); raw_local.append(raw_pose[pid,pose_ids])
        if world is not None: world_seq.append(world[0,joint_ids])
        else: world_seq.append(joints[0,joint_ids])
        zero=torch.zeros_like(deltas[t]); _,_,rj,_,rw=_h0_person_geometry(raw_pose,pid,pose_ids,zero,pred,f,smpl_layer,smpl_model,Kf)
        raw_world_seq.append((rw if rw is not None else rj)[0,joint_ids])
        if constraint_data is not None:
            c=constraint_data[t]; achieved=torch.dot(joints[0,joint_ids[c["joint_local"]]]-c["raw_point"],c["axis"])
            constraint.append(((achieved-c["target"])/c["scale"]).square())
    local_seq=torch.stack(local_seq); raw_local=torch.stack(raw_local); world_seq=torch.stack(world_seq); raw_world_seq=torch.stack(raw_world_seq)
    pose_acc=torch.zeros((),dtype=coeff.dtype,device=coeff.device); world_acc=pose_acc.clone(); motion=pose_acc.clone()
    if len(frames)>=3:
        ca=local_seq[2:]-2*local_seq[1:-1]+local_seq[:-2]; ra=raw_local[2:]-2*raw_local[1:-1]+raw_local[:-2]
        pose_acc=F.smooth_l1_loss(ca-ra,torch.zeros_like(ca),beta=0.03)
        cwa=world_seq[2:]-2*world_seq[1:-1]+world_seq[:-2]; rwa=raw_world_seq[2:]-2*raw_world_seq[1:-1]+raw_world_seq[:-2]
        limb=torch.linalg.norm(raw_world_seq[:,-1]-raw_world_seq[:,0],dim=-1).mean().detach().clamp_min(0.15)
        world_acc=F.smooth_l1_loss((cwa-rwa)/limb,torch.zeros_like(cwa),beta=0.03)
        u=torch.linspace(0,1,len(frames),device=coeff.device,dtype=coeff.dtype)[:,None,None]
        linear=(1-u)*raw_local[0:1]+u*raw_local[-1:]
        motion=F.smooth_l1_loss(local_seq-linear,torch.zeros_like(local_seq),beta=0.05)
    boundary=(deltas[0].square().mean()+deltas[-1].square().mean())
    if len(frames)>=2:
        boundary=boundary+F.smooth_l1_loss((world_seq[1]-world_seq[0])-(raw_world_seq[1]-raw_world_seq[0]),torch.zeros_like(world_seq[0]),beta=0.02)
        boundary=boundary+F.smooth_l1_loss((world_seq[-1]-world_seq[-2])-(raw_world_seq[-1]-raw_world_seq[-2]),torch.zeros_like(world_seq[0]),beta=0.02)
    constraint_loss=torch.stack(constraint).mean() if constraint else torch.zeros_like(boundary)
    total=(cfg["w_reproj"]*torch.stack(reproj).mean()+cfg["w_direction"]*torch.stack(direction).mean()+cfg["w_anchor"]*torch.stack(anchor).mean()+
        cfg["w_pose_acc"]*pose_acc+cfg["w_world_acc"]*world_acc+cfg["w_boundary"]*boundary+cfg["w_motion"]*motion+cfg["w_constraint"]*constraint_loss)
    return total,{"reprojection":torch.stack(reproj).mean(),"direction":torch.stack(direction).mean(),"pose_acc":pose_acc,
        "world_acc":world_acc,"boundary":boundary,"motion":motion,"constraint":constraint_loss,"anchor":torch.stack(anchor).mean()},deltas,world_seq,raw_world_seq


def _h0v3_constraint_data(segment, variant, pred, smpl_layer, smpl_model, K_fallback):
    if not ("depth" in variant or "bend" in variant): return None,None
    sign=float(H0V3_VARIANTS[variant]["sign"]); chain=CHAINS[segment["chain"]]; pose_ids=list(chain["pose"]); joint_ids=list(chain["j14"])
    raw_list=_SEQUENCE_STATES[segment["sequence"]]["raw_rotvec"]; direction_targets=[]; data=[]
    for obs in segment["frames"]:
        f=int(obs["frame_id"]); pid=int(obs["pid"]); raw_pose=raw_list[f]; Kf=_h0v3_frame_K(pred,f,K_fallback)
        if chain["kind"]=="arm":
            direction=_h0_depth_preserving_direction(raw_pose,pid,pose_ids,joint_ids,pred,f,smpl_layer,smpl_model,Kf,obs["scale"])
            axis=torch.tensor([0.,0.,1.],dtype=raw_pose.dtype,device=raw_pose.device); jlocal=2
        else:
            direction,axis=_h0v2_bend_preserving_direction(raw_pose,pid,pose_ids,joint_ids,pred,f,smpl_layer,smpl_model,Kf,obs["scale"]); jlocal=1
        if direction is None or axis is None: return None,"missing_constraint_direction"
        zero=torch.zeros_like(direction); _,_,rj,_,_=_h0_person_geometry(raw_pose,pid,pose_ids,zero,pred,f,smpl_layer,smpl_model,Kf)
        trust=float(H0V3_VARIANTS[variant]["trust"]); probe=direction*(0.78*trust)*sign
        _,_,pj,_,_=_h0_person_geometry(raw_pose,pid,pose_ids,probe,pred,f,smpl_layer,smpl_model,Kf)
        achievable=float(torch.dot(pj[0,joint_ids[jlocal]]-rj[0,joint_ids[jlocal]],axis).detach().cpu())
        min_reachable = 0.012 if chain["kind"]=="arm" else 0.008
        max_target = 0.080 if chain["kind"]=="arm" else 0.055
        if not np.isfinite(achievable) or achievable * sign <= 0.0 or abs(achievable) < min_reachable:
            return None,"constraint_unreachable"
        target=float(np.sign(achievable) * min(0.70*abs(achievable), max_target))
        limb=torch.linalg.norm(rj[0,joint_ids[-1]]-rj[0,joint_ids[0]]).detach().clamp_min(0.15)
        data.append({"joint_local":jlocal,"axis":axis.detach(),"raw_point":rj[0,joint_ids[jlocal]].detach(),
            "target":torch.tensor(target,dtype=raw_pose.dtype,device=raw_pose.device),"scale":limb})
        direction_targets.append(direction*min(0.14,0.55*trust)*sign)
    return (data,torch.stack(direction_targets)),None


def _h0v3_optimize_segment(sequence, segment, variant, pred, smpl_layer, smpl_model, K_fallback):
    cfg=H0V3_VARIANTS[variant]; segment=dict(segment); segment["sequence"]=sequence
    frames=segment["frames"]; chain=CHAINS[segment["chain"]]; pose_ids=list(chain["pose"])
    raw_list=_SEQUENCE_STATES[sequence]["raw_rotvec"]; device=raw_list[frames[0]["frame_id"]].device; dtype=raw_list[frames[0]["frame_id"]].dtype
    basis,taper=_h0v3_basis(len(frames),device,dtype); target_delta=torch.zeros((len(frames),len(pose_ids),3),device=device,dtype=dtype)
    constraint_data=None
    if variant=="segment_motion":
        raw_local=torch.stack([raw_list[x["frame_id"]][x["pid"],pose_ids] for x in frames])
        u=torch.linspace(0,1,len(frames),device=device,dtype=dtype)[:,None,None]
        target_delta=(1-u)*raw_local[0:1]+u*raw_local[-1:]-raw_local
    elif "depth" in variant or "bend" in variant:
        packed,reason=_h0v3_constraint_data(segment,variant,pred,smpl_layer,smpl_model,K_fallback)
        if packed is None: return None,reason
        constraint_data,target_delta=packed
    coeff=_h0v3_fit_coeff(target_delta,basis,taper).detach().requires_grad_(True)
    opt=torch.optim.Adam([coeff],lr=float(cfg["lr"])); _h0v3_project_coeff(coeff,basis,taper,cfg["trust"])
    for _ in range(int(cfg["steps"])):
        opt.zero_grad(set_to_none=True); total,parts,_,_,_=_h0v3_segment_loss(coeff,segment,variant,pred,smpl_layer,smpl_model,K_fallback,constraint_data)
        if not torch.isfinite(total): return None,"nonfinite_loss"
        total.backward(); torch.nn.utils.clip_grad_norm_([coeff],2.0); opt.step(); _h0v3_project_coeff(coeff,basis,taper,cfg["trust"])
    with torch.no_grad():
        total,parts,deltas,world_seq,raw_world_seq=_h0v3_segment_loss(coeff,segment,variant,pred,smpl_layer,smpl_model,K_fallback,constraint_data)
    frame_results=[]; ratios=[]; max_body=0.; constraint_actual=[]; constraint_target=[]
    for t,obs in enumerate(frames):
        f=int(obs["frame_id"]); pid=int(obs["pid"]); raw_pose=raw_list[f]; Kf=_h0v3_frame_K(pred,f,K_fallback)
        candidate=raw_pose.detach().clone(); candidate[pid,pose_ids]+=deltas[t].detach()
        _,_,cj,c2d,_=_h0_person_geometry(raw_pose,pid,pose_ids,deltas[t],pred,f,smpl_layer,smpl_model,Kf)
        zero=torch.zeros_like(deltas[t]); _,_,rj,r2d,_=_h0_person_geometry(raw_pose,pid,pose_ids,zero,pred,f,smpl_layer,smpl_model,Kf)
        raw_error=_chain_error(_safe_numpy(r2d[0,chain["j14"]]),obs["target"],obs["conf"],obs["scale"])
        final_error=_chain_error(_safe_numpy(c2d[0,chain["j14"]]),obs["target"],obs["conf"],obs["scale"])
        ratio=final_error/max(raw_error,1e-6); ratios.append(ratio)
        body_rms,body_max=_person_body_delta(candidate,raw_pose,pid); max_body=max(max_body,body_max)
        ctarget=0.; cactual=0.; ckind="none"
        if constraint_data is not None:
            c=constraint_data[t]; cactual=float(torch.dot(cj[0,chain["j14"][c["joint_local"]]]-c["raw_point"],c["axis"]).cpu())*1000.; ctarget=float(c["target"].cpu())*1000.
            constraint_actual.append(cactual); constraint_target.append(ctarget); ckind="segment_arm_depth" if chain["kind"]=="arm" else "segment_leg_bend"
        frame_results.append({"pose":candidate,"pid":pid,"chain":segment["chain"],"variant":variant,"track_id":segment["track_id"],
            "proposal_rank":segment["segment_id"],"segment_id":segment["segment_id"],"segment_start":frames[0]["frame_id"],"segment_end":frames[-1]["frame_id"],
            "segment_center":segment["center_frame"],"segment_length":len(frames),"segment_offset":f-segment["center_frame"],
            "segment_peak_probability":segment["peak_probability"],"temporal_basis_dim":3,"boundary_taper":float(taper[t].cpu()),
            "history_depth":max(0,t),"risk_score":segment["peak_probability"],"proposal_probability":segment["peak_probability"],
            "raw_chain_error":raw_error,"final_chain_error":final_error,"reprojection_improvement":(raw_error-final_error)/max(raw_error,1e-6),
            "raw_direction_error":_direction_error(_safe_numpy(r2d[0,chain["j14"]]),obs["target"]),
            "final_direction_error":_direction_error(_safe_numpy(c2d[0,chain["j14"]]),obs["target"]),
            "trust_radius_rad":cfg["trust"],"optimizer_steps":cfg["steps"],"depth_target_mm":ctarget if chain["kind"]=="arm" else 0.,
            "depth_actual_mm":cactual if chain["kind"]=="arm" else 0.,"constraint_kind":ckind,"constraint_target_mm":ctarget,"constraint_actual_mm":cactual,
            "body_rms":body_rms,"body_max":body_max,"final_total_loss":float(total.cpu()),"final_reprojection_loss":float(parts["reprojection"].cpu()),
            "final_direction_loss":float(parts["direction"].cpu()),"final_temporal_pose_loss":float(parts["pose_acc"].cpu()),
            "final_world_motion_loss":float(parts["world_acc"].cpu()),"final_depth_loss":float(parts["constraint"].cpu()),"final_anchor_loss":float(parts["anchor"].cpu()),
            "p_2d":obs["p_2d"],"p_temporal":obs["p_temporal"],"p_occlusion":obs["p_occlusion"],"p_disagreement":obs["p_disagreement"],"p_depth":obs["p_depth"],"p_mask":obs["p_mask"],
        })
    ratios=np.asarray(ratios,float); boundary=max(frame_results[0]["body_max"],frame_results[-1]["body_max"])
    raw_acc=torch.linalg.norm(raw_world_seq[2:]-2*raw_world_seq[1:-1]+raw_world_seq[:-2],dim=-1).mean() if len(frames)>=3 else torch.tensor(0.,device=device)
    cand_acc=torch.linalg.norm(world_seq[2:]-2*world_seq[1:-1]+world_seq[:-2],dim=-1).mean() if len(frames)>=3 else torch.tensor(0.,device=device)
    acc_ratio=float((cand_acc/(raw_acc+1e-4)).cpu())
    reason="accepted"
    if not np.isfinite(ratios).all(): reason="nonfinite_geometry"
    elif max_body>cfg["trust"]+1e-4: reason="trust_region_violation"
    elif float(np.mean(ratios))>1.18 or float(np.quantile(ratios,0.9))>1.35: reason="catastrophic_reprojection"
    elif boundary>0.16: reason="boundary_discontinuity"
    elif acc_ratio>3.0 and float(cand_acc.cpu())>0.08: reason="catastrophic_world_acceleration"
    elif constraint_data is not None:
        med_actual=float(np.median(constraint_actual)); med_target=float(np.median(constraint_target))
        if abs(med_actual)<0.30*abs(med_target): reason="constraint_too_small"
        elif med_actual*med_target<=0: reason="constraint_wrong_sign"
    if reason!="accepted":
        return {"reason":reason,"mean_reprojection_ratio":float(np.mean(ratios)),"p90_reprojection_ratio":float(np.quantile(ratios,0.9)),
            "max_body_delta":max_body,"boundary_delta":boundary,"constraint_target_mm":float(np.median(constraint_target)) if constraint_target else 0.,
            "constraint_actual_mm":float(np.median(constraint_actual)) if constraint_actual else 0.,"world_acceleration_ratio":acc_ratio},reason
    return frame_results,"accepted"



# ---------------------------------------------------------------------------
# H0-v3-fast: batched segment optimization for six-sequence selector trials.
# The original H0-v3 implementation is kept intact for reproducibility.
# ---------------------------------------------------------------------------

def _h0v3_fast_batch_geometry(segment, deltas, pred, smpl_layer, smpl_model, K_fallback):
    sequence = segment["sequence"]
    raw_list = _SEQUENCE_STATES[sequence]["raw_rotvec"]
    pose_ids = list(CHAINS[segment["chain"]]["pose"])
    person_poses = []
    shapes = []
    transls = []
    expressions = []
    cameras = []
    transforms = []
    for t, obs in enumerate(segment["frames"]):
        frame_id = int(obs["frame_id"])
        pid = int(obs["pid"])
        raw_pose = raw_list[frame_id]
        person_pose = raw_pose[pid].clone()
        person_pose[pose_ids] = person_pose[pose_ids] + deltas[t]
        person_poses.append(person_pose)
        shapes.append(pred["shape"][frame_id][pid])
        transls.append(pred["transl"][frame_id][pid])
        expressions.append(pred["expression"][frame_id][pid])
        Kf = _h0v3_frame_K(pred, frame_id, K_fallback)
        cameras.append(Kf if Kf.ndim == 2 else Kf[pid])
        transforms.append(pred.get("T_c2w", [None])[frame_id])
    person_poses = torch.stack(person_poses, dim=0)
    shapes = torch.stack(shapes, dim=0)
    transls = torch.stack(transls, dim=0)
    expressions = torch.stack(expressions, dim=0)
    cameras = torch.stack(cameras, dim=0)
    out = smpl_layer(
        person_poses, shapes, transls, None, None,
        K=cameras, expression=expressions,
    )
    vx = out["smpl_v3d"]
    vertices = smpl_model.smplx2smpl.to(vx) @ vx
    joints = smpl_model.j_regressor.to(vertices) @ vertices
    j2d = torch.einsum(
        "bij,bkj->bki", cameras,
        joints / joints[..., 2:3].clamp_min(1e-4),
    )[..., :2]
    worlds = []
    for t, transform in enumerate(transforms):
        world = _h0_torch_transform_points(transform, joints[t:t + 1])
        worlds.append((world if world is not None else joints[t:t + 1])[0])
    return person_poses, vertices, joints, j2d, torch.stack(worlds, dim=0)


def _h0v3_fast_prepare_segment(sequence, segment, pred, smpl_layer, smpl_model, K_fallback):
    prepared = dict(segment)
    prepared["sequence"] = sequence
    frames = prepared["frames"]
    chain = CHAINS[prepared["chain"]]
    pose_ids = list(chain["pose"])
    joint_ids = list(chain["j14"])
    raw_list = _SEQUENCE_STATES[sequence]["raw_rotvec"]
    device = raw_list[int(frames[0]["frame_id"])].device
    dtype = raw_list[int(frames[0]["frame_id"])].dtype
    zeros = torch.zeros((len(frames), len(pose_ids), 3), device=device, dtype=dtype)
    with torch.no_grad():
        _, raw_vertices, raw_joints, raw_j2d, raw_world = _h0v3_fast_batch_geometry(
            prepared, zeros, pred, smpl_layer, smpl_model, K_fallback
        )
    targets = torch.stack([
        torch.as_tensor(obs["target"], device=device, dtype=dtype)
        for obs in frames
    ])
    weights = torch.stack([
        torch.as_tensor(obs["conf"], device=device, dtype=dtype).clamp_min(0.05)
        for obs in frames
    ])
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    scales = torch.as_tensor(
        [max(float(obs["scale"]), 1.0) for obs in frames],
        device=device, dtype=dtype,
    )
    raw_local = torch.stack([
        raw_list[int(obs["frame_id"])][int(obs["pid"]), pose_ids]
        for obs in frames
    ])
    prepared.update({
        "pose_ids": pose_ids,
        "joint_ids": joint_ids,
        "device": device,
        "dtype": dtype,
        "targets": targets,
        "weights": weights,
        "scales": scales,
        "raw_local": raw_local.detach(),
        "raw_vertices": raw_vertices.detach(),
        "raw_joints": raw_joints.detach(),
        "raw_j2d": raw_j2d.detach(),
        "raw_world": raw_world.detach(),
        "constraint_base": None,
    })
    return prepared


def _h0v3_fast_direction_loss(points, targets):
    losses = []
    for a, b in ((0, 1), (1, 2)):
        u = points[:, b] - points[:, a]
        v = targets[:, b] - targets[:, a]
        denom = torch.linalg.norm(u, dim=-1).clamp_min(1e-6) * torch.linalg.norm(v, dim=-1).clamp_min(1e-6)
        cosine = torch.sum(u * v, dim=-1) / denom
        losses.append(1.0 - torch.clamp(cosine, -1.0, 1.0))
    return torch.stack(losses, dim=1).mean()


def _h0v3_fast_constraint_base(prepared, pred, smpl_layer, smpl_model, K_fallback):
    if prepared.get("constraint_base") is not None:
        return prepared["constraint_base"], None
    chain = CHAINS[prepared["chain"]]
    pose_ids = prepared["pose_ids"]
    joint_ids = prepared["joint_ids"]
    raw_list = _SEQUENCE_STATES[prepared["sequence"]]["raw_rotvec"]
    directions = []
    axes = []
    joint_locals = []
    for obs in prepared["frames"]:
        frame_id = int(obs["frame_id"])
        pid = int(obs["pid"])
        raw_pose = raw_list[frame_id]
        Kf = _h0v3_frame_K(pred, frame_id, K_fallback)
        if chain["kind"] == "arm":
            direction = _h0_depth_preserving_direction(
                raw_pose, pid, pose_ids, joint_ids, pred, frame_id,
                smpl_layer, smpl_model, Kf, obs["scale"],
            )
            axis = torch.tensor([0.0, 0.0, 1.0], device=raw_pose.device, dtype=raw_pose.dtype)
            joint_local = 2
        else:
            direction, axis = _h0v2_bend_preserving_direction(
                raw_pose, pid, pose_ids, joint_ids, pred, frame_id,
                smpl_layer, smpl_model, Kf, obs["scale"],
            )
            joint_local = 1
        if direction is None or axis is None:
            return None, "missing_constraint_direction"
        directions.append(direction.detach())
        axes.append(axis.detach())
        joint_locals.append(int(joint_local))
    base = {
        "directions": torch.stack(directions, dim=0),
        "axes": torch.stack(axes, dim=0),
        "joint_locals": joint_locals,
    }
    prepared["constraint_base"] = base
    return base, None


def _h0v3_fast_constraint_data(prepared, variant, pred, smpl_layer, smpl_model, K_fallback):
    if not ("depth" in variant or "bend" in variant):
        return None, None
    base, reason = _h0v3_fast_constraint_base(prepared, pred, smpl_layer, smpl_model, K_fallback)
    if base is None:
        return None, reason
    cfg = H0V3_VARIANTS[variant]
    sign = float(cfg["sign"])
    chain = CHAINS[prepared["chain"]]
    trust = float(cfg["trust"])
    probes = base["directions"] * (0.78 * trust * sign)
    with torch.no_grad():
        _, _, probe_joints, _, _ = _h0v3_fast_batch_geometry(
            prepared, probes, pred, smpl_layer, smpl_model, K_fallback
        )
    data = []
    direction_targets = []
    for t, obs in enumerate(prepared["frames"]):
        joint_local = base["joint_locals"][t]
        joint_index = prepared["joint_ids"][joint_local]
        axis = base["axes"][t]
        raw_point = prepared["raw_joints"][t, joint_index]
        achievable = float(torch.dot(probe_joints[t, joint_index] - raw_point, axis).detach().cpu())
        min_reachable = 0.012 if chain["kind"] == "arm" else 0.008
        max_target = 0.080 if chain["kind"] == "arm" else 0.055
        if not np.isfinite(achievable) or achievable * sign <= 0.0 or abs(achievable) < min_reachable:
            return None, "constraint_unreachable"
        target = float(np.sign(achievable) * min(0.70 * abs(achievable), max_target))
        limb = torch.linalg.norm(
            prepared["raw_joints"][t, prepared["joint_ids"][-1]] -
            prepared["raw_joints"][t, prepared["joint_ids"][0]]
        ).detach().clamp_min(0.15)
        data.append({
            "joint_local": joint_local,
            "axis": axis,
            "raw_point": raw_point,
            "target": torch.tensor(target, device=prepared["device"], dtype=prepared["dtype"]),
            "scale": limb,
        })
        direction_targets.append(base["directions"][t] * min(0.14, 0.55 * trust) * sign)
    return (data, torch.stack(direction_targets, dim=0)), None


def _h0v3_fast_loss(coeff, prepared, variant, pred, smpl_layer, smpl_model, K_fallback, constraint_data):
    cfg = H0V3_VARIANTS[variant]
    frames = prepared["frames"]
    basis, taper = _h0v3_basis(len(frames), coeff.device, coeff.dtype)
    deltas = _h0v3_delta_from_coeff(coeff, basis, taper)
    _, vertices, joints, j2d, world = _h0v3_fast_batch_geometry(
        prepared, deltas, pred, smpl_layer, smpl_model, K_fallback
    )
    points = j2d[:, prepared["joint_ids"]]
    residual = (points - prepared["targets"]) / prepared["scales"][:, None, None]
    dist = torch.linalg.norm(residual, dim=-1)
    reproj = torch.sum(
        prepared["weights"] * F.smooth_l1_loss(
            dist, torch.zeros_like(dist), reduction="none", beta=0.02
        ), dim=1,
    ).mean()
    direction = _h0v3_fast_direction_loss(points, prepared["targets"])
    anchor = torch.mean((deltas / max(float(cfg["trust"]), 1e-4)) ** 2)
    local_seq = prepared["raw_local"] + deltas
    raw_local = prepared["raw_local"]
    world_seq = world[:, prepared["joint_ids"]]
    raw_world = prepared["raw_world"][:, prepared["joint_ids"]]
    zero = torch.zeros((), device=coeff.device, dtype=coeff.dtype)
    pose_acc = zero
    world_acc = zero
    motion = zero
    if len(frames) >= 3:
        ca = local_seq[2:] - 2 * local_seq[1:-1] + local_seq[:-2]
        ra = raw_local[2:] - 2 * raw_local[1:-1] + raw_local[:-2]
        pose_acc = F.smooth_l1_loss(ca - ra, torch.zeros_like(ca), beta=0.03)
        cwa = world_seq[2:] - 2 * world_seq[1:-1] + world_seq[:-2]
        rwa = raw_world[2:] - 2 * raw_world[1:-1] + raw_world[:-2]
        limb = torch.linalg.norm(raw_world[:, -1] - raw_world[:, 0], dim=-1).mean().detach().clamp_min(0.15)
        world_acc = F.smooth_l1_loss((cwa - rwa) / limb, torch.zeros_like(cwa), beta=0.03)
        u = torch.linspace(0, 1, len(frames), device=coeff.device, dtype=coeff.dtype)[:, None, None]
        linear = (1 - u) * raw_local[0:1] + u * raw_local[-1:]
        motion = F.smooth_l1_loss(local_seq - linear, torch.zeros_like(local_seq), beta=0.05)
    boundary = deltas[0].square().mean() + deltas[-1].square().mean()
    if len(frames) >= 2:
        boundary = boundary + F.smooth_l1_loss(
            (world_seq[1] - world_seq[0]) - (raw_world[1] - raw_world[0]),
            torch.zeros_like(world_seq[0]), beta=0.02,
        )
        boundary = boundary + F.smooth_l1_loss(
            (world_seq[-1] - world_seq[-2]) - (raw_world[-1] - raw_world[-2]),
            torch.zeros_like(world_seq[0]), beta=0.02,
        )
    constraint_values = []
    if constraint_data is not None:
        for t, item in enumerate(constraint_data):
            joint_index = prepared["joint_ids"][item["joint_local"]]
            achieved = torch.dot(joints[t, joint_index] - item["raw_point"], item["axis"])
            constraint_values.append(((achieved - item["target"]) / item["scale"]).square())
    constraint = torch.stack(constraint_values).mean() if constraint_values else zero
    total = (
        cfg["w_reproj"] * reproj + cfg["w_direction"] * direction +
        cfg["w_anchor"] * anchor + cfg["w_pose_acc"] * pose_acc +
        cfg["w_world_acc"] * world_acc + cfg["w_boundary"] * boundary +
        cfg["w_motion"] * motion + cfg["w_constraint"] * constraint
    )
    parts = {
        "reprojection": reproj, "direction": direction, "pose_acc": pose_acc,
        "world_acc": world_acc, "boundary": boundary, "motion": motion,
        "constraint": constraint, "anchor": anchor,
    }
    return total, parts, deltas, vertices, joints, j2d, world_seq, raw_world


def _h0v3_fast_steps(variant):
    if "depth" in variant or "bend" in variant:
        default = 8
        env = "WAO_HMR_H0V3_FAST_STEPS_CONSTRAINT"
    elif variant == "segment_motion":
        default = 7
        env = "WAO_HMR_H0V3_FAST_STEPS_MOTION"
    else:
        default = 6
        env = "WAO_HMR_H0V3_FAST_STEPS_FIT"
    return max(2, int(os.environ.get(env, str(default))))


def _h0v3_fast_optimize_segment(prepared, variant, pred, smpl_layer, smpl_model, K_fallback):
    cfg = H0V3_VARIANTS[variant]
    frames = prepared["frames"]
    chain = CHAINS[prepared["chain"]]
    basis, taper = _h0v3_basis(len(frames), prepared["device"], prepared["dtype"])
    target_delta = torch.zeros(
        (len(frames), len(prepared["pose_ids"]), 3),
        device=prepared["device"], dtype=prepared["dtype"],
    )
    constraint_data = None
    if variant == "segment_motion":
        raw_local = prepared["raw_local"]
        u = torch.linspace(0, 1, len(frames), device=prepared["device"], dtype=prepared["dtype"])[:, None, None]
        target_delta = (1 - u) * raw_local[0:1] + u * raw_local[-1:] - raw_local
    elif "depth" in variant or "bend" in variant:
        packed, reason = _h0v3_fast_constraint_data(
            prepared, variant, pred, smpl_layer, smpl_model, K_fallback
        )
        if packed is None:
            return None, reason
        constraint_data, target_delta = packed
    coeff = _h0v3_fit_coeff(target_delta, basis, taper).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([coeff], lr=float(cfg["lr"]))
    _h0v3_project_coeff(coeff, basis, taper, cfg["trust"])
    steps = _h0v3_fast_steps(variant)
    patience = max(1, int(os.environ.get("WAO_HMR_H0V3_FAST_PATIENCE", "2")))
    min_steps = max(3, int(os.environ.get("WAO_HMR_H0V3_FAST_MIN_STEPS", "4")))
    best = float("inf")
    stale = 0
    actual_steps = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total, _, _, _, _, _, _, _ = _h0v3_fast_loss(
            coeff, prepared, variant, pred, smpl_layer, smpl_model, K_fallback, constraint_data
        )
        if not torch.isfinite(total):
            return None, "nonfinite_loss"
        total.backward()
        torch.nn.utils.clip_grad_norm_([coeff], 2.0)
        optimizer.step()
        _h0v3_project_coeff(coeff, basis, taper, cfg["trust"])
        actual_steps = step + 1
        value = float(total.detach().cpu())
        if (not np.isfinite(best)) or value < best - max(1e-5, 2e-3 * max(abs(best), 1.0)):
            best = value
            stale = 0
        else:
            stale += 1
        if actual_steps >= min_steps and stale >= patience:
            break
    with torch.no_grad():
        total, parts, deltas, _, joints, j2d, world_seq, raw_world = _h0v3_fast_loss(
            coeff, prepared, variant, pred, smpl_layer, smpl_model, K_fallback, constraint_data
        )
    ratios = []
    frame_results = []
    constraint_actual = []
    constraint_target = []
    max_body = 0.0
    raw_list = _SEQUENCE_STATES[prepared["sequence"]]["raw_rotvec"]
    for t, obs in enumerate(frames):
        frame_id = int(obs["frame_id"])
        pid = int(obs["pid"])
        raw_pose = raw_list[frame_id]
        candidate = raw_pose.detach().clone()
        candidate[pid, prepared["pose_ids"]] += deltas[t].detach()
        raw_points = prepared["raw_j2d"][t, prepared["joint_ids"]]
        cand_points = j2d[t, prepared["joint_ids"]]
        raw_error = _chain_error(_safe_numpy(raw_points), obs["target"], obs["conf"], obs["scale"])
        final_error = _chain_error(_safe_numpy(cand_points), obs["target"], obs["conf"], obs["scale"])
        ratio = final_error / max(raw_error, 1e-6)
        ratios.append(ratio)
        body_rms, body_max = _person_body_delta(candidate, raw_pose, pid)
        max_body = max(max_body, body_max)
        ctarget = 0.0
        cactual = 0.0
        ckind = "none"
        if constraint_data is not None:
            item = constraint_data[t]
            joint_index = prepared["joint_ids"][item["joint_local"]]
            cactual = float(torch.dot(joints[t, joint_index] - item["raw_point"], item["axis"]).cpu()) * 1000.0
            ctarget = float(item["target"].cpu()) * 1000.0
            constraint_actual.append(cactual)
            constraint_target.append(ctarget)
            ckind = "segment_arm_depth" if chain["kind"] == "arm" else "segment_leg_bend"
        frame_results.append({
            "pose": candidate, "pid": pid, "chain": prepared["chain"], "variant": variant,
            "track_id": prepared["track_id"], "proposal_rank": prepared["segment_id"],
            "segment_id": prepared["segment_id"],
            "segment_start": frames[0]["frame_id"], "segment_end": frames[-1]["frame_id"],
            "segment_center": prepared["center_frame"], "segment_length": len(frames),
            "segment_offset": frame_id - prepared["center_frame"],
            "segment_peak_probability": prepared["peak_probability"],
            "temporal_basis_dim": 3, "boundary_taper": float(taper[t].cpu()),
            "history_depth": max(0, t), "risk_score": prepared["peak_probability"],
            "proposal_probability": prepared["peak_probability"],
            "raw_chain_error": raw_error, "final_chain_error": final_error,
            "reprojection_improvement": (raw_error - final_error) / max(raw_error, 1e-6),
            "raw_direction_error": _direction_error(_safe_numpy(raw_points), obs["target"]),
            "final_direction_error": _direction_error(_safe_numpy(cand_points), obs["target"]),
            "trust_radius_rad": cfg["trust"], "optimizer_steps": actual_steps,
            "depth_target_mm": ctarget if chain["kind"] == "arm" else 0.0,
            "depth_actual_mm": cactual if chain["kind"] == "arm" else 0.0,
            "constraint_kind": ckind, "constraint_target_mm": ctarget,
            "constraint_actual_mm": cactual, "body_rms": body_rms, "body_max": body_max,
            "final_total_loss": float(total.cpu()),
            "final_reprojection_loss": float(parts["reprojection"].cpu()),
            "final_direction_loss": float(parts["direction"].cpu()),
            "final_temporal_pose_loss": float(parts["pose_acc"].cpu()),
            "final_world_motion_loss": float(parts["world_acc"].cpu()),
            "final_depth_loss": float(parts["constraint"].cpu()),
            "final_anchor_loss": float(parts["anchor"].cpu()),
            "p_2d": obs["p_2d"], "p_temporal": obs["p_temporal"],
            "p_occlusion": obs["p_occlusion"], "p_disagreement": obs["p_disagreement"],
            "p_depth": obs["p_depth"], "p_mask": obs["p_mask"],
            "fast_batch_mode": 1,
        })
    ratios = np.asarray(ratios, dtype=float)
    boundary = max(frame_results[0]["body_max"], frame_results[-1]["body_max"])
    if len(frames) >= 3:
        raw_acc = torch.linalg.norm(raw_world[2:] - 2 * raw_world[1:-1] + raw_world[:-2], dim=-1).mean()
        cand_acc = torch.linalg.norm(world_seq[2:] - 2 * world_seq[1:-1] + world_seq[:-2], dim=-1).mean()
    else:
        raw_acc = torch.tensor(0.0, device=prepared["device"])
        cand_acc = torch.tensor(0.0, device=prepared["device"])
    acc_ratio = float((cand_acc / (raw_acc + 1e-4)).cpu())
    reason = "accepted"
    if not np.isfinite(ratios).all():
        reason = "nonfinite_geometry"
    elif max_body > cfg["trust"] + 1e-4:
        reason = "trust_region_violation"
    elif float(np.mean(ratios)) > 1.18 or float(np.quantile(ratios, 0.9)) > 1.35:
        reason = "catastrophic_reprojection"
    elif boundary > 0.16:
        reason = "boundary_discontinuity"
    elif acc_ratio > 3.0 and float(cand_acc.cpu()) > 0.08:
        reason = "catastrophic_world_acceleration"
    elif constraint_data is not None:
        med_actual = float(np.median(constraint_actual))
        med_target = float(np.median(constraint_target))
        if abs(med_actual) < 0.30 * abs(med_target):
            reason = "constraint_too_small"
        elif med_actual * med_target <= 0:
            reason = "constraint_wrong_sign"
    if reason != "accepted":
        return {
            "reason": reason,
            "mean_reprojection_ratio": float(np.mean(ratios)),
            "p90_reprojection_ratio": float(np.quantile(ratios, 0.9)),
            "max_body_delta": max_body, "boundary_delta": boundary,
            "constraint_target_mm": float(np.median(constraint_target)) if constraint_target else 0.0,
            "constraint_actual_mm": float(np.median(constraint_actual)) if constraint_actual else 0.0,
            "world_acceleration_ratio": acc_ratio,
        }, reason
    return frame_results, "accepted"


def _h0v3_fast_precompute_sequence(*, sequence, image, pred, smpl_layer, smpl_model, K_fallback):
    if sequence in _H0V3_PRECOMPUTED:
        return
    print("[H0V3_FAST_BEGIN]", sequence, "batched segment precompute", flush=True)
    height, width = int(image.shape[-2]), int(image.shape[-1])
    observations, total_instances = _h0v3_scan_sequence(
        sequence, pred, smpl_layer, smpl_model, K_fallback, height, width
    )
    _h0v2_write_rows(
        "h0v3_frame_scores.csv", H0V3_SCORE_FIELDS,
        [{k: v for k, v in x.items() if k in H0V3_SCORE_FIELDS} for x in observations],
    )
    segments, segment_rows = _h0v3_extract_segments(sequence, observations, total_instances)
    _h0v2_write_rows("h0v3_segments.csv", H0V3_SEGMENT_FIELDS, segment_rows)
    include_negative = os.environ.get("WAO_HMR_H0V3_FAST_INCLUDE_NEGATIVE", "0") == "1"
    rejects = []
    accepted_rows = 0
    for idx, segment in enumerate(segments):
        prepared = _h0v3_fast_prepare_segment(
            sequence, segment, pred, smpl_layer, smpl_model, K_fallback
        )
        chain = CHAINS[segment["chain"]]
        variants = ["segment_fit", "segment_motion"]
        if chain["kind"] == "arm":
            variants += ["segment_depth_near", "segment_depth_far"]
        else:
            variants += ["segment_bend_positive"]
            if include_negative:
                variants += ["segment_bend_negative"]
        for variant in variants:
            result, reason = _h0v3_fast_optimize_segment(
                prepared, variant, pred, smpl_layer, smpl_model, K_fallback
            )
            if reason != "accepted":
                meta = result or {}
                rejects.append({
                    "sequence": sequence, "segment_id": segment["segment_id"],
                    "track_id": segment["track_id"], "chain": segment["chain"],
                    "variant": variant, "reason": reason,
                    "start_frame": segment["frames"][0]["frame_id"],
                    "end_frame": segment["frames"][-1]["frame_id"],
                    "center_frame": segment["center_frame"],
                    "peak_probability": segment["peak_probability"],
                    **{
                        k: meta.get(k, "") for k in H0V3_REJECT_FIELDS
                        if k not in {
                            "sequence", "segment_id", "track_id", "chain", "variant",
                            "reason", "start_frame", "end_frame", "center_frame",
                            "peak_probability",
                        }
                    },
                })
                continue
            for item in result:
                frame_id = int(item["segment_center"] + item["segment_offset"])
                _H0V3_FRAME_PROPOSALS.setdefault((sequence, frame_id), []).append(item)
                accepted_rows += 1
        prepared.clear()
        if torch.cuda.is_available() and (idx % 12 == 0):
            torch.cuda.empty_cache()
        if idx % 5 == 0 or idx + 1 == len(segments):
            print(
                "[H0V3_FAST_OPT]", sequence, f"{idx + 1}/{len(segments)}",
                "accepted_rows=", accepted_rows, flush=True,
            )
    _h0v2_write_rows("h0v3_candidate_rejections.csv", H0V3_REJECT_FIELDS, rejects)
    state = _SEQUENCE_STATES[sequence]
    accepted_instances = {
        (frame_id, int(item["track_id"]))
        for (seq, frame_id), values in _H0V3_FRAME_PROPOSALS.items() if seq == sequence
        for item in values
    }
    state["h0_proposals"] = len(accepted_instances)
    state["h0_candidates"] = accepted_rows
    _H0V3_PRECOMPUTED.add(sequence)
    print(
        "[H0V3_FAST_END]", sequence, "observations=", len(observations),
        "segments=", len(segments), "accepted_instances=", len(accepted_instances),
        "candidate_rows=", accepted_rows, flush=True,
    )


def _h0v3_precompute_sequence(*, sequence, image, pred, smpl_layer, smpl_model, K_fallback):
    if sequence in _H0V3_PRECOMPUTED: return
    print("[H0V3_BEGIN]",sequence,"segment-level precompute",flush=True)
    height,width=int(image.shape[-2]),int(image.shape[-1])
    observations,total_instances=_h0v3_scan_sequence(sequence,pred,smpl_layer,smpl_model,K_fallback,height,width)
    _h0v2_write_rows("h0v3_frame_scores.csv",H0V3_SCORE_FIELDS,[{k:v for k,v in x.items() if k in H0V3_SCORE_FIELDS} for x in observations])
    segments,segment_rows=_h0v3_extract_segments(sequence,observations,total_instances)
    _h0v2_write_rows("h0v3_segments.csv",H0V3_SEGMENT_FIELDS,segment_rows)
    rejects=[]; accepted_rows=0
    for idx,segment in enumerate(segments):
        chain=CHAINS[segment["chain"]]
        variants=["segment_fit","segment_motion"]
        if chain["kind"]=="arm": variants += ["segment_depth_near","segment_depth_far"]
        else: variants += ["segment_bend_positive","segment_bend_negative"]
        for variant in variants:
            result,reason=_h0v3_optimize_segment(sequence,segment,variant,pred,smpl_layer,smpl_model,K_fallback)
            if reason!="accepted":
                meta=result or {}; rejects.append({"sequence":sequence,"segment_id":segment["segment_id"],"track_id":segment["track_id"],"chain":segment["chain"],"variant":variant,"reason":reason,
                    "start_frame":segment["frames"][0]["frame_id"],"end_frame":segment["frames"][-1]["frame_id"],"center_frame":segment["center_frame"],"peak_probability":segment["peak_probability"],
                    **{k:meta.get(k,"") for k in H0V3_REJECT_FIELDS if k not in {"sequence","segment_id","track_id","chain","variant","reason","start_frame","end_frame","center_frame","peak_probability"}}})
                continue
            for item in result:
                _H0V3_FRAME_PROPOSALS.setdefault((sequence,int(item["segment_center"]+item["segment_offset"])),[]).append(item); accepted_rows+=1
        if idx%10==0 or idx+1==len(segments):
            print("[H0V3_OPT]",sequence,f"{idx+1}/{len(segments)}","accepted_rows=",accepted_rows,flush=True)
    _h0v2_write_rows("h0v3_candidate_rejections.csv",H0V3_REJECT_FIELDS,rejects)
    state=_SEQUENCE_STATES[sequence]
    accepted_instances={(f,int(x["track_id"])) for (s,f),vals in _H0V3_FRAME_PROPOSALS.items() if s==sequence for x in vals}
    state["h0_proposals"]=len(accepted_instances); state["h0_candidates"]=accepted_rows
    _H0V3_PRECOMPUTED.add(sequence)
    print("[H0V3_END]",sequence,"observations=",len(observations),"segments=",len(segments),"accepted_instances=",len(accepted_instances),"candidate_rows=",accepted_rows,flush=True)

def _h0_candidate_common_row(*, sequence, frame_id, pid, gid, candidate_name, family, chain, variant, metadata, pose, raw_pose, joints, vertices, j2d, raw_geometry, pred, gt_j3d_c, gt_v3d_c, gt_j3d_w, gt_v3d_w, pelvis, image, pred_j2d, history):
    raw_vertices, raw_joints, raw_j2d = raw_geometry
    cand_j_cam = _safe_numpy(joints)[pid]
    cand_v_cam = _safe_numpy(vertices)[pid]
    raw_j_cam = _safe_numpy(raw_joints)[pid]
    raw_v_cam = _safe_numpy(raw_vertices)[pid]
    transform = _safe_numpy(pred.get("T_c2w", [None])[frame_id])
    cand_j_world = _g0_transform_points(transform, cand_j_cam)
    cand_v_world = _g0_transform_points(transform, cand_v_cam)
    raw_j_world = _g0_transform_points(transform, raw_j_cam)
    raw_v_world = _g0_transform_points(transform, raw_v_cam)
    gt_j_world = None if gt_j3d_w is None or gid >= len(gt_j3d_w) else _safe_numpy(gt_j3d_w)[gid]
    gt_v_world = None if gt_v3d_w is None or gid >= len(gt_v3d_w) else _safe_numpy(gt_v3d_w)[gid]
    metrics = _metric_bundle(cand_j_cam, cand_v_cam, _safe_numpy(gt_j3d_c)[gid], _safe_numpy(gt_v3d_c)[gid], pelvis)
    raw_metrics = _metric_bundle(raw_j_cam, raw_v_cam, _safe_numpy(gt_j3d_c)[gid], _safe_numpy(gt_v3d_c)[gid], pelvis)
    world_metrics = _g0_world_metric_bundle(cand_j_world, cand_v_world, gt_j_world, gt_v_world, pelvis)
    raw_world_metrics = _g0_world_metric_bundle(raw_j_world, raw_v_world, gt_j_world, gt_v_world, pelvis)
    body_rms, body_max = _person_body_delta(pose, raw_pose, pid)
    n = min(len(cand_j_cam), len(raw_j_cam))
    pi = min(pelvis, n - 1)
    cam_delta = cand_j_cam[:n] - raw_j_cam[:n]
    rel_delta = (cand_j_cam[:n] - cand_j_cam[pi]) - (raw_j_cam[:n] - raw_j_cam[pi])
    world_delta = cand_j_world[:n] - raw_j_world[:n]
    cam_delta_norm = np.linalg.norm(cam_delta, axis=-1)
    world_delta_norm = np.linalg.norm(world_delta, axis=-1)
    rigid_invariance = float(np.max(np.abs(cam_delta_norm - world_delta_norm)) * 1000.0) if n else 0.0
    world_reconstruction = float("nan")
    try:
        transform_mat = np.asarray(transform, dtype=np.float64)
        if transform_mat.shape == (3, 4):
            transform_mat = np.concatenate([
                transform_mat, np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            ], axis=0)
        elif transform_mat.shape == (3, 3):
            transform_mat = np.block([
                [transform_mat, np.zeros((3, 1), dtype=np.float64)],
                [np.zeros((1, 3), dtype=np.float64), np.ones((1, 1), dtype=np.float64)],
            ])
        inv_transform = np.linalg.inv(transform_mat)
        reconstructed_cam = _g0_transform_points(inv_transform, cand_j_world[:n])
        world_reconstruction = float(np.linalg.norm(reconstructed_cam - cand_j_cam[:n], axis=-1).max() * 1000.0)
    except Exception:
        pass
    endpoints = _g0_valid_indices(n, _G0_ENDPOINTS)
    depth_delta = np.abs(cand_j_cam[:n, 2] - raw_j_cam[:n, 2])
    cand2d = _safe_numpy(j2d)[pid]
    raw2d = _safe_numpy(raw_j2d)[pid]
    n2 = min(len(cand2d), len(raw2d))
    diag = max(float(np.hypot(int(image.shape[-1]), int(image.shape[-2]))), 1.0)
    pair2d = float(np.sqrt(np.mean(np.sum((cand2d[:n2] - raw2d[:n2]) ** 2, axis=-1)))) if n2 else float("nan")
    temporal = _g0_temporal_features(frame_id, cand_j_world, cand_j_cam, raw_j_world, raw_j_cam, history)
    rot = _safe_numpy(pose)[pid, 1:22]
    rot_norm = np.linalg.norm(rot, axis=-1)
    limit_ratio = float(np.mean(rot_norm > 2.6)) if len(rot_norm) else 0.0
    return {
        "sequence": sequence, "frame_id": int(frame_id), "pred_id": int(pid), "gt_id": int(gid),
        "candidate": candidate_name, "candidate_family": family,
        "chain": chain, "variant": variant,
        "proposal_rank": metadata.get("proposal_rank", ""), "track_id": metadata.get("track_id", ""),
        "history_depth": metadata.get("history_depth", temporal.get("temporal_history", 0)),
        "risk_score": metadata.get("risk_score", ""),
        "proposal_probability": metadata.get("proposal_probability", ""),
        "p_2d": metadata.get("p_2d", ""),
        "p_temporal": metadata.get("p_temporal", ""),
        "p_occlusion": metadata.get("p_occlusion", ""),
        "p_disagreement": metadata.get("p_disagreement", ""),
        "p_depth": metadata.get("p_depth", ""),
        "p_mask": metadata.get("p_mask", ""),
        "constraint_kind": metadata.get("constraint_kind", ""),
        "constraint_target_mm": metadata.get("constraint_target_mm", ""),
        "constraint_actual_mm": metadata.get("constraint_actual_mm", ""),
        "raw_chain_error": metadata.get("raw_chain_error", ""),
        "final_chain_error": metadata.get("final_chain_error", ""),
        "reprojection_improvement": metadata.get("reprojection_improvement", ""),
        "raw_direction_error": metadata.get("raw_direction_error", ""),
        "final_direction_error": metadata.get("final_direction_error", ""),
        "trust_radius_rad": metadata.get("trust_radius_rad", ""),
        "optimizer_steps": metadata.get("optimizer_steps", ""),
        "depth_target_mm": metadata.get("depth_target_mm", ""),
        "depth_actual_mm": metadata.get("depth_actual_mm", ""),
        "final_total_loss": metadata.get("final_total_loss", ""),
        "final_reprojection_loss": metadata.get("final_reprojection_loss", ""),
        "final_direction_loss": metadata.get("final_direction_loss", ""),
        "final_temporal_pose_loss": metadata.get("final_temporal_pose_loss", ""),
        "final_world_motion_loss": metadata.get("final_world_motion_loss", ""),
        "final_depth_loss": metadata.get("final_depth_loss", ""),
        "final_anchor_loss": metadata.get("final_anchor_loss", ""),
        "body_rms_from_raw": body_rms, "body_max_from_raw": body_max,
        "camera_joint_rms_from_raw_mm": _g0_rms_mm(cam_delta),
        "pelvis_relative_joint_rms_from_raw_mm": _g0_rms_mm(rel_delta),
        "world_joint_rms_from_raw_mm": _g0_rms_mm(world_delta),
        "world_endpoint_rms_from_raw_mm": _g0_rms_mm(world_delta[endpoints]) if endpoints else float("nan"),
        "mean_depth_delta_from_raw_mm": float(depth_delta.mean() * 1000.0) if len(depth_delta) else float("nan"),
        "max_depth_delta_from_raw_mm": float(depth_delta.max() * 1000.0) if len(depth_delta) else float("nan"),
        "pairwise_2d_rms_from_raw_norm": pair2d / diag if np.isfinite(pair2d) else float("nan"),
        "external_reprojection_error": _external_reprojection_for_person(j2d, pred_j2d, sequence, frame_id, pid, image),
        "temporal_history": temporal.get("temporal_history", 0),
        "world_velocity_excess_mm": temporal.get("world_velocity_excess_mm", float("nan")),
        "world_acceleration_excess_mm": temporal.get("world_acceleration_excess_mm", float("nan")),
        "world_jerk_excess_mm": temporal.get("world_jerk_excess_mm", float("nan")),
        "endpoint_velocity_excess_mm": temporal.get("endpoint_velocity_excess_mm", float("nan")),
        "endpoint_acceleration_excess_mm": temporal.get("endpoint_acceleration_excess_mm", float("nan")),
        "depth_jump_excess_mm": temporal.get("depth_jump_excess_mm", float("nan")),
        "limb_angle_jump_excess_deg": temporal.get("limb_angle_jump_excess_deg", float("nan")),
        "joint_limit_excess_ratio": limit_ratio,
        "physics_penalty_proxy": _g0_physics_penalty(temporal, limit_ratio),
        **metrics, **world_metrics,
        "delta_mpjpe_from_raw": metrics["mpjpe"] - raw_metrics["mpjpe"],
        "delta_pve_from_raw": metrics["pve"] - raw_metrics["pve"],
        "delta_pa_mpjpe_from_raw": metrics["pa_mpjpe"] - raw_metrics["pa_mpjpe"],
        "delta_pa_pve_from_raw": metrics["pa_pve"] - raw_metrics["pa_pve"],
        "delta_world_mpjpe_from_raw": world_metrics["world_mpjpe"] - raw_world_metrics["world_mpjpe"],
        "delta_world_pve_from_raw": world_metrics["world_pve"] - raw_world_metrics["world_pve"],
        "rigid_transform_invariance_error_mm": rigid_invariance,
        "world_reconstruction_error_mm": world_reconstruction,
        "frozen_contract_ok": int(_root_delta_max(pose, raw_pose) <= 1e-7),
    }, {
        "j_cam": cand_j_cam, "j_world": cand_j_world, "j2d": cand2d,
        "delta_mpjpe": metrics["mpjpe"] - raw_metrics["mpjpe"],
        "delta_pve": metrics["pve"] - raw_metrics["pve"],
        "frozen": int(_root_delta_max(pose, raw_pose) <= 1e-7),
    }


def _h0_record_oracle(*, sequence, frame_id, image, pred, pred_j2d, gt_j3d_c, gt_v3d_c, gt_j3d_w, gt_v3d_w, best_match, pelvis_idx, K, smpl_layer, smpl_model, candidates, proposals):
    if pred is None or best_match is None or len(best_match) == 0:
        return
    raw_pose = candidates.get("raw")
    if raw_pose is None:
        return
    pelvis = int(_safe_numpy(pelvis_idx).reshape(-1)[0]) if pelvis_idx is not None else 0
    raw_values = _candidate_geometry(raw_pose, pred, frame_id, smpl_layer, smpl_model, K)
    if raw_values[0] is None:
        return
    raw_geometry = raw_values
    match_map = {int(pid): int(gid) for pid, gid in np.asarray(best_match, dtype=np.int64)}
    output = _run_root()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    pair_rows = []
    per_instance_candidates = defaultdict(list)
    standard = [("a2", "a2"), ("a2lite", "a2lite"), ("bbox", "bbox"), ("window", "window")]
    standard_geometry = {}
    for name, _family in standard:
        pose = candidates.get(name)
        if pose is None:
            continue
        values = _candidate_geometry(pose, pred, frame_id, smpl_layer, smpl_model, K)
        if values[0] is not None:
            standard_geometry[name] = values
    for pid, gid in match_map.items():
        history = _H0_GT_HISTORY.setdefault((sequence, gid), [])
        raw_row, raw_item = _h0_candidate_common_row(
            sequence=sequence, frame_id=frame_id, pid=pid, gid=gid,
            candidate_name="raw", family="raw", chain="", variant="raw", metadata={},
            pose=raw_pose, raw_pose=raw_pose, joints=raw_values[1], vertices=raw_values[0], j2d=raw_values[2],
            raw_geometry=raw_geometry, pred=pred, gt_j3d_c=gt_j3d_c, gt_v3d_c=gt_v3d_c,
            gt_j3d_w=gt_j3d_w, gt_v3d_w=gt_v3d_w, pelvis=pelvis, image=image,
            pred_j2d=pred_j2d, history=history,
        )
        rows.append(raw_row)
        per_instance_candidates[(pid, gid)].append(("raw", "", "", raw_item))
        for name, family in standard:
            pose = candidates.get(name)
            if pose is None or pid >= int(pose.shape[0]):
                continue
            body_rms, body_max = _person_body_delta(pose, raw_pose, pid)
            if not np.isfinite(body_max) or body_max <= 1e-7:
                continue
            values = standard_geometry.get(name)
            if values is None:
                continue
            row, item = _h0_candidate_common_row(
                sequence=sequence, frame_id=frame_id, pid=pid, gid=gid,
                candidate_name=name, family=family, chain=(candidates.get("e4_stats", {}).get("selected_chain", "") if name == "window" else ""),
                variant=name, metadata={}, pose=pose, raw_pose=raw_pose,
                joints=values[1], vertices=values[0], j2d=values[2], raw_geometry=raw_geometry,
                pred=pred, gt_j3d_c=gt_j3d_c, gt_v3d_c=gt_v3d_c,
                gt_j3d_w=gt_j3d_w, gt_v3d_w=gt_v3d_w, pelvis=pelvis, image=image,
                pred_j2d=pred_j2d, history=history,
            )
            rows.append(row)
            per_instance_candidates[(pid, gid)].append((name, row["chain"], "", item))
        raw_j_world = _g0_transform_points(_safe_numpy(pred.get("T_c2w", [None])[frame_id]), _safe_numpy(raw_values[1])[pid])
        history.append({"frame_id": int(frame_id), "world": np.array(raw_j_world, copy=True), "cam": np.array(_safe_numpy(raw_values[1])[pid], copy=True)})
        if len(history) > 3:
            del history[:-3]
    for proposal in proposals:
        pid = int(proposal["pid"])
        if pid not in match_map:
            continue
        gid = match_map[pid]
        history = _H0_GT_HISTORY.get((sequence, gid), [])
        # The current raw frame was appended above; remove it temporarily so the
        # candidate is evaluated as an insertion into strictly causal history.
        causal_history = history[:-1] if history and int(history[-1]["frame_id"]) == int(frame_id) else history
        pose = proposal["pose"]
        values = _candidate_geometry(pose, pred, frame_id, smpl_layer, smpl_model, K)
        candidate_name = f"h0:{proposal['chain']}:{proposal['variant']}"
        row, item = _h0_candidate_common_row(
            sequence=sequence, frame_id=frame_id, pid=pid, gid=gid,
            candidate_name=candidate_name, family="h0", chain=proposal["chain"],
            variant=proposal["variant"], metadata=proposal, pose=pose, raw_pose=raw_pose,
            joints=values[1], vertices=values[0], j2d=values[2], raw_geometry=raw_geometry,
            pred=pred, gt_j3d_c=gt_j3d_c, gt_v3d_c=gt_v3d_c,
            gt_j3d_w=gt_j3d_w, gt_v3d_w=gt_v3d_w, pelvis=pelvis, image=image,
            pred_j2d=pred_j2d, history=causal_history,
        )
        rows.append(row)
        per_instance_candidates[(pid, gid)].append((candidate_name, proposal["chain"], proposal["proposal_rank"], item))
    for (pid, gid), items in per_instance_candidates.items():
        h0_items = [item for item in items if item[0].startswith("h0:")]
        for first, second in itertools.combinations(h0_items, 2):
            name_a, chain_a, rank_a, a = first
            name_b, chain_b, rank_b, b = second
            if chain_a != chain_b or rank_a != rank_b:
                continue
            n2 = min(len(a["j2d"]), len(b["j2d"]))
            nj = min(len(a["j_cam"]), len(b["j_cam"]))
            if n2 == 0 or nj == 0:
                continue
            diag = max(float(np.hypot(int(image.shape[-1]), int(image.shape[-2]))), 1.0)
            pair2d = float(np.sqrt(np.mean(np.sum((a["j2d"][:n2] - b["j2d"][:n2]) ** 2, axis=-1)))) / diag
            pi = min(pelvis, nj - 1)
            a_rel = a["j_cam"][:nj] - a["j_cam"][pi]
            b_rel = b["j_cam"][:nj] - b["j_cam"][pi]
            world_delta = a["j_world"][:nj] - b["j_world"][:nj]
            endpoints = _g0_valid_indices(nj, _G0_ENDPOINTS)
            distal = CHAINS[chain_a]["j14"][-1]
            depth_diff = abs(float(a["j_cam"][distal, 2] - b["j_cam"][distal, 2])) * 1000.0 if distal < nj else float("nan")
            pair3d = _g0_rms_mm(a_rel - b_rel)
            pair_rows.append({
                "sequence": sequence, "frame_id": int(frame_id), "pred_id": pid, "gt_id": gid,
                "chain": chain_a, "proposal_rank": rank_a,
                "candidate_a": name_a, "candidate_b": name_b,
                "pair_2d_rms_norm": pair2d,
                "pair_camera_pelvis_relative_rms_mm": pair3d,
                "pair_world_joint_rms_mm": _g0_rms_mm(world_delta),
                "pair_world_endpoint_rms_mm": _g0_rms_mm(world_delta[endpoints]) if endpoints else float("nan"),
                "pair_distal_depth_difference_mm": depth_diff,
                "near_2d_0025": int(pair2d <= 0.0025), "near_2d_005": int(pair2d <= 0.005),
                "near_2d_01": int(pair2d <= 0.01),
                "different_3d_20mm": int(pair3d >= 20.0),
                "different_3d_40mm": int(pair3d >= 40.0),
                "different_3d_60mm": int(pair3d >= 60.0),
                "candidate_a_delta_mpjpe": a["delta_mpjpe"], "candidate_b_delta_mpjpe": b["delta_mpjpe"],
                "candidate_a_delta_pve": a["delta_pve"], "candidate_b_delta_pve": b["delta_pve"],
                "frozen_contract_ok": int(a["frozen"] and b["frozen"]),
            })
    if rows:
        _st = _stage()
        _name = "h0v3_segment_candidate_oracle.csv" if _st in {"h0v3_segment_oracle_audit", "h0v3_fast_h1b_oracle_audit"} else ("h0v2_strong_candidate_oracle.csv" if _st == "h0v2_high_recall_oracle_audit" else "h0_strong_candidate_oracle.csv")
        path = output / _name
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=H0_ORACLE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    if pair_rows:
        _st = _stage()
        _name = "h0v3_segment_candidate_pairs.csv" if _st in {"h0v3_segment_oracle_audit", "h0v3_fast_h1b_oracle_audit"} else ("h0v2_strong_candidate_pairs.csv" if _st == "h0v2_high_recall_oracle_audit" else "h0_strong_candidate_pairs.csv")
        path = output / _name
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=H0_PAIR_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(pair_rows)


def record_eval_frame(
    *, sequence, frame_id, pred_j2d, pred=None, pred_j3d_c=None,
    pred_v3d_c=None, gt_j3d_c=None, gt_v3d_c=None, best_match=None,
    smpl_layer=None, smpl_model=None, pelvis_idx=None, K=None, image=None,
    gt_j3d_w=None, gt_v3d_w=None, pred_T_c2w=None,
    **unused,
):
    if os.environ.get("WAO_HMR_SAFE", "0") != "1":
        return
    stage = _stage()
    # ===== GT_RESIDUAL_V2_RECORD_DISPATCH =====
    if stage == "gt_residual_v2_test":
        from wao_gt_residual_v2_online import record_frame
        record_frame(
            sequence=sequence,
            frame_id=frame_id,
            pred_j2d=pred_j2d,
            **unused,
        )
        return
    # ===== GT_RESIDUAL_V2_RECORD_DISPATCH_END =====
    if stage in {"e4_window_legacy", "e4_online"}:
        _e4_record_online_audit(sequence=sequence, frame_id=frame_id, pred_j2d=pred_j2d)
        return

    if stage in BRANCH_GENERATION_STAGES:
        state = _SEQUENCE_STATES.get(sequence)
        if state is not None:
            state["j2d"][int(frame_id)] = pred_j2d.detach().cpu() if torch.is_tensor(pred_j2d) else np.array(pred_j2d, copy=True)
        return

    if stage not in B_STAGES:
        return

    row = _B_FRAME_STATS.pop((sequence, int(frame_id)), None)
    _FRAME_STATS.pop((sequence, int(frame_id)), None)
    output = _run_root()
    output.mkdir(parents=True, exist_ok=True)
    if row is not None:
        path = output / "b_candidate_audit.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=B_AUDIT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    if stage in {"b6_consensus_fusion", "b6_expanded_consensus", "b6_mask_aware_consensus"}:
        selection_rows = _B6_FRAME_SELECTIONS.pop(
            (sequence, int(frame_id)), []
        )
        if selection_rows:
            path = output / "b6_selection.csv"
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=B6_SELECTION_FIELDS
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(selection_rows)

    if stage not in {"b3_multisource_precision_audit", "b3_sam3_precision_audit", "b5_person_mask_constraint_audit", "g0_world_physics_audit", "h0_strong_local_oracle_audit", "h0v2_high_recall_oracle_audit", "h0v3_segment_oracle_audit", "h0v3_fast_h1b_oracle_audit"}:
        _B_FRAME_CANDIDATES.pop((sequence, int(frame_id)), None)
        return
    candidates = _B_FRAME_CANDIDATES.pop((sequence, int(frame_id)), None)
    if not candidates or pred is None or best_match is None or len(best_match) == 0:
        return
    if smpl_layer is None or smpl_model is None or K is None:
        return
    if stage == "g0_world_physics_audit":
        _g0_record_world_physics(
            sequence=sequence, frame_id=frame_id, image=image, pred=pred,
            pred_j2d=pred_j2d, gt_j3d_c=gt_j3d_c, gt_v3d_c=gt_v3d_c,
            gt_j3d_w=gt_j3d_w, gt_v3d_w=gt_v3d_w,
            best_match=best_match, pelvis_idx=pelvis_idx, K=K,
            smpl_layer=smpl_layer, smpl_model=smpl_model,
            candidates=candidates,
        )
        return
    if stage in {"h0_strong_local_oracle_audit", "h0v2_high_recall_oracle_audit", "h0v3_segment_oracle_audit", "h0v3_fast_h1b_oracle_audit"}:
        proposals = _H0_FRAME_PROPOSALS.pop((sequence, int(frame_id)), [])
        _h0_record_oracle(
            sequence=sequence, frame_id=frame_id, image=image, pred=pred,
            pred_j2d=pred_j2d, gt_j3d_c=gt_j3d_c, gt_v3d_c=gt_v3d_c,
            gt_j3d_w=gt_j3d_w, gt_v3d_w=gt_v3d_w,
            best_match=best_match, pelvis_idx=pelvis_idx, K=K,
            smpl_layer=smpl_layer, smpl_model=smpl_model,
            candidates=candidates, proposals=proposals,
        )
        return
    raw_pose = candidates.get("raw")
    geometry = {}
    precision_names = ["raw", "a2", "window", "visible", "bbox"]
    if stage == "b3_sam3_precision_audit":
        precision_names = [
            "raw", "a2", "a2lite", "window", "bbox", "sam3_context"
        ]
    elif stage == "b5_person_mask_constraint_audit":
        precision_names = ["raw", "a2", "a2lite", "window", "bbox"]
    for name in precision_names:
        pose = candidates.get(name)
        if pose is None:
            continue
        geometry[name] = (*_candidate_geometry(pose, pred, frame_id, smpl_layer, smpl_model, K), pose)
    if "raw" not in geometry:
        return
    raw_metrics_by_pid = {}
    rows = []

    if stage == "b5_person_mask_constraint_audit":
        height, width = int(image.shape[-2]), int(image.shape[-1])
        masks = _sam3_instance_masks(sequence, frame_id, height, width)
        raw_vertices, raw_joints, raw_j2d, _ = geometry["raw"]
        person_masks = _match_raw_people_to_masks(raw_j2d, masks, height, width)
        projected_vertices = {
            name: _project_vertices_2d(values[0], K)
            for name, values in geometry.items()
        }
        b5_rows = []
        pelvis = int(_safe_numpy(pelvis_idx).reshape(-1)[0]) if pelvis_idx is not None else 0
        for pair in np.asarray(best_match, dtype=np.int64):
            pid, gid = int(pair[0]), int(pair[1])
            mask_item = person_masks.get(pid)
            if mask_item is None:
                continue
            raw_metrics = None
            raw_mask_loss = None
            per_candidate = []
            for name, (vertices, joints, cand_j2d, pose) in geometry.items():
                if pid >= int(pose.shape[0]):
                    continue
                body_rms, body_max = _person_body_delta(pose, raw_pose, pid)
                changed = int(name != "raw" and np.isfinite(body_max) and body_max > 1e-7)
                if name != "raw" and not changed:
                    continue
                v2d = projected_vertices.get(name)
                if v2d is None or pid >= int(v2d.shape[0]):
                    continue
                features = _mask_candidate_features(
                    _safe_numpy(v2d)[pid], _safe_numpy(cand_j2d)[pid],
                    mask_item, height, width,
                )
                if features is None:
                    continue
                metrics = _metric_bundle(
                    _safe_numpy(joints)[pid], _safe_numpy(vertices)[pid],
                    _safe_numpy(gt_j3d_c)[gid], _safe_numpy(gt_v3d_c)[gid], pelvis,
                )
                if name == "raw":
                    raw_metrics = metrics
                    raw_mask_loss = features["mask_loss"]
                per_candidate.append((name, pose, cand_j2d, body_rms, body_max, changed, features, metrics))
            if raw_metrics is None or raw_mask_loss is None:
                continue
            for name, pose, cand_j2d, body_rms, body_max, changed, features, metrics in per_candidate:
                b5_rows.append({
                    "sequence": sequence, "frame_id": int(frame_id),
                    "pred_id": pid, "gt_id": gid, "candidate": name,
                    "candidate_changed": changed, "mask_matched": 1,
                    "mask_file": mask_item.get("path", ""),
                    "mask_fraction": mask_item.get("fraction", float("nan")),
                    **features,
                    "mask_loss_delta_from_raw": features["mask_loss"] - raw_mask_loss,
                    "body_rms_from_raw": body_rms, "body_max_from_raw": body_max,
                    "consensus_rms": _consensus_rms(candidates, name, pid),
                    "external_reprojection_error": _external_reprojection_for_person(
                        cand_j2d, pred_j2d, sequence, frame_id, pid, image
                    ),
                    "a2_frame_class": candidates.get("a2_frame_class", ""),
                    "window_selected_chain": candidates.get("e4_stats", {}).get("selected_chain", ""),
                    "window_selected_alpha": candidates.get("e4_stats", {}).get("selected_alpha", ""),
                    "window_reprojection_improvement": candidates.get("e4_stats", {}).get("relative_reprojection_improvement", ""),
                    "window_temporal_acceleration_improvement": candidates.get("e4_stats", {}).get("temporal_acceleration_improvement", ""),
                    **metrics,
                    "delta_mpjpe_from_raw": 0.0 if name == "raw" else metrics["mpjpe"] - raw_metrics["mpjpe"],
                    "delta_pve_from_raw": 0.0 if name == "raw" else metrics["pve"] - raw_metrics["pve"],
                    "delta_pa_mpjpe_from_raw": 0.0 if name == "raw" else metrics["pa_mpjpe"] - raw_metrics["pa_mpjpe"],
                    "delta_pa_pve_from_raw": 0.0 if name == "raw" else metrics["pa_pve"] - raw_metrics["pa_pve"],
                    "frozen_contract_ok": int(_root_delta_max(pose, raw_pose) <= 1e-7),
                })
        if b5_rows:
            path = output / "b5_mask_constraint_precision.csv"
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=B5_MASK_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerows(b5_rows)
        return
    pelvis = int(_safe_numpy(pelvis_idx).reshape(-1)[0]) if pelvis_idx is not None else 0
    for pair in np.asarray(best_match, dtype=np.int64):
        pid, gid = int(pair[0]), int(pair[1])
        for name, (vertices, joints, cand_j2d, pose) in geometry.items():
            if pid >= int(pose.shape[0]):
                continue
            body_rms, body_max = _person_body_delta(pose, raw_pose, pid)
            changed = int(name != "raw" and np.isfinite(body_max) and body_max > 1e-7)
            if name != "raw" and not changed:
                continue
            metrics = _metric_bundle(
                _safe_numpy(joints)[pid], _safe_numpy(vertices)[pid],
                _safe_numpy(gt_j3d_c)[gid], _safe_numpy(gt_v3d_c)[gid], pelvis,
            )
            if name == "raw":
                raw_metrics_by_pid[pid] = metrics
            raw_m = raw_metrics_by_pid.get(pid)
            if raw_m is None and name != "raw":
                continue
            rows.append({
                "sequence": sequence, "frame_id": int(frame_id),
                "pred_id": pid, "gt_id": gid, "candidate": name,
                "candidate_available": 1, "candidate_changed": changed,
                "a2_frame_class": candidates.get("a2_frame_class", ""),
                "body_rms_from_raw": body_rms, "body_max_from_raw": body_max,
                "consensus_rms": _consensus_rms(candidates, name, pid),
                "external_reprojection_error": _external_reprojection_for_person(
                    cand_j2d, pred_j2d, sequence, frame_id, pid, image
                ),
                "window_selected_chain": candidates.get("e4_stats", {}).get("selected_chain", ""),
                "window_selected_alpha": candidates.get("e4_stats", {}).get("selected_alpha", ""),
                "window_reprojection_improvement": candidates.get("e4_stats", {}).get("relative_reprojection_improvement", ""),
                "window_temporal_acceleration_improvement": candidates.get("e4_stats", {}).get("temporal_acceleration_improvement", ""),
                **metrics,
                "delta_mpjpe_from_raw": 0.0 if name == "raw" else metrics["mpjpe"]-raw_m["mpjpe"],
                "delta_pve_from_raw": 0.0 if name == "raw" else metrics["pve"]-raw_m["pve"],
                "delta_pa_mpjpe_from_raw": 0.0 if name == "raw" else metrics["pa_mpjpe"]-raw_m["pa_mpjpe"],
                "delta_pa_pve_from_raw": 0.0 if name == "raw" else metrics["pa_pve"]-raw_m["pa_pve"],
                "frozen_contract_ok": int(_root_delta_max(pose, raw_pose) <= 1e-7),
            })
    if rows:
        path = output / "b_candidate_precision.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=B_PRECISION_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

def finish_sequence(sequence: str):
    stage = _stage()
    # ===== GT_RESIDUAL_V2_FINISH_DISPATCH =====
    if stage == "gt_residual_v2_test":
        from wao_gt_residual_v2_online import finish_sequence as finish_gt_v2
        finish_gt_v2(sequence)
        _reset_e4_sequence(sequence)
        return
    # ===== GT_RESIDUAL_V2_FINISH_DISPATCH_END =====
    if stage in BRANCH_GENERATION_STAGES:
        state = _SEQUENCE_STATES.pop(sequence, None)
        if state is not None:
            n_frames = int(state.get("n_frames", 0))
            j2d = [state["j2d"].get(i, torch.empty((0,14,2))) for i in range(n_frames)]
            root = _branch_candidate_root() / state["branch"]
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{sequence}.pt"
            torch.save({
                "sequence": sequence, "branch": state["branch"],
                "rotvec": state["rotvec"], "j2d": j2d,
                "coordinates_unchanged": True,
                "body_only_use_required": True,
            }, path)
            print("[WAO_B_BRANCH_SAVED]", "sequence=", sequence, "branch=", state["branch"], "path=", path, flush=True)
    elif stage in B_STAGES:
        output = _run_root() / "b_candidate_audit.csv"
        state = _SEQUENCE_STATES.pop(sequence, None)
        print(
            "[WAO_UNIFIED_B_END]",
            "sequence=", sequence, "stage=", stage, "audit=", output,
            "a2_updates=", 0 if state is None else state["a2_stats"].get("updates", 0),
            "h0_proposals=", 0 if state is None else state.get("h0_proposals", 0),
            "h0_candidates=", 0 if state is None else state.get("h0_candidates", 0),
            flush=True,
        )
    for key in list(_B_FRAME_CANDIDATES):
        if key[0] == sequence:
            _B_FRAME_CANDIDATES.pop(key, None)
    for key in list(_B6_FRAME_SELECTIONS):
        if key[0] == sequence:
            _B6_FRAME_SELECTIONS.pop(key, None)
    for key in list(_G0_RAW_HISTORY):
        if key[0] == sequence:
            _G0_RAW_HISTORY.pop(key, None)
    for key in list(_H0_FRAME_PROPOSALS):
        if key[0] == sequence:
            _H0_FRAME_PROPOSALS.pop(key, None)
    for key in list(_H0_GT_HISTORY):
        if key[0] == sequence:
            _H0_GT_HISTORY.pop(key, None)
    _H0_TRACKS.pop(sequence, None)
    _H0_NEXT_TRACK.pop(sequence, None)
    for key in list(_H0V2_FEATURE_HISTORY):
        if key[0] == sequence:
            _H0V2_FEATURE_HISTORY.pop(key, None)
    for key in list(_H0V2_LAST_CHAIN_PROPOSAL):
        if key[0] == sequence:
            _H0V2_LAST_CHAIN_PROPOSAL.pop(key, None)
    _H0V3_PRECOMPUTED.discard(sequence)
    for key in list(_H0V3_FRAME_PROPOSALS):
        if key[0] == sequence:
            _H0V3_FRAME_PROPOSALS.pop(key, None)
    _reset_e4_sequence(sequence)
