import importlib.util
import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.global_human.metadata import dataset_metadata
from eval.global_human.utils import *

from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r

from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="bedlam",
        choices=list(dataset_metadata.keys()),
    )

    parser.add_argument("--crop_res", type=int, nargs=2, metavar=("W", "H"), default=None)
    parser.add_argument("--size", type=int, default="224")
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--is_naive", action="store_true", default=False)
    parser.add_argument("--use_ttt3r", action="store_true", default=False)
    parser.add_argument("--reset_interval", type=int, default=100000000)
    parser.add_argument("--use_fake_K", action="store_true", default=False)
    return parser

def get_seq_list(metadata, img_path):
    get_seq_func = metadata.get("get_seq_func", None)
    split = metadata.get("split", "")
    
    if get_seq_func:
        annots = metadata["get_annot_func"](img_path, split)
        seq_list, seq_to_images = get_seq_func(img_path, split, annots)
        return seq_list, seq_to_images, annots
    
    if metadata.get("full_seq", False):
        seq_dir = f"{img_path}/{split}"
        seq_list = [seq for seq in os.listdir(seq_dir) 
                   if os.path.isdir(os.path.join(seq_dir, seq))]
        return sorted(seq_list), None, None
    else:
        return  sorted(metadata.get("seq_list", [])), None, None

def get_file_list(metadata, img_path, seq, seq_to_images=None):
    dir_path = metadata["dir_path_func"](img_path, seq)
    subsample = metadata.get("subsample", 1)
    max_frames = metadata.get("max_frames", None)

    if seq_to_images is not None:
        filelist = [os.path.join(dir_path, name) for name in seq_to_images[seq]]
    else:
        filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]

    filelist.sort()
    
    if max_frames is not None:
        filelist = filelist[:max_frames]

    sampled_indices = list(range(0, len(filelist), subsample))
    filelist = filelist[::subsample]
    return filelist, sampled_indices

def run_inference(
        device, args, metadata, filelist, sampled_indices, annots, model, smpl_model):
    from dust3r.inference import inference_recurrent_lighter
    
    get_view_func=metadata.get("get_view_func", None)
    mask_path_func = metadata.get("mask_path_func", None)    
    mask_path_list = mask_path_func(filelist) if mask_path_func is not None else []

    views = prepare_input(
        filelist,
        [True for _ in filelist],
        msk_paths=mask_path_list,
        size=args.size,
        crop=not args.no_crop,
        revisit=args.revisit,
        update=not args.freeze_state,
        load_func=get_view_func,
        annots=annots,
        sampled_indices=sampled_indices,
        reset_interval=args.reset_interval,
        crop_res=args.crop_res
    )
    with torch.no_grad():
        smpl_model.update_smpl_gt_eval(views, args.eval_dataset)     
    gt = prepare_gt(views)

    # Unified B-group input branch hook. GT is prepared from the original
    # views first; only the model input image is changed afterwards.
    if os.environ.get("WAO_HMR_SAFE", "0") == "1":
        method_dir = "/root/data1/experiments/Human3R/wao_hmr/method"
        if method_dir not in sys.path:
            sys.path.insert(0, method_dir)
        try:
            from wao_hmr_unified import prepare_input_branch
            views = prepare_input_branch(
                views, filelist=filelist,
                sampled_indices=sampled_indices, args=args,
            )
        except Exception as exc:
            print(
                "[WAO_UNIFIED_INPUT_HOOK] failed, use raw views:",
                repr(exc), flush=True,
            )

    keep_keys = set(
        ["img", "img_mask", "true_shape", "img_mhmr", "reset", "update", "K_mhmr"])
    for i, view in enumerate(views):
        views[i] = {key: view[key] for key in keep_keys if key in view}

    outputs, _ = inference_recurrent_lighter(
        views, model, device, verbose=False, is_naive=args.is_naive, use_ttt3r=args.use_ttt3r)

    pred = prepare_output(
        outputs, revisit=args.revisit, solve_pose=args.solve_pose, is_save=args.save)

    pred = _wao_safe_apply_to_pred(pred, filelist=filelist, sampled_indices=sampled_indices, args=args)

    del outputs, views
    gc.collect()
    torch.cuda.empty_cache()

    return gt, pred


# ====================== WAO_HMR_SAFE_HOOK_BEGIN ======================
def _wao_safe_apply_to_pred(pred, filelist=None, sampled_indices=None, args=None):
    """
    Thin unified WAO-HMR sequence hook.

    All mainline A2/E4/B logic lives in:
      /root/data1/experiments/Human3R/wao_hmr/method/wao_hmr_unified.py
    """
    if os.environ.get("WAO_HMR_SAFE", "0") != "1":
        return pred

    method_dir = "/root/data1/experiments/Human3R/wao_hmr/method"
    if method_dir not in sys.path:
        sys.path.insert(0, method_dir)

    try:
        from wao_hmr_unified import apply_sequence_stage
        return apply_sequence_stage(
            pred,
            filelist=filelist,
            sampled_indices=sampled_indices,
            args=args,
        )
    except Exception as exc:
        print(
            "[WAO_UNIFIED_SEQUENCE_HOOK] failed, return raw pred:",
            repr(exc),
            flush=True,
        )
        return pred
# ====================== WAO_HMR_SAFE_HOOK_END ======================

def get_pred_smpl(pred, gt, f_id, is_naive, smpl_layer, mhmr_img_res, K_to_proj):
    n_humans_i = pred['shape'][f_id].shape[0]
    expand = lambda x: x.expand(n_humans_i, -1, -1)

    with torch.no_grad():
        if is_naive:
            dist = pred['transl'][f_id][:, 0].unsqueeze(-1)
            dist = to_euclidean_dist(
                mhmr_img_res, dist, expand(gt['K_mhmr'][f_id]))
            smpl_out = smpl_layer(
                pred['rotvec'][f_id], 
                pred['shape'][f_id], 
                None, 
                pred['loc'][f_id], 
                dist, 
                K=expand(gt['K_mhmr'][f_id]), 
                expression=pred['expression'][f_id],
                K_to_proj=expand(gt['K'][f_id]),
                )
            pred['transl'][f_id] = smpl_out['smpl_transl']
        else:
            smpl_out = smpl_layer(
                pred['rotvec'][f_id], 
                pred['shape'][f_id], 
                pred['transl'][f_id], 
                None, None, 
                K=expand(K_to_proj[f_id]), 
                expression=pred['expression'][f_id])
    
    return smpl_out['smpl_v3d']

def match_2d(pr_j2d, gt_j2d):
    # match pred to gt - based on 2d bbox
    gt_j2d = gt_j2d.numpy()
    bestMatch, falsePositives, misses = match_2d_greedy(
        pr_j2d.numpy()[:,:gt_j2d.shape[1]], 
        gt_j2d, 
        np.ones_like(gt_j2d[...,0]).astype(np.bool_))

    update = {
        'count': len(gt_j2d),
        'miss': len(misses), 
        'fp': len(falsePositives)
    }

    return bestMatch, update



def eval_smpl_error(args, model, smpl_model, smpl_layer, save_dir=None):
    from dust3r.utils.geometry import perspective_projection, geotrf

    # ===== WAO_J2S_IMPORT =====
    _wao_j2s_refine = None
    _wao_j2s_stage = os.getenv(
        "WAO_HMR_STAGE", ""
    ).strip().lower()
    if _wao_j2s_stage == "joint_to_smpl_bounded":
        _wao_j2s_method_dir = (
            "/root/data1/experiments/Human3R/wao_hmr/method"
        )
        if _wao_j2s_method_dir not in sys.path:
            sys.path.insert(
                0,
                _wao_j2s_method_dir,
            )
        from wao_joint_correction import (
            refine_joint_to_smpl_frame
            as _wao_j2s_refine,
        )
        print(
            "[WAO_J2S] official frame hook loaded",
            flush=True,
        )
    # ===== WAO_J2S_IMPORT_END =====

    # ===== WAO_E5_DIRECT_IMPORT =====
    e5_refine = None
    e5_record = None
    _wao_e5_stage = os.getenv("WAO_HMR_STAGE", "").strip().lower()
    print("[WAO_E5_DIRECT] requested_stage=", _wao_e5_stage or "<inactive>", flush=True)
    if _wao_e5_stage in {
        "e5_pixel_audit",
        "e5_pixel_refit",
        "e5_temporal_refit",
        "e5_shadow_counterfactual",
    }:
        import sys as _wao_e5_sys
        _wao_e5_method_dir = "/root/data1/experiments/Human3R/wao_hmr/method"
        if _wao_e5_method_dir not in _wao_e5_sys.path:
            _wao_e5_sys.path.insert(0, _wao_e5_method_dir)
        if _wao_e5_stage == "e5_temporal_refit":
            from wao_e5_temporal_refit import (
                record_online_audit as e5_record,
                refine_online_frame as e5_refine,
            )
        elif _wao_e5_stage == "e5_shadow_counterfactual":
            from wao_e5_shadow_counterfactual import (
                record_online_audit as e5_record,
                refine_online_frame as e5_refine,
            )
        else:
            from wao_e5_pixel_refit import (
                record_online_audit as e5_record,
                refine_online_frame as e5_refine,
            )
        print(
            "[WAO_E5_DIRECT] hook_loaded=True",
            "refine=", e5_refine.__module__,
            "record=", e5_record.__module__,
            flush=True,
        )
    # ===== WAO_E5_DIRECT_IMPORT_END =====

    # ===== WAO_T1_JCR_IMPORT =====
    _wao_t1_record = None
    _wao_t1_stage = os.getenv("WAO_HMR_STAGE", "").strip().lower()
    if _wao_t1_stage == "t1_joint_dump":
        import sys as _wao_t1_sys
        _wao_t1_method_dir = (
            "/root/data1/experiments/Human3R/wao_hmr/method"
        )
        if _wao_t1_method_dir not in _wao_t1_sys.path:
            _wao_t1_sys.path.insert(0, _wao_t1_method_dir)
        from wao_t1_joint_dump import (
            record_joint_dump as _wao_t1_record,
        )
        print("[WAO_T1_JCR] dump hook loaded", flush=True)
    # ===== WAO_T1_JCR_IMPORT_END =====


    # ===== WAO_UNIFIED_FRAME_IMPORT =====
    wao_frame_process = None
    wao_frame_record = None
    wao_finish_sequence = None
    if os.environ.get("WAO_HMR_SAFE", "0") == "1":
        method_dir = "/root/data1/experiments/Human3R/wao_hmr/method"
        if method_dir not in sys.path:
            sys.path.insert(0, method_dir)
        from wao_hmr_unified import (
            finish_sequence as wao_finish_sequence,
            process_eval_frame as wao_frame_process,
            record_eval_frame as wao_frame_record,
        )
    # ===== WAO_UNIFIED_FRAME_IMPORT_END =====

    # ===== WAO_GT_RESIDUAL_SPLIT_OVERRIDE_BEGIN =====
    metadata = deepcopy(dataset_metadata.get(args.eval_dataset))
    _wao_3dpw_split = os.environ.get("WAO_3DPW_SPLIT", "").strip().lower()
    if args.eval_dataset == "3dpw" and _wao_3dpw_split:
        if _wao_3dpw_split not in {"train", "validation", "test"}:
            raise RuntimeError(f"invalid WAO_3DPW_SPLIT={_wao_3dpw_split!r}")
        metadata["split"] = _wao_3dpw_split
        print("[WAO_3DPW_SPLIT]", _wao_3dpw_split, flush=True)
    # ===== WAO_GT_RESIDUAL_SPLIT_OVERRIDE_END =====
    img_path = metadata["img_path"]
    mhmr_img_res = getattr(model, "mhmr_img_res", None)
    subsample = metadata.get("subsample", 1)
    is_global = metadata["is_global"](metadata.get("split", ""))
    pelvis_idx = smpl_model.pelvis_idx

    seq_list, seq_to_images, annots = get_seq_list(metadata, img_path)

    # ===== WAO_J2S_SEQUENCE_FILTER =====
    _wao_j2s_filter = {
        item.strip()
        for item in os.getenv(
            "WAO_J2S_SEQ_FILTER", ""
        ).split(",")
        if item.strip()
    }
    if _wao_j2s_filter:
        seq_list = [
            sequence
            for sequence in seq_list
            if sequence in _wao_j2s_filter
        ]
        if not seq_list:
            raise RuntimeError(
                "WAO_J2S_SEQ_FILTER matched no sequence"
            )
        print(
            "[WAO_J2S] sequence_filter=",
            seq_list,
            flush=True,
        )
    # ===== WAO_J2S_SEQUENCE_FILTER_END =====



    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        if len(seq_list) < distributed_state.num_processes:
            if distributed_state.process_index >= len(seq_list):
                seqs = []
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"


        # ===== WAO_E5_SEQUENCE_FILTER =====
        _wao_e5_seq_filter = {
            value.strip()
            for value in os.getenv("WAO_E4_SEQ_FILTER", "").split(",")
            if value.strip()
        }
        if _wao_e5_seq_filter:
            seqs = [
                sequence
                for sequence in list(seqs)
                if sequence in _wao_e5_seq_filter
            ]
            print(
                "[WAO_E5_DIRECT] sequence_filter=",
                sorted(_wao_e5_seq_filter),
                "resolved_sequences=", seqs,
                flush=True,
            )
        # ===== WAO_E5_SEQUENCE_FILTER_END =====

        # ===== WAO_T1_JCR_SEQUENCE_LIMIT =====
        if os.getenv("WAO_HMR_STAGE", "").strip().lower() == "t1_joint_dump":
            _wao_t1_max_seq = int(
                os.getenv("WAO_T1_MAX_SEQUENCES", "0") or 0
            )
            if _wao_t1_max_seq > 0:
                seqs = list(seqs)[:_wao_t1_max_seq]
                print(
                    "[WAO_T1_JCR] limited sequences=",
                    seqs,
                    flush=True,
                )
        # ===== WAO_T1_JCR_SEQUENCE_LIMIT_END =====

        for seq_idx, seq in enumerate(tqdm(seqs)):
            try:
                print(f"Evaluating sequence: {seq}")
                filelist, sampled_indices = get_file_list(metadata, img_path, seq, seq_to_images)
                gt, pred = run_inference(
                    device, args, metadata, filelist, sampled_indices, annots, model, smpl_model)

                K_to_proj = gt['K'] if args.is_naive else pred['K'] # CHOOSE ONE in: pred['K'] or gt['K']
                T_c2w = pred['T_c2w'] # CHOOSE ONE in: pred['T_c2w'] or gt['T_c2w']

                # ===== WAO_GT_RESIDUAL_SEQUENCE_BEGIN =====
                _wao_gt_stage = os.environ.get("WAO_HMR_STAGE", "").strip().lower()
                if _wao_gt_stage == "gt_residual_dump":
                    import sys as _wao_gt_sys
                    _wao_gt_method_dir = "/root/data1/experiments/Human3R/wao_hmr/method"
                    if _wao_gt_method_dir not in _wao_gt_sys.path:
                        _wao_gt_sys.path.insert(0, _wao_gt_method_dir)
                    from wao_gt_residual_data import begin_sequence as _wao_gt_begin_sequence
                    _wao_gt_begin_sequence(sequence=seq, pred=pred, filelist=filelist)
                # ===== WAO_GT_RESIDUAL_SEQUENCE_END =====



                global_batch = []
                metrics = defaultdict(list)
                counter = Counter()
                for f_id in range(len(filelist)):
                    n_humans_i = pred['shape'][f_id].shape[0]
                    if n_humans_i > 0:

                        pred_v3d_c = get_pred_smpl(
                            pred, gt, f_id, args.is_naive, smpl_layer, mhmr_img_res, K_to_proj)

                        pred_v3d_c = smpl_model.smplx2smpl @ pred_v3d_c
                        pred_j3d_c = smpl_model.j_regressor @ pred_v3d_c
                        pr_j2d = perspective_projection(
                            pred_j3d_c, K_to_proj[f_id].expand(n_humans_i, -1 , -1))
                        # ===== WAO_J2S_FRAME_REFINE =====
                        if _wao_j2s_refine is not None:
                            _wao_j2s_modified = _wao_j2s_refine(
                                sequence=seq,
                                frame_id=f_id,
                                image=gt["img"][f_id],
                                pred=pred,
                                raw_pred_j3d_c=pred_j3d_c,
                                raw_pred_j2d=pr_j2d,
                                smpl_layer=smpl_layer,
                                smpl_model=smpl_model,
                                K=K_to_proj[f_id],
                                pelvis_idx=pelvis_idx,
                            )

                            if _wao_j2s_modified:
                                pred_v3d_c = get_pred_smpl(
                                    pred,
                                    gt,
                                    f_id,
                                    args.is_naive,
                                    smpl_layer,
                                    mhmr_img_res,
                                    K_to_proj,
                                )
                                pred_v3d_c = smpl_model.smplx2smpl @ pred_v3d_c
                                pred_j3d_c = smpl_model.j_regressor @ pred_v3d_c
                                pr_j2d = perspective_projection(
                                    pred_j3d_c,
                                    K_to_proj[f_id].expand(
                                        n_humans_i, -1, -1
                                    ),
                                )
                        # ===== WAO_J2S_FRAME_REFINE_END =====
                        # ===== WAO_E5_DIRECT_REFIT =====
                        if e5_refine is not None:
                            _wao_e5_raw_j3d_c = pred_j3d_c
                            _wao_e5_raw_v3d_c = pred_v3d_c
                            _wao_e5_modified = e5_refine(
                                sequence=seq,
                                frame_id=f_id,
                                image=gt["img"][f_id],
                                pred=pred,
                                raw_pred_j2d=pr_j2d,
                                raw_pred_j3d_c=_wao_e5_raw_j3d_c,
                                raw_pred_v3d_c=_wao_e5_raw_v3d_c,
                                smpl_layer=smpl_layer,
                                smpl_model=smpl_model,
                                K=K_to_proj[f_id],
                            )
                            if _wao_e5_modified:
                                pred_v3d_c = get_pred_smpl(
                                    pred,
                                    gt,
                                    f_id,
                                    args.is_naive,
                                    smpl_layer,
                                    mhmr_img_res,
                                    K_to_proj,
                                )
                                pred_v3d_c = (
                                    smpl_model.smplx2smpl @ pred_v3d_c
                                )
                                pred_j3d_c = (
                                    smpl_model.j_regressor @ pred_v3d_c
                                )
                                pr_j2d = perspective_projection(
                                    pred_j3d_c,
                                    K_to_proj[f_id].expand(
                                        n_humans_i, -1, -1
                                    ),
                                )
                        # ===== WAO_E5_DIRECT_REFIT_END =====
                        # ===== WAO_UNIFIED_FRAME_PROCESS =====
                        if wao_frame_process is not None:
                            wao_modified = wao_frame_process(
                                sequence=seq,
                                frame_id=f_id,
                                image=gt["img"][f_id],
                                pred=pred,
                                raw_pred_j2d=pr_j2d,
                                smpl_layer=smpl_layer,
                                smpl_model=smpl_model,
                                K=K_to_proj[f_id],
                            )
                            if wao_modified:
                                pred_v3d_c = get_pred_smpl(
                                    pred,
                                    gt,
                                    f_id,
                                    args.is_naive,
                                    smpl_layer,
                                    mhmr_img_res,
                                    K_to_proj,
                                )
                                pred_v3d_c = (
                                    smpl_model.smplx2smpl @ pred_v3d_c
                                )
                                pred_j3d_c = (
                                    smpl_model.j_regressor @ pred_v3d_c
                                )
                                pr_j2d = perspective_projection(
                                    pred_j3d_c,
                                    K_to_proj[f_id].expand(
                                        n_humans_i, -1, -1
                                    ),
                                )
                        # ===== WAO_UNIFIED_FRAME_PROCESS_END =====
                    else:
                        pred_v3d_c = torch.empty(0, 6890, 3, dtype=torch.float32)
                        pr_j2d = torch.empty(0, 24, 2, dtype=torch.float32)
                    
                    bestMatch, update = match_2d(pr_j2d, gt['j2d'][f_id])
                    counter.update(update)

                    # ===== WAO_E5_DIRECT_RECORD =====
                    if e5_record is not None:
                        e5_record(
                            sequence=seq,
                            frame_id=f_id,
                            pred_j2d=pr_j2d,
                            pred_j3d_c=(
                                pred_j3d_c if n_humans_i > 0 else None
                            ),
                            pred_v3d_c=(
                                pred_v3d_c if n_humans_i > 0 else None
                            ),
                            gt_j3d_c=gt["j3d_c"][f_id],
                            gt_v3d_c=gt["v3d_c"][f_id],
                            best_match=bestMatch,
                            pelvis_idx=pelvis_idx,
                        )
                    # ===== WAO_E5_DIRECT_RECORD_END =====

                    # ===== WAO_T1_JCR_RECORD =====
                    if _wao_t1_record is not None:
                        _wao_t1_record(
                            sequence=seq,
                            frame_id=f_id,
                            pred_j2d=pr_j2d,
                            pred_j3d_c=(
                                pred_j3d_c
                                if n_humans_i > 0
                                else None
                            ),
                            gt_j2d=gt["j2d"][f_id],
                            gt_j3d_c=gt["j3d_c"][f_id],
                            best_match=bestMatch,
                            pelvis_idx=pelvis_idx,
                        )
                    # ===== WAO_T1_JCR_RECORD_END =====

                    # ===== GT_RESIDUAL_V2_DIRECT_AUDIT =====
                    if os.environ.get("WAO_HMR_STAGE", "").lower() == "gt_residual_v2_test":
                        from wao_gt_residual_v2_online import record_frame as _gt_v2_record_direct
                        _gt_v2_record_direct(
                            sequence=seq,
                            frame_id=f_id,
                            pred_j3d_c=(pred_j3d_c if n_humans_i > 0 else None),
                            pred_v3d_c=(pred_v3d_c if n_humans_i > 0 else None),
                            gt_j3d_c=gt["j3d_c"][f_id],
                            gt_v3d_c=gt["v3d_c"][f_id],
                            best_match=bestMatch,
                            T_c2w=T_c2w[f_id],
                        )
                    # ===== GT_RESIDUAL_V2_DIRECT_AUDIT_END =====

                    # ===== WAO_GT_RESIDUAL_FRAME_BEGIN =====
                    if os.environ.get("WAO_HMR_STAGE", "").strip().lower() == "gt_residual_dump":
                        from wao_gt_residual_data import record_frame as _wao_gt_record_frame
                        _wao_gt_record_frame(
                            sequence=seq,
                            frame_id=f_id,
                            image_path=filelist[f_id],
                            pred=pred,
                            pred_j3d_c=pred_j3d_c if n_humans_i > 0 else None,
                            pred_j2d=pr_j2d,
                            gt_j3d_c=gt['j3d_c'][f_id],
                            gt_v3d_c=gt['v3d_c'][f_id],
                            best_match=bestMatch,
                            K=K_to_proj[f_id],
                            # ===== GT_RESIDUAL_V2_AUDIT_ARGS_SAFE_1 =====
                            pred_v3d_c=(
                                pred_v3d_c if n_humans_i > 0 else None
                            ),
                            # ===== GT_RESIDUAL_V2_AUDIT_ARGS_SAFE_1_END =====
                            T_c2w=T_c2w[f_id],
                        )
                    # ===== WAO_GT_RESIDUAL_FRAME_END =====

                    # ===== WAO_UNIFIED_FRAME_RECORD =====
                    if wao_frame_record is not None:
                        wao_frame_record(
                            sequence=seq,
                            frame_id=f_id,
                            n_frames=len(filelist),
                            image_path=filelist[f_id],
                            image=gt["img"][f_id],
                            pred=pred,
                            pred_j3d_c=(
                                pred_j3d_c if n_humans_i > 0 else None
                            ),
                            pred_v3d_c=(
                                pred_v3d_c if n_humans_i > 0 else None
                            ),
                            pred_j2d=pr_j2d,
                            gt_j2d=gt["j2d"][f_id],
                            gt_j3d_c=gt["j3d_c"][f_id],
                            gt_v3d_c=gt["v3d_c"][f_id],
                            gt_j3d_w=(gt.get("j3d_w", [None] * len(filelist))[f_id] if gt.get("j3d_w") is not None else None),
                            gt_v3d_w=(gt.get("v3d_w", [None] * len(filelist))[f_id] if gt.get("v3d_w") is not None else None),
                            pred_T_c2w=T_c2w[f_id],
                            best_match=bestMatch,
                            K=K_to_proj[f_id],
                            # ===== GT_RESIDUAL_V2_AUDIT_ARGS_SAFE_2 =====
                            T_c2w=T_c2w[f_id],
                            # ===== GT_RESIDUAL_V2_AUDIT_ARGS_SAFE_2_END =====
                            smpl_layer=smpl_layer,
                            smpl_model=smpl_model,
                            pelvis_idx=pelvis_idx,
                        )
                    # ===== WAO_UNIFIED_FRAME_RECORD_END =====

                    # 3d metrics
                    if len(bestMatch) > 0:
                        counter.update({'n_human': len(bestMatch)})
                        pid, gid = bestMatch[:, 0], bestMatch[:, 1]

                        # camera coordinate metrics
                        camcoord_batch = {
                            "pred_j3d": pred_j3d_c[pid],
                            "target_j3d": gt['j3d_c'][f_id][gid],
                            "pred_v3d": pred_v3d_c[pid],
                            "target_v3d": gt['v3d_c'][f_id][gid],
                        }
                        camcoord_metrics = eval_camcoord(camcoord_batch, pelvis_idx)

                        for k, v in camcoord_metrics.items():
                            metrics[k].append(v)

                        if is_global:
                            expand = lambda x: x.expand(len(bestMatch), -1, -1)
                            global_batch.append({
                                "pred_j3d": geotrf(expand(T_c2w[f_id]), pred_j3d_c[pid]),
                                "target_j3d": gt['j3d_w'][f_id][gid],
                                "pred_v3d": geotrf(expand(T_c2w[f_id]), pred_v3d_c[pid]),
                                "target_v3d": gt['v3d_w'][f_id][gid],
                            })

                    if args.save:
                        color = 0.5 * (gt['img'][f_id].permute(1, 2, 0) + 1.0)
                        out_dir = f"{save_dir}/{seq}/{f_id:06d}"
                        for k in ["pts3d", "conf", "color", "camera", "smpl", "mask"]:
                            os.makedirs(os.path.join(out_dir, k), exist_ok=True)
                        np.save(os.path.join(out_dir, "pts3d", f"{f_id:06d}.npy"), pred['pts3d_self'][f_id])
                        np.save(os.path.join(out_dir, "conf", f"pred_{f_id:06d}.npy"), pred['conf_self'][f_id])
                        np.save(os.path.join(out_dir, "color", f"{f_id:06d}.npy"), color)
                        np.save(os.path.join(out_dir, "mask", f"pred_{f_id:06d}.npy"), pred['msk'][f_id])
                        np.savez(os.path.join(out_dir, "camera", f"pred_{f_id:06d}.npz"), 
                                pose=pred['T_c2w'][f_id], K=pred['K'][f_id])
                        np.savez(os.path.join(out_dir, "camera", f"gt_{f_id:06d}.npz"), 
                                pose=gt['T_c2w'][f_id], K=gt['K'][f_id])
                        if len(bestMatch) > 0:
                            np.save(os.path.join(out_dir, "smpl", f"pred_{f_id:06d}.npy"), pred_v3d_c[pid])
                            np.save(os.path.join(out_dir, "smpl", f"gt_{f_id:06d}.npy"), gt['v3d_c'][f_id][gid])
                        else:
                            np.save(os.path.join(out_dir, "smpl", f"pred_{f_id:06d}.npy"), pred_v3d_c)
                            np.save(os.path.join(out_dir, "smpl", f"gt_{f_id:06d}.npy"), gt['v3d_c'][f_id])

                    if args.vis:
                        visualize(
                            save_dir=f"{save_dir}/{seq}",
                            img_path=filelist[f_id],
                            view=gt['img'][f_id],
                            gt_v3d_c=gt['v3d_c'][f_id],
                            pred_v3d_c=pred_v3d_c,
                            K_to_proj=K_to_proj[f_id],
                            gt_K=gt['K'][f_id],
                            bestMatch=bestMatch,
                            smpl_face=smpl_model.smpl_faces['smpl'],
                            )

                metrics['precision'], metrics['recall'], metrics['f1_score']= compute_prf1(
                    counter['count'], counter['miss'], counter['fp'])
                 
                # global coordinate metrics
                if is_global:
                    global_batch = {k: torch.cat([b[k] for b in global_batch]) for k in global_batch[0]}
                    global_metrics = eval_global(global_batch, subsample)
                    for k, v in global_metrics.items():
                        metrics[k].append(v)
                        
                torch.cuda.empty_cache()

                # Write to error log after each sequence
                os.makedirs(save_dir, exist_ok=True)
                write_log(error_log_path, args.eval_dataset, seq, counter, metrics)
                if wao_finish_sequence is not None:
                    wao_finish_sequence(seq)

                # ===== WAO_GT_RESIDUAL_FINISH_BEGIN =====
                if os.environ.get("WAO_HMR_STAGE", "").strip().lower() == "gt_residual_dump":
                    from wao_gt_residual_data import finish_sequence as _wao_gt_finish_sequence
                    _wao_gt_finish_sequence(seq)
                # ===== WAO_GT_RESIDUAL_FINISH_END =====

                del gt, pred, filelist, sampled_indices
                del global_batch, metrics, counter
                if 'global_metrics' in locals():
                    del global_metrics
                gc.collect()


            except Exception as e:
                print(f"Exception in sequence {seq}: {str(e)}")
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e

    distributed_state.wait_for_everyone()
    torch.cuda.empty_cache()

    results = process_directory(save_dir)
    summary = calculate_averages(results)

    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())

            log = get_summary_log(summary)
            f.write(log) 
    
        print(log.strip())
        


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.utils.image import load_masks_for_eval as load_masks
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import weighted_procrustes, to_euclidean_dist, matrix_cumprod
    from dust3r.smpl_model import SMPLModel
    from dust3r.utils import SMPL_Layer
    from dust3r.utils.image import unpad_image

    args.no_crop = False

    def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 1, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        pts3ds_self = pts3ds_self.reshape(B, -1, 3)
        pts3ds_other = pts3ds_other.reshape(B, -1, 3)
        conf_self = conf_self.reshape(B, -1)
        conf_other = conf_other.reshape(B, -1)
        # weighted procrustes
        c2w = weighted_procrustes(
            pts3ds_self,
            pts3ds_other,
            torch.log(conf_self) * torch.log(conf_other),
            use_weights=True,
            return_T=True,
        )
        return c2w, focal, pp.reshape(B, 2)
    
    def _crop_resize(image, intrinsics, crop_res):
        import dust3r.datasets.utils.cropping as cropping
        from dust3r.utils.image import ImgNorm
        
        # image is a tensor in CHW with values in [-1, 1]; convert to HWC uint8 for PIL
        img_device = image.device if isinstance(image, torch.Tensor) else None
        had_batch_dim = False
        if isinstance(image, torch.Tensor):
            # accept [3,H,W] or [1,3,H,W]; squeeze batch if present
            if image.dim() == 4:
                assert image.shape[0] == 1, "_crop_resize expects a single image; got a batch"
                image = image.squeeze(0)
                had_batch_dim = True
            elif image.dim() != 3:
                raise RuntimeError(f"Unexpected image tensor shape {tuple(image.shape)}; expected CHW or 1xCHW")

            image_np = (
                (image.detach().cpu().permute(1, 2, 0).numpy() * 0.5 +0.5) * 255
            ).clip(0, 255).astype(np.uint8)
        else:
            image_np = image

        target_resolution = np.array(crop_res)
        image_pil, _, intrinsics = cropping.rescale_image_depthmap(
            image_np, None, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image_pil.size, crop_res, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, crop_res
        )
        image_pil, _, intrinsics2 = cropping.crop_image_depthmap(
            image_pil, None, intrinsics, crop_bbox
        )

        # convert back to normalized CHW tensor on original device
        image_arr = np.array(image_pil)
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., None], 3, axis=2)
        image_tensor = ImgNorm(image_pil)  # CHW, [-1, 1]
        if had_batch_dim:
            image_tensor = image_tensor.unsqueeze(0)
        if img_device is not None:
            image_tensor = image_tensor.to(img_device)

        # return intrinsics as torch tensor with batch dim like upstream expects
        intrinsics_tensor = torch.from_numpy(intrinsics2).unsqueeze(0)

        return image_tensor, intrinsics_tensor

    def prepare_input(
        img_paths,
        img_mask,
        msk_paths,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
        load_func=None,
        annots=None,
        sampled_indices=None,
        reset_interval=100,
        crop_res=None
    ):
        images = load_images(img_paths, size=size, verbose=False, crop=crop)
        images = load_func((img_paths, images, annots, sampled_indices))

        has_msk = len(msk_paths) > 0
        if has_msk:
            msks = load_masks(msk_paths, size=size, verbose=False, crop=crop)

        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        images[i]["camera_pose"]
                    ).unsqueeze(0),
                    "camera_intrinsics": torch.from_numpy(
                        images[i]["intrinsics"]
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
                }
                
                if crop_res is not None:
                    view["img"], view["camera_intrinsics"] = _crop_resize(
                        images[i]["img"], images[i]["intrinsics"], crop_res)
                    # update true_shape to reflect cropped image size (H, W)
                    view["true_shape"] = torch.tensor([view["img"].shape[-2:]], dtype=torch.int32)

                if has_msk:
                    view["msk"] = msks[i]
                for key in images[i].keys():
                    if key.startswith(("smpl", "T_w2c")):
                        view[key] = torch.tensor(images[i][key]).unsqueeze(0)
                views.append(view)
                if (i+1) % reset_interval == 0:
                    overlap_view = deepcopy(view)
                    overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                    views.append(overlap_view)

        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        images[i]["camera_pose"]
                    ).unsqueeze(0),
                    "camera_intrinsics": torch.from_numpy(
                        images[i]["intrinsics"]
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
                }

                if crop_res is not None:
                    view["img"], view["camera_intrinsics"] = _crop_resize(
                        images[i]["img"], images[i]["intrinsics"], crop_res)
                    # update true_shape to reflect cropped image size (H, W)
                    view["true_shape"] = torch.tensor([view["img"].shape[-2:]], dtype=torch.int32)

                if has_msk:
                    view["msk"] = (
                        msks[j]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    )
                for key in images[i].keys():
                    if key.startswith(("smpl", "T_w2c")):
                        view[key] = torch.tensor(images[i][key]).unsqueeze(0)
                
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
                if (i+1) % reset_interval == 0:
                    overlap_view = deepcopy(view)
                    overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                    views.append(overlap_view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        
        del images
        return views

    def prepare_gt(gts, revisit=1):
        target_out = defaultdict(list)

        valid_length = len(gts) // revisit
        gts = gts[-valid_length:]

        # delet overlaps: reset_mask=True
        gts = [gt for gt in gts if not gt["reset"]]

        intrinsics = [gt["camera_intrinsics"] for gt in gts]
        K_mhmr = [gt["K_mhmr"] for gt in gts]
        camera_pose = [gt["camera_pose"] for gt in gts]
        imgs = [gt["img"] for gt in gts]
        target_out['K'] = torch.cat(intrinsics, 0)
        target_out['K_mhmr'] = torch.cat(K_mhmr, 0)
        target_out['T_c2w'] = torch.cat(camera_pose, 0)
        target_out['img'] = torch.cat(imgs, 0)
        if 'T_w2c' in gts[0]:
            T_w2c_list = [gt["T_w2c"] for gt in gts]
            target_out['T_w2c'] = torch.cat(T_w2c_list, 0)

        smpl_mask_list = [gt["smpl_mask"] for gt in gts]
        for gt, smpl_mask in zip(gts, smpl_mask_list):
            target_out['v3d_c'].append(gt["smpl_v3d_c"][smpl_mask])
            target_out['j3d_c'].append(gt["smpl_j3d_c"][smpl_mask])
            target_out['v3d_w'].append(gt["smpl_v3d_w"][smpl_mask])
            target_out['j3d_w'].append(gt["smpl_j3d_w"][smpl_mask])
            target_out['v2d'].append(gt["smpl_v2d"][smpl_mask])
            target_out['j2d'].append(gt["smpl_j2d"][smpl_mask])
        
        del gts
        return target_out
    
    def prepare_output(outputs, revisit=1, solve_pose=False, is_save=False):
        pred_out = defaultdict(list)

        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
        reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
        shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
        outputs["pred"] = [
            pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
        reset_mask = reset_mask[~shifted_reset_mask]

        if solve_pose:
            pts3ds_self_to_save = [
                output["pts3d_in_self_view"] for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"] for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"] for output in outputs["pred"]]
            conf_other = [output["conf"] for output in outputs["pred"]]
            pr_poses, focal, pp = recover_cam_params(
                torch.cat(pts3ds_self_to_save, 0),
                torch.cat(pts3ds_other, 0),
                torch.cat(conf_self, 0),
                torch.cat(conf_other, 0),
            )
            pts3ds_self = torch.cat(pts3ds_self_to_save, 0)
        else:
            pts3ds_self_to_save = [
                output["pts3d_in_self_view"] for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"] for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"] for output in outputs["pred"]]
            conf_other = [output["conf"] for output in outputs["pred"]]
            pts3ds_self = torch.cat(pts3ds_self_to_save, 0)
            pr_poses = [
                pose_encoding_to_camera(pred["camera_pose"].clone())
                for pred in outputs["pred"]
            ]
            pr_poses = torch.cat(pr_poses, 0)

            B, H, W, _ = pts3ds_self.shape
            pp = (
                torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
                .float()
                .repeat(B, 1)
                .reshape(B, 2)
            )
            focal = estimate_focal_knowing_depth(
                pts3ds_self, pp, focal_mode="weiszfeld"
            )

        if is_save:
            has_mask = "msk" in outputs["pred"][0]
            if has_mask:
                msks = [output["msk"][..., 0] for output in outputs["pred"]]
                msks = [unpad_image(m, [H, W]) for m in msks]
            else:
                msks = [torch.zeros(1, H, W) for _ in range(B)]
            pred_out["msk"] = msks
            pred_out["pts3d_self"] = pts3ds_self_to_save
            pred_out["conf_self"] = conf_self

        if reset_mask.any():
            identity = torch.eye(4, device=pr_poses.device)
            reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
            cumulative_bases = matrix_cumprod(reset_poses)
            shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
            pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        pred_out['T_c2w'] = pr_poses

        intrinsics = torch.eye(3, device=pp.device).unsqueeze(0).repeat(B, 1, 1)
        intrinsics[:, 0, 0] = focal  # fx
        intrinsics[:, 1, 1] = focal  # fy
        intrinsics[:, [0, 1], 2] = pp
        pred_out['K'] = intrinsics

        # get SMPL parameters from outputs
        pred_out['shape'] = [output.get(
            "smpl_shape", torch.empty(1,0,10))[0] for output in outputs["pred"]]
        pred_out['rotvec'] = [roma.rotmat_to_rotvec(output.get(
                "smpl_rotmat", torch.empty(1,0,53,3,3))[0]) for output in outputs["pred"]]
        pred_out['transl'] = [output.get(
            "smpl_transl", torch.empty(1,0,3))[0] for output in outputs["pred"]]
        pred_out['expression'] = [output.get(
            "smpl_expression", [None])[0] for output in outputs["pred"]]
        pred_out['loc'] = [output.get(
            "smpl_loc", torch.empty(1,0,2))[0] for output in outputs["pred"]]
        
        del outputs
        return pred_out


    model = ARCroco3DStereo.from_pretrained(args.weights)
    # SMPL model for gt
    smpl_model = SMPLModel(
        "cpu", 
        model_args={
            'patch_size': model.croco_args['patch_size'], 
            'mhmr_img_res': model.mhmr_img_res, 
            'bb_patch_size': model.bb_patch_size
        },
        eval_args={
            'dataset': args.eval_dataset,
            'use_fake_K': args.use_fake_K
        }
    )
    # SMPL layer for pred
    smpl_layer = SMPL_Layer(type='smplx',
                            gender='neutral',
                            num_betas=10,
                            kid=False, 
                            person_center='head')

    eval_smpl_error(args, model, smpl_model, smpl_layer, save_dir=args.output_dir)
