#!/usr/bin/env python3
"""Executable ZIP: 5-frame JAR-HMR 3DPW center-two-person montage, no floor, render-only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import time
from pathlib import Path
from PIL import Image

H3R=Path('/root/data1/models/Human3R')
PRJ=Path('/root/data1/experiments/Human3R/wao_hmr')
TMP=PRJ/'tmp_install'
OUT=PRJ/'results_safe/paper_ablation_official_v3/downtown_crossStreets_00_two_person_montage_v9'
CKPT=PRJ/'training/t1_jcr/model_multiseq_magnitude_calibrated/best.pt'
ASSETS=PRJ/'assets/e4_2d_observations.pkl'
ZIPSELF=Path(sys.argv[0]).resolve()
PINNED={
    'METHOD':(PRJ/'method/wao_joint_correction.py','2767206ff8d4db0a45af1c84afca79568542047167bcc4990c5000a8dac10a10'),
    'PIPELINE':(PRJ/'full_3dpw/scripts/wao_joint_correction_pipeline.py','a23b3d5b6d83a5a18ef20ccf72cb2f0049f4ba2130019ddc94ee4dd42fb57d11'),
    'LAUNCH':(H3R/'eval/global_human/launch.py','afc22623607802d43b659a769d6b9dd4fba31daf83e7e7dff5704957d5fbd165'),
    'UNIFIED':(PRJ/'method/wao_hmr_unified.py','e894e81780f5484d32511562cbc9a7432ad7a1701f93b3b200804479c500b665'),
}
PLANNED=[70,90,110,130,150]


def sha256(p):
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def offline_hub():
    candidates=[]
    if os.environ.get('TORCH_HOME'): candidates.append(Path(os.environ['TORCH_HOME']))
    candidates.extend([H3R/'cache/torch',Path('/root/.cache/torch')])
    for home in dict.fromkeys(candidates):
        hub=home/'hub'
        if hub.is_dir() and any((p/'hubconf.py').is_file() for p in hub.glob('facebookresearch_dinov2_*')):
            return home
    raise RuntimeError('DINOv2 OFFLINE CACHE NOT FOUND')


def preflight(strict_sha=False):
    checks={}
    for name,(p,digest) in PINNED.items():
        if not p.is_file(): raise RuntimeError(f'REQUIRED_METHOD_FILE_MISSING_{name}={p}')
        good=sha256(p)==digest
        checks[name]='PASS' if good else 'CHANGED_FROM_OLD_PACKAGE'
        if not good:
            print(f'[VERSION NOTICE] {name} differs from old ZIP; {p}',flush=True)
            if strict_sha: raise RuntimeError(f'STRICT_SHA_MISMATCH_{name}')
    for name,p in [('HUMAN3R_896L',H3R/'src/human3r_896L.pth'),('JCR_CHECKPOINT',CKPT),('EXTERNAL_2D_OBSERVATIONS',ASSETS)]:
        if not p.is_file(): raise RuntimeError(f'{name} missing: {p}')
    return checks,offline_hub()


def list_official_frames(name):
    os.chdir(H3R)
    for p in [str(H3R),str(H3R/'src')]:
        if p not in sys.path: sys.path.insert(0,p)
    from eval.global_human.metadata import dataset_metadata
    from eval.global_human.launch import get_seq_list,get_file_list
    meta=dict(dataset_metadata['3dpw']); meta['img_path']='/root/data1/datasets/3DPW'
    names,images,_=get_seq_list(meta,meta['img_path'])
    if name not in names: raise RuntimeError(f'OFFICIAL_SEQUENCE_NOT_FOUND={name}; available={names}')
    paths,idx=get_file_list(meta,meta['img_path'],name,images)
    return list(map(str,paths)),list(idx)


def invoke_worker(frames,cache_dir,hub,search_radius):
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('WAO_'): env.pop(k)
    env.update(TORCH_HOME=str(hub),TORCH_EXTENSIONS_DIR=str(H3R/'cache/torch_extensions'),
               HUMAN3R_NO_VIEWER='1',PYOPENGL_PLATFORM='egl',PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH='src:.',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               WAO_HMR_STAGE='joint_to_smpl_bounded',WAO_JCR_CKPT=str(CKPT),
               WAO_JCR_OBS=str(ASSETS),WAO_J2S_AUDIT=str(cache_dir.parent/'method_audit.jsonl'))
    cmd=['accelerate','launch','--num_processes','1','--num_machines','1',
         '--mixed_precision','no','--dynamo_backend','no','--main_process_port','29791',
         str(ZIPSELF),'--worker','--sequence','downtown_crossStreets_00',
         '--frames',','.join(map(str,frames)),'--mode','jar','--temp',str(cache_dir),
         '--fallback-radius',str(search_radius),'--fast-window']
    cache_dir.mkdir(parents=True,exist_ok=True)
    print('[OFFICIAL MULTI-FRAME GPU CAPTURE] requested=',frames,'neighbor_radius=',search_radius,flush=True)
    tail=[]
    with subprocess.Popen(cmd,cwd=str(H3R),env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,bufsize=1,text=True,errors='replace') as proc:
        for line in proc.stdout:
            print(line,end='',flush=True)
            tail.append(line.rstrip())
            if len(tail)>40: tail.pop(0)
        rc=proc.wait()
    if rc!=0: raise RuntimeError(f'OFFICIAL_GPU_RETURN_CODE={rc}; last_output='+' / '.join(tail)[-3000:])
    files=[]
    for frame in frames:
        file=cache_dir/f'frame_{frame:05d}.npz'
        if not file.is_file(): raise RuntimeError(f'CAPTURE_NOT_CREATED={file}')
        files.append(file)
    return files


def atomic_png(image,path):
    tmp=path.with_suffix('.incomplete.png')
    image.save(tmp,'PNG',compress_level=5)
    with Image.open(tmp) as chk:
        if chk.size!=image.size: raise RuntimeError('SAVED_SIZE_MISMATCH')
        chk.verify()
    tmp.replace(path)
    print('[SAVED]',path,'SIZE',image.size,flush=True)


def group_image_path(out,slot,requested):
    return out/'group_images'/f'group_{slot:02d}_f{requested:05d}.png'


def group_record_path(out,slot,requested):
    return out/'group_images'/f'group_{slot:02d}_f{requested:05d}.json'


def save_group_record(out,slot,requested,actual,selected,render_meta):
    record={'slot':int(slot),'requested_frame':int(requested),'actual_frame':int(actual),
            'selection':selected,'render':render_meta}
    record_path=group_record_path(out,slot,requested)
    temporary=record_path.with_suffix('.incomplete.json')
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(record_path)
    return record


def load_saved_groups(out):
    images=[];records=[]
    for i,requested in enumerate(PLANNED):
        png=group_image_path(out,i,requested)
        info=group_record_path(out,i,requested)
        if not png.is_file() or not info.is_file():
            raise RuntimeError('COMPOSE_ONLY_NEEDS_ALL_5_SAVED_GROUPS; missing=' + str(png if not png.is_file() else info))
        record=json.loads(info.read_text(encoding='utf-8'))
        if int(record.get('requested_frame',-1)) != requested or int(record.get('slot',-1)) != i:
            raise RuntimeError('GROUP_RECORD_MISMATCH=' + str(info))
        with Image.open(png) as im:
            im.load()
            images.append(im.convert('RGB'))
        records.append(record)
    return images,records


def finalize_montage(out,images,group_meta,args):
    from render_preview import compose_groups_horizontally
    montage,mmeta=compose_groups_horizontally(images,group_gap_px=args.group_gap_px,
                                               outer_margin_px=args.outer_margin_px)
    name=('downtown_crossStreets_00_f00070_00090_00110_00130_00150_'
          'JAR_HMR_two_center_5groups_sidefront25_nofloor.png')
    atomic_png(montage,out/name)
    manifest={'status':'PASS','dataset':'3dpw','sequence':'downtown_crossStreets_00',
              'planned_frames':PLANNED,'actual_frames':[int(g['actual_frame']) for g in group_meta],
              'render_only':True,'floor_plane':'disabled',
              'rgb_comparison_image':'disabled','camera':{
                  'pixel_multiplier':args.pixel_multiplier,'camera_azimuth':args.camera_azimuth,
                  'camera_elevation':args.camera_elevation,'camera_distance':args.camera_distance,
                  'camera_target':args.camera_target,'orthographic_scale':args.orthographic_scale,
                  'mesh_gap_px':args.mesh_gap_px,'safe_margin_px':args.safe_margin_px,
                  'group_gap_px':args.group_gap_px,'outer_margin_px':args.outer_margin_px,
                  'supersample':args.supersample,'align_body_front':args.align_body_front},
              'groups':group_meta,'montage':mmeta,'image_mesh_montage':name,
              'saved_group_images':'group_images/',
              'official_code_edited':False}
    tmp=out/'render_manifest.incomplete.json'
    tmp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(out/'render_manifest.json')
    return name,mmeta


def run(args):
    started=time.monotonic()
    cache=None
    summary={'STATUS':'BLOCKED','SEQUENCE':'downtown_crossStreets_00','PLANNED_FRAMES':PLANNED,
             'PIXEL_MULTIPLIER':args.pixel_multiplier,'CAMERA_AZIMUTH':args.camera_azimuth,
             'CAMERA_ELEVATION':args.camera_elevation,'OUTPUT_DIR':str(Path(args.output_dir).resolve())}
    try:
        if not 0<=args.fallback_radius<=12: raise ValueError('fallback-radius must be in [0,12]')
        if not 1<=args.pixel_multiplier<=5: raise ValueError('pixel-multiplier must be in [1,5]')
        if args.stage=='self-test':
            from render_preview import synthetic_montage_self_test
            synthetic_montage_self_test()
            summary['STATUS']='SELF_TEST_PASS'
            return
        if args.stage=='compose':
            out=Path(args.output_dir).resolve()
            imgs,group_meta=load_saved_groups(out)
            name,mmeta=finalize_montage(out,imgs,group_meta,args)
            summary.update(STATUS='PASS_COMPOSE_ONLY',MONTAGE_IMAGE=name,OUTPUT_SIZE=mmeta['canvas_size'],
                           ACTUAL_FRAMES=[g['actual_frame'] for g in group_meta],
                           GPU_RECOMPUTED='NO')
            return
        audit,hub=preflight(args.strict_sha)
        summary['AUDIT']=audit
        paths,_=list_official_frames('downtown_crossStreets_00')
        total=len(paths)
        if total <= PLANNED[-1]:
            raise RuntimeError(f'SEQUENCE_TOO_SHORT_FOR_FIXED_PLAN total={total}, needs frame {PLANNED[-1]}')
        for f in PLANNED:
            if not Path(paths[f]).is_file(): raise RuntimeError('PLANNED_SOURCE_FRAME_FILE_MISSING='+paths[f])
        summary['TOTAL_SEQUENCE_FRAMES']=total
        if args.stage=='check':
            summary['STATUS']='READY'
            return
        out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
        cache=Path(tempfile.mkdtemp(prefix='_jar_crossstreets_montage_',dir=str(TMP)))
        npz_files=invoke_worker(PLANNED,cache/'captures',hub,args.fallback_radius)
        from render_preview import choose_center_two,render_pair_no_floor,compose_groups_horizontally
        pair_images=[]; group_meta=[]
        for requested,npz in zip(PLANNED,npz_files):
            pred,faces,selected=choose_center_two(npz)
            pair_img,pmeta=render_pair_no_floor(pred,faces,pixel_multiplier=args.pixel_multiplier,
                reference_ortho_scale=args.orthographic_scale,
                camera_azimuth=args.camera_azimuth,camera_elevation=args.camera_elevation,
                camera_distance=args.camera_distance,camera_target=args.camera_target,
                mesh_gap_px=args.mesh_gap_px,safe_margin_px=args.safe_margin_px,
                source_native_height=args.source_native_height,
                align_body_front=args.align_body_front,supersample=args.supersample)
            slot=len(pair_images)
            group_png=group_image_path(out,slot,requested)
            group_png.parent.mkdir(parents=True,exist_ok=True)
            atomic_png(pair_img,group_png)
            record=save_group_record(out,slot,requested,int(selected['actual_frame']),selected,pmeta)
            pair_images.append(pair_img)
            group_meta.append(record)
        name,mmeta=finalize_montage(out,pair_images,group_meta,args)
        summary.update(STATUS='PASS',MONTAGE_IMAGE=name,OUTPUT_SIZE=mmeta['canvas_size'],
                       ACTUAL_FRAMES=[g['actual_frame'] for g in group_meta])
    except Exception as e:
        summary['ERROR']=f'{type(e).__name__}: {e}'
        print(traceback.format_exc(),flush=True)
    finally:
        if cache is not None and cache.exists():
            if args.cleanup_temp:
                shutil.rmtree(cache)
                summary['OWN_TEMP_REMOVED']='YES'
            else:
                summary['OWN_TEMP_DIR']=str(cache)
        summary['ELAPSED_SECONDS']=round(time.monotonic()-started,2)
        print('========== JAR-HMR CENTER-TWO 5-GROUP MONTAGE SUMMARY ==========',flush=True)
        for k,v in summary.items():
            print(f'{k}={json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v}',flush=True)
        print('========== SUMMARY END ==========',flush=True)


def main():
    p=argparse.ArgumentParser(description='Five-frame JAR-HMR 3x-mesh downtown_crossStreets_00 two-center montage, no floor, render only')
    p.add_argument('--stage',choices=['run','check','self-test','compose'],default='run',
                   help='compose: reassemble saved five group PNGs without CUDA/model loading')
    p.add_argument('--output-dir',default=str(OUT))
    p.add_argument('--pixel-multiplier',type=float,default=3.)
    p.add_argument('--camera_azimuth','--camera-azimuth',dest='camera_azimuth',type=float,default=65.,
                   help='65 degrees = camera 25 degrees toward person front from pure 90-degree side view; -65 flips side')
    p.add_argument('--camera_elevation','--camera-elevation',dest='camera_elevation',type=float,default=0.,
                   help='0 degrees = strictly horizontal camera')
    p.add_argument('--camera_distance','--camera-distance',dest='camera_distance',type=float,default=4.5)
    p.add_argument('--camera_target','--camera-target',dest='camera_target',nargs=3,type=float,default=[0.,0.,0.])
    p.add_argument('--orthographic_scale','--orthographic-scale',dest='orthographic_scale',type=float,default=9.)
    p.add_argument('--mesh-gap-px',type=float,default=220.,help='Minimum pixel blank interval between the two projected body bounding boxes inside each group at 3x size')
    p.add_argument('--group-gap-px',type=float,default=520.,help='Pixel blank interval between neighboring two-person groups in the final montage')
    p.add_argument('--outer-margin-px',type=float,default=140.,help='Final montage outer margin')
    p.add_argument('--safe-margin-px',type=float,default=140.,help='Per-group image boundary margin after high-resolution rendering')
    p.add_argument('--source-native-height',type=int,default=1920,help='Reference native video height used to scale 3x raster density')
    p.add_argument('--align-body-front',choices=['on','off'],default='on')
    p.add_argument('--fallback-radius',type=int,default=2,help='Process each target +/- radius to find >=2 predicted people')
    p.add_argument('--supersample',choices=[1,2],type=int,default=1,help='At 3x actual output resolution default 1 avoids extra RAM; optional 2x AA')
    p.add_argument('--strict-sha',action='store_true')
    p.add_argument('--cleanup-temp',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--sequence',default='downtown_crossStreets_00',help=argparse.SUPPRESS)
    p.add_argument('--frames',default='',help=argparse.SUPPRESS)
    p.add_argument('--mode',choices=['jar','gt'],default='jar',help=argparse.SUPPRESS)
    p.add_argument('--temp',default='',help=argparse.SUPPRESS)
    p.add_argument('--fast-window',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        from scene_worker import capture_official
        capture_official(args)
        return
    run(args)

if __name__=='__main__':
    main()
