"""Helpers for center-two-person render-only montage without floor or RGB strip."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw


def unit(a):
    a=np.asarray(a,dtype=np.float64)
    n=float(np.linalg.norm(a))
    return a/n if n>1e-10 else np.array([0.,0.,1.])


def camera_axes(azimuth, elevation):
    a,e=math.radians(azimuth),math.radians(elevation)
    eye=unit([math.sin(a)*math.cos(e), math.sin(e), math.cos(a)*math.cos(e)])
    right=unit(np.cross(np.array([0.,1.,0.]), eye))
    up=unit(np.cross(eye, right))
    return eye,right,up


def from_official_camera(vertices):
    v=np.asarray(vertices,np.float64).copy(); v[:,1:]*=-1.; return v


def hip_anchor(v):
    if len(v)>6008: return (v[1454]+v[6008])*.5
    return np.median(v,axis=0)


def rotate_y(v,deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    return v@np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=np.float64).T


def front_yaw(v):
    if len(v)<=6470: return 0., 'no_SMPL_landmarks'
    l,r=v[3011],v[6470]
    shoulder=l-r; shoulder[1]=0.
    forward=np.cross(shoulder,[0.,1.,0.]); forward[1]=0.
    if np.linalg.norm(forward)<.06: return 0.,'low_shoulders_confidence'
    forward=unit(forward)
    nose=v[332]-(l+r)*.5; nose[1]=0.
    if np.linalg.norm(nose)>.045 and float(np.dot(forward,nose))<0: forward*=-1.
    return math.degrees(math.atan2(float(forward[0]),float(forward[2]))),'single_reference_SMPL_shoulders_nose'


def choose_center_two(npz):
    with np.load(npz,allow_pickle=False) as f:
        verts=np.asarray(f['pred_v3d'],np.float32)
        faces=np.asarray(f['faces'],np.int32)
        K=np.asarray(f['K'],np.float64)
        actual=int(f['actual_frame'].item()) if 'actual_frame' in f.files else int(f['planned_frame'].item())
        source=str(f['source_path'].item())
    if verts.ndim!=3 or verts.shape[0]<2 or verts.shape[2]!=3:
        raise RuntimeError(f'REQUIRE_TWO_CENTER_PEOPLE_BUT_DETECTED={len(verts) if verts.ndim else 0}')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.max()>=verts.shape[1]:
        raise RuntimeError('FACES_INCOMPATIBLE_WITH_PREDICTED_MESH')
    if not np.isfinite(verts).all() or not np.isfinite(K).all():
        raise RuntimeError('NON_FINITE_MESH_OR_INTRINSICS')
    if K.shape!=(3,3): raise RuntimeError('K_MUST_BE_3x3')
    c0=np.array([K[0,2],K[1,2]],np.float64)
    fx=max(1.,abs(K[0,0])); fy=max(1.,abs(K[1,1]))
    scores=[]
    for i,v in enumerate(verts):
        positive=np.asarray(v[:,2]>.05)
        if positive.sum()<150:
            scores.append((float('inf'),float('inf'),i,[None,None])); continue
        pts=(np.asarray(v[positive],np.float64) @ K.T)
        xy=pts[:,:2]/pts[:,2,None]
        xy=xy[np.isfinite(xy).all(1)]
        if len(xy)<150:
            scores.append((float('inf'),float('inf'),i,[None,None])); continue
        low,high=np.percentile(xy,[5,95],axis=0)
        mid=(low+high)*.5
        dist=float(np.linalg.norm((mid-c0)/[fx,fy]))
        area=float(np.prod(np.maximum(high-low,1.)))
        scores.append((dist,-area,i,mid.tolist()))
    chosen=sorted(scores,key=lambda t:(t[0],t[1],t[2]))[:2]
    if any(not math.isfinite(x[0]) for x in chosen):
        raise RuntimeError('TWO_VALID_PERSON_PROJECTIONS_NOT_AVAILABLE')
    ordered=sorted(chosen,key=lambda t:t[3][0])
    indices=[int(t[2]) for t in ordered]
    return [verts[i] for i in indices],faces,{'source_path':source,'actual_frame':actual,
        'detected_people':len(verts),'selected_indices':indices,
        'selected_projected_centers':[t[3] for t in ordered],
        'selection':'two predicted mesh projections closest to image principal point, displayed left-to-right'}


def surface_shades(v,faces,eye,base):
    tri=v[faces]
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    length=np.linalg.norm(n,axis=1)
    n/=np.maximum(length[:,None],1.e-10)
    key=unit([-0.36,0.86,0.37])
    fill=unit([0.64,0.53,0.58])
    diffuse=np.clip(n@key,0,1)
    secondary=np.clip(n@fill,0,1)
    silhouette=1.-np.abs(n@eye)
    smooth=np.zeros_like(v,dtype=np.float64)
    for k in range(3): np.add.at(smooth,faces[:,k],n)
    smooth/=np.maximum(np.linalg.norm(smooth,axis=1,keepdims=True),1.e-10)
    agree=np.clip(np.mean(np.sum(smooth[faces]*n[:,None,:],axis=2),axis=1),0,1)
    cavity=(1.-agree)
    brightness=np.clip(.82+.10*diffuse+.05*secondary-.03*silhouette-.075*cavity,.70,.97)
    return np.clip(np.asarray(base)[None,:]*brightness[:,None],0,255).astype(np.uint8)


def _bounds(v,right,up,px_per_m):
    xy=np.stack([v@right,-v@up],axis=1)*px_per_m
    return xy,np.array([xy[:,0].min(),xy[:,1].min(),xy[:,0].max(),xy[:,1].max()])


def _draw_mesh(draw,v,faces,eye,right,up,base,px_per_m,shift_xy,W,H):
    projection=np.stack([v@right,-v@up],axis=1)*px_per_m+np.asarray(shift_xy)
    depth=v@eye
    tri=projection[faces]
    area=(tri[:,1,0]-tri[:,0,0])*(tri[:,2,1]-tri[:,0,1])-(tri[:,1,1]-tri[:,0,1])*(tri[:,2,0]-tri[:,0,0])
    valid=np.isfinite(tri).all(axis=(1,2)) & (abs(area)>.12)
    valid &= (np.max(tri[:,:,0],axis=1)>-1)&(np.min(tri[:,:,0],axis=1)<W+1)
    valid &= (np.max(tri[:,:,1],axis=1)>-1)&(np.min(tri[:,:,1],axis=1)<H+1)
    idx=np.flatnonzero(valid)
    if len(idx)<20: raise RuntimeError(f'INSUFFICIENT_VISIBLE_TRIANGLES {len(idx)}')
    order=idx[np.argsort(depth[faces[idx]].mean(axis=1),kind='stable')]
    colors=surface_shades(v,faces,eye,base)
    for i in order:
        draw.polygon([tuple(map(int,np.rint(pt))) for pt in tri[i]], fill=tuple(map(int,colors[i])))
    return {'visible_triangles':int(len(order)),'bbox_px':[float(projection[:,0].min()),float(projection[:,1].min()),float(projection[:,0].max()),float(projection[:,1].max())]}


def _prepare_pair(meshes,pixel_multiplier,reference_ortho_scale,camera_azimuth,camera_elevation,camera_target,mesh_gap_px,safe_margin_px,source_native_height,align_body_front):
    ppm=source_native_height/reference_ortho_scale*pixel_multiplier
    eye,right,up=camera_axes(camera_azimuth,camera_elevation)
    world=[from_official_camera(v) for v in meshes]
    if align_body_front=='on':
        yaws=[front_yaw(v)[0] for v in world]; yaw=float(np.mean(yaws)); alignment='mean_two_person_SMPL_front'
    else:
        yaw,alignment=0.,'disabled'
    hips=[hip_anchor(v) for v in world]
    pair_center=np.mean(np.stack(hips,axis=0),axis=0)
    rotated=[rotate_y(v-pair_center,-yaw) for v in world]
    global_ymin=min(float(v[:,1].min()) for v in rotated)
    rotated=[v-np.array([0.,global_ymin,0.],dtype=np.float64)-np.asarray(camera_target,dtype=np.float64) for v in rotated]
    projected=[_bounds(v,right,up,ppm) for v in rotated]
    boxes=[item[1] for item in projected]
    shifted=[b.copy() for b in boxes]
    left_idx=0 if boxes[0][0] <= boxes[1][0] else 1
    right_idx=1-left_idx
    current_gap=boxes[right_idx][0]-boxes[left_idx][2]
    extra_gap=max(0.,mesh_gap_px-current_gap)
    if extra_gap>0:
        rotated[right_idx]=rotated[right_idx].copy()
        rotated[right_idx]+=right[None,:]*(extra_gap/ppm)
        projected=[_bounds(v,right,up,ppm) for v in rotated]
        boxes=[item[1] for item in projected]
    pair_box=np.array([min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)],dtype=np.float64)
    shift_x=safe_margin_px-pair_box[0]
    shift_y=safe_margin_px-pair_box[1]
    W=int(math.ceil(max(640,pair_box[2]-pair_box[0]+2*safe_margin_px)/16.)*16)
    H=int(math.ceil(max(640,pair_box[3]-pair_box[1]+2*safe_margin_px)/16.)*16)
    return {'rotated':rotated,'eye':eye,'right':right,'up':up,'ppm':ppm,'yaw':yaw,'alignment':alignment,
            'shift':(shift_x,shift_y),'pair_box':pair_box,'boxes':boxes,'W':W,'H':H,'hips':hips,'camera_target':camera_target}


def render_pair_no_floor(meshes,faces,*,pixel_multiplier=3.,reference_ortho_scale=9.,camera_azimuth=65.,camera_elevation=0.,camera_distance=4.5,camera_target=(0.,0.,0.),mesh_gap_px=220.,safe_margin_px=140.,source_native_height=1920,align_body_front='on',supersample=1):
    if len(meshes)!=2: raise RuntimeError('EXPECTED_EXACTLY_TWO_MESHES')
    if not (1.<=pixel_multiplier<=5.): raise ValueError('pixel-multiplier must be between 1 and 5')
    if not (-10.<=camera_elevation<=35.): raise ValueError('camera_elevation out of range')
    if camera_distance<=0 or reference_ortho_scale<=0: raise ValueError('invalid camera/ortho scale')
    if supersample not in (1,2): raise ValueError('supersample must be 1 or 2')
    prep=_prepare_pair(meshes,pixel_multiplier,reference_ortho_scale,camera_azimuth,camera_elevation,camera_target,mesh_gap_px,safe_margin_px,source_native_height,align_body_front)
    W,H=prep['W'],prep['H']
    if W*H*supersample**2>33000000:
        if supersample==2:
            supersample=1
            print('[RENDER] Falling back to supersample 1 to prevent excessive memory',flush=True)
        if W*H>50000000: raise RuntimeError('PAIR_IMAGE_TOO_LARGE_REDUCE_PIXEL_MULTIPLIER')
    w,h=W*supersample,H*supersample
    image=Image.new('RGB',(w,h),(255,255,255))
    dr=ImageDraw.Draw(image,'RGB')
    colors=[[252,235,177],[224,206,247]]
    records=[]
    for i,(v,base) in enumerate(zip(prep['rotated'],colors)):
        rec=_draw_mesh(dr,v,faces,prep['eye'],prep['right'],prep['up'],base,prep['ppm']*supersample,
                       (prep['shift'][0]*supersample,prep['shift'][1]*supersample),w,h)
        bbox=(np.asarray(rec['bbox_px'])/supersample).tolist()
        if bbox[0]<safe_margin_px-2 or bbox[2]>W-safe_margin_px+2 or bbox[1]<safe_margin_px-2 or bbox[3]>H-safe_margin_px+2:
            raise RuntimeError(f'MESH_{i}_CLIPPED_OR_BAD_MARGIN: {bbox}, size={W}x{H}')
        rec['bbox_output_px']=[round(float(x),1) for x in bbox]; rec.pop('bbox_px')
        rec['material']='pale_yellow' if i==0 else 'pale_purple'
        records.append(rec)
        print(f'[PAIR MESH {i+1}/2] triangles={rec["visible_triangles"]} bbox={rec["bbox_output_px"]}',flush=True)
    rb=[np.array(r['bbox_output_px'],dtype=np.float64) for r in records]
    order=np.argsort([b[0] for b in rb]); gap=float(rb[order[1]][0]-rb[order[0]][2])
    if gap<mesh_gap_px-2: raise RuntimeError('MESH_GAP_VALIDATION_FAILED')
    if supersample>1: image=image.resize((W,H),Image.Resampling.LANCZOS)
    real_pair_hip_distance=float(np.linalg.norm(prep['hips'][0]-prep['hips'][1]))
    aligned_pair_hip_offset=(hip_anchor(prep['rotated'][1])-hip_anchor(prep['rotated'][0])).tolist()
    meta={'canvas_size':[W,H],'source_native_height':int(source_native_height),'pixel_multiplier':pixel_multiplier,
          'pixel_per_meter':prep['ppm'],'reference_orthographic_scale':reference_ortho_scale,
          'mesh_bbox_gap_px':round(gap,1),'safe_margin_px':safe_margin_px,
          'camera_azimuth':camera_azimuth,'camera_elevation':camera_elevation,'camera_distance':camera_distance,
          'camera_target':list(camera_target),'front_calibration_yaw_deg':prep['yaw'],'front_calibration_method':prep['alignment'],
          'projection':'orthographic','ground_plane':'disabled','shading':'V5 diffuse + cavity coefficients unchanged; different pastel base albedos',
          'supersample_effective':supersample,'real_pair_hip_distance_world':round(real_pair_hip_distance,4),
          'aligned_pair_hip_offset_world':[round(float(x),4) for x in aligned_pair_hip_offset],'people':records}
    return image,meta


def compose_groups_horizontally(images,group_gap_px=520,outer_margin_px=140):
    """Combine five independently rasterized pairs with integer-only PIL offsets.

    Every pair PNG has already passed its own no-crop and two-person gap checks.
    No resizing is performed: 3x mesh detail remains native raster detail.
    """
    if len(images)!=5:
        raise RuntimeError('EXPECTED_EXACTLY_5_GROUP_IMAGES')
    if not all(img.mode == 'RGB' and img.width > 0 and img.height > 0 for img in images):
        raise RuntimeError('INVALID_GROUP_PNG_IMAGES')
    if not (0 <= group_gap_px <= 10000 and 0 <= outer_margin_px <= 10000):
        raise ValueError('INVALID_GROUP_GAP_OR_MARGIN')
    group_gap=int(round(group_gap_px))
    outer_margin=int(round(outer_margin_px))
    heights=[int(img.height) for img in images]
    widths=[int(img.width) for img in images]
    W=sum(widths)+group_gap*(len(images)-1)+outer_margin*2
    H=max(heights)+outer_margin*2
    canvas=Image.new('RGB',(W,H),(255,255,255))
    x=int(outer_margin)
    placements=[]
    for idx,img in enumerate(images):
        y=int((H-img.height)//2)
        canvas.paste(img,(int(x),int(y)))
        placements.append({'group_index':int(idx),'paste_xy':[int(x),int(y)],
                           'size':[int(img.width),int(img.height)]})
        x = int(x + img.width + group_gap)
    assert placements[-1]['paste_xy'][0] + placements[-1]['size'][0] <= W - outer_margin
    return canvas,{'canvas_size':[int(W),int(H)],
                   'group_gap_px':int(group_gap),
                   'outer_margin_px':int(outer_margin),
                   'groups':placements}


def synthetic_montage_self_test():
    ring=18
    t=np.linspace(0,2*math.pi,ring,endpoint=False)
    rings=[]
    for yy,rx,rz in [(0.,.17,.15),(.65,.29,.18),(1.25,.28,.17),(1.65,.14,.15)]:
        rings.append(np.stack([rx*np.cos(t),np.full_like(t,yy),rz*np.sin(t)],axis=1))
    v=np.concatenate(rings)
    faces=[]
    for k in range(len(rings)-1):
        for j in range(ring):
            nj=(j+1)%ring
            a=k*ring+j; b=k*ring+nj; c=(k+1)*ring+j; d=(k+1)*ring+nj
            faces.extend(((a,b,d),(a,d,c)))
    faces=np.asarray(faces,np.int32)
    imgs=[]
    for i in range(5):
        v1=v.copy(); v2=v.copy(); v2[:,0]+=0.05*i; v2[:,2]+=0.12+0.03*i
        c1=v1.copy(); c1[:,1:]*=-1; c2=v2.copy(); c2[:,1:]*=-1
        pic,meta=render_pair_no_floor([c1,c2],faces,source_native_height=720,pixel_multiplier=3.,reference_ortho_scale=9.,mesh_gap_px=90.,safe_margin_px=60.,camera_azimuth=65.,camera_elevation=0.)
        assert len(meta['people'])==2 and meta['ground_plane']=='disabled'
        imgs.append(pic)
    montage,mmeta=compose_groups_horizontally(imgs,group_gap_px=180,outer_margin_px=60)
    assert montage.width>sum(img.width for img in imgs)
    assert len(mmeta['groups'])==5
    # Reproduce V8 failure inputs: floats from argparse must never reach PIL paste.
    float_montage,fl=compose_groups_horizontally(imgs,group_gap_px=180.0,outer_margin_px=60.0)
    assert float_montage.size==montage.size and fl['groups']==mmeta['groups']
    assert all(isinstance(p['paste_xy'][0],int) for p in fl['groups'])
    print('[CPU SELF-TEST PASS] 5 groups, float-valued CLI gaps fixed, no floor, horizontal camera:',montage.size,flush=True)
