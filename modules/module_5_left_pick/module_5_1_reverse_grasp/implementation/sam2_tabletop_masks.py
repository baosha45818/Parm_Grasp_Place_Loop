#!/usr/bin/env python3
"""Generate one SAM2 mask per elevated tabletop component from an RGB-D capture."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import cv2, numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tool.inspection_support.mask_depth import depth_to_xyz
from sam2_client import prompt_segment

def distributed_positive_points(labels, members, count=5):
    """Choose seed pixels across the whole component, not just its centroid.

    A long workpiece can have one depth-connected seed component while its
    wide head and narrow body have very different RGB appearance.  SAM2 only
    receives points, not this seed silhouette, so use farthest-point sampling
    to positively mark both ends and the centre of that component.
    """
    yy,xx=np.nonzero(np.isin(labels,members)); points=np.column_stack((xx,yy))
    if len(points)==0: return []
    take=min(int(count),len(points)); chosen=[]
    mean=points.mean(0); first=int(np.argmin(np.sum((points-mean)**2,axis=1)))
    chosen.append(first); nearest=np.sum((points-points[first])**2,axis=1)
    for _ in range(1,take):
        index=int(np.argmax(nearest)); chosen.append(index)
        nearest=np.minimum(nearest,np.sum((points-points[index])**2,axis=1))
    return [[int(v) for v in points[index]] for index in chosen]

def distributed_binary_points(binary, count=6):
    """Farthest-point sample a boolean image region for SAM negative prompts."""
    yy, xx = np.nonzero(binary); points = np.column_stack((xx, yy))
    if len(points) == 0:
        return []
    take = min(int(count), len(points)); chosen = []
    mean = points.mean(0); first = int(np.argmin(np.sum((points - mean) ** 2, axis=1)))
    chosen.append(first); nearest = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, take):
        point = int(np.argmax(nearest)); chosen.append(point)
        nearest = np.minimum(nearest, np.sum((points - points[point]) ** 2, axis=1))
    return [[int(v) for v in points[index]] for index in chosen]

def workspace_roi_mask(workspace, image_shape):
    """Rasterize the calibrated tabletop polygon for this RGB-D frame."""
    height, width = image_shape[:2]
    reference_width, reference_height = workspace['reference_color_size_wh']
    polygon = np.asarray(workspace['safe_roi_polygon_px'], dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError('safe_roi_polygon_px must contain at least three [x, y] points')
    polygon = np.rint(polygon.astype(float) * [width / reference_width, height / reference_height]).astype(np.int32)
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi, [polygon], 1)
    return roi

def main():
 workspace=json.loads((MODULE_ROOT/'config/tabletop_workspace.json').read_text())
 p=argparse.ArgumentParser(); p.add_argument('--capture-dir',type=Path,required=True); p.add_argument('--socket',default=os.environ.get('PARM_SAM2_URL','unix:///home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/sam2-service/runtime/sam2-service.sock')); p.add_argument('--height-mm',type=float,default=3.0); p.add_argument('--min-seed-area',type=int,default=90); p.add_argument('--roi',nargs=4,type=int,default=None,metavar=('X','Y','W','H'),help='debug-only rectangular override; production uses safe_roi_polygon_px'); p.add_argument('--out',type=Path,default=None); a=p.parse_args()
 meta=json.loads((a.capture_dir/'metadata.json').read_text()); rgb=cv2.imread(str(a.capture_dir/'rgb_000.png')); d=cv2.imread(str(a.capture_dir/'depth_aligned_000.png'),cv2.IMREAD_UNCHANGED)
 k=meta['color_intrinsics']; K=np.array([[k['fx'],0,k['ppx']],[0,k['fy'],k['ppy']],[0,0,1.]])
 xyz=depth_to_xyz(d,K,float(meta['depth_scale_m'])); valid=(d>0)&(d<65535); pixel_x=np.indices(d.shape)[1]; pts=xyz[valid]
 import open3d as o3d
 cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)); plane,_=cloud.voxel_down_sample(.003).segment_plane(.004,3,1200); plane=np.asarray(plane); n=plane[:3]/np.linalg.norm(plane[:3]); c=pts.mean(0)
 if np.dot(n,-c)<0: plane=-plane
 h=(xyz@plane[:3]+plane[3])
 # Candidate seeds use geometry only: elevated pixels with valid depth.
 seeds=((h>=a.height_mm/1000)&valid).astype(np.uint8)
 if a.roi is None:
  roi=workspace_roi_mask(workspace, seeds.shape)
 else:
  roi=np.zeros_like(seeds); rx,ry,rw,rh=a.roi; roi[max(0,ry):min(seeds.shape[0],ry+rh),max(0,rx):min(seeds.shape[1],rx+rw)]=1
 seeds*=roi
 seeds=cv2.morphologyEx(seeds,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); seeds=cv2.dilate(seeds,np.ones((5,5),np.uint8),iterations=1)
 count,lab,stats,centers=cv2.connectedComponentsWithStats(seeds)
 out=a.out or a.capture_dir/'sam2_masks'; out.mkdir(parents=True,exist_ok=True); overlay=rgb.copy(); prompt_overlay=rgb.copy(); index=0; masks=[]
 candidates=[label for label in range(1,count) if stats[label,cv2.CC_STAT_AREA]>=a.min_seed_area]
 if not candidates:
  cv2.imwrite(str(out/'largest_component_rgb_overlay.png'),rgb); cv2.imwrite(str(out/'sam2_prompt_overlay.png'),rgb); cv2.imwrite(str(out/'sam2_masks_overlay.png'),rgb)
  np.savez_compressed(out/'masks.npz',masks=np.empty((0,*rgb.shape[:2]),dtype=np.uint8)); (out/'meta.json').write_text(json.dumps({'masks':0,'plane':plane.tolist(),'prompt_groups':[]},indent=2)); print(out); return
 largest_label=max(candidates,key=lambda label: int(stats[label,cv2.CC_STAT_AREA])); members=[largest_label]
 largest_component=(lab==largest_label)
 largest_overlay=rgb.copy(); largest_overlay[largest_component]=(.55*largest_overlay[largest_component]+.45*np.array((0,255,0))).astype(np.uint8)
 cv2.imwrite(str(out/'largest_component_rgb_overlay.png'),largest_overlay)
 prompt_groups=[]
 for _ in range(1):
  positives=distributed_positive_points(lab,members)
  x,y,w,h_box=(int(value) for value in stats[largest_label,:4]); x2=x+w-1; y2=y+h_box-1
  pad=45; box=[max(0,x-pad),max(0,y-pad),min(rgb.shape[1]-1,x2+pad),min(rgb.shape[0]-1,y2+pad)]
  # Constrain SAM2 with table pixels to the left of the seeded workpiece.  This
  # uses geometry only, so it remains valid on dark cloth or dark metal.
  bx0,by0,bx1,by1=box; in_box=np.zeros_like(valid,dtype=bool); in_box[by0:by1+1,bx0:bx1+1]=True
  left_of_seed=np.zeros_like(valid,dtype=bool); left_of_seed[:,:max(0,x-8)]=True
  # The left strip is separated from the positive seed and constrained to the
  # tabletop ROI, so valid depth is sufficient without colour/height tests.
  table_negative=in_box & left_of_seed & roi.astype(bool) & valid
  negatives=distributed_binary_points(table_negative, count=6)
  if not negatives:
   # Candidates touching the table's left edge have no local left strip.  Fall
   # back to seed-free pixels on the left half of the calibrated tabletop.
   left_table=(pixel_x < rgb.shape[1] * .45) & roi.astype(bool) & valid & ~seeds.astype(bool)
   negatives=distributed_binary_points(left_table, count=6)
  cv2.rectangle(prompt_overlay,(box[0],box[1]),(box[2],box[3]),(0,255,255),2)
  for px,py in positives: cv2.circle(prompt_overlay,(px,py),5,(0,255,0),-1)
  for px,py in negatives: cv2.circle(prompt_overlay,(px,py),5,(0,0,255),-1)
  # Keep only SAM2 pixels with a valid depth measurement inside the calibrated
  # four-corner tabletop ROI.  No RGB, height, or component post-processing.
  mask=prompt_segment(rgb,box,positives,negatives,a.socket).astype(np.uint8)
  mask=(mask & valid.astype(np.uint8) & roi.astype(np.uint8)).astype(np.uint8)
  if np.count_nonzero(mask)<350: continue
  # Buffer Check visual contract: masks are unambiguously pure red in BGR.
  overlay[mask>0]=(0,0,255); masks.append(mask); index+=1
  prompt_groups.append({'seed_labels':members,'positives_xy':positives,'negatives_xy':negatives,'box_xyxy':box,'auto_negatives':True,'sam_mask_pixels':int(np.count_nonzero(mask))})
 cv2.imwrite(str(out/'sam2_prompt_overlay.png'),prompt_overlay); cv2.imwrite(str(out/'sam2_masks_overlay.png'),overlay); np.savez_compressed(out/'masks.npz',masks=np.asarray(masks,dtype=np.uint8)); (out/'meta.json').write_text(json.dumps({'masks':index,'plane':plane.tolist(),'prompt_groups':prompt_groups},indent=2)); print(out)
if __name__=='__main__': main()
