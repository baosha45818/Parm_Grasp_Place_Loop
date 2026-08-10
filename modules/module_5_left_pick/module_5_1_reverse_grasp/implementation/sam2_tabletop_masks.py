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

def box_gap(a, b):
    """Euclidean gap between two xywh boxes (zero means overlap)."""
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    dx=max(ax-(bx+bw), bx-(ax+aw), 0); dy=max(ay-(by+bh), by-(ay+ah), 0)
    return float(np.hypot(dx,dy))

def dark_line_support(gray, first, second):
    """Evidence that two raised seed fragments are joined by dark metal."""
    a=np.asarray(first,float); b=np.asarray(second,float); steps=max(8,int(np.linalg.norm(b-a)))
    xy=np.rint(np.linspace(a,b,steps)).astype(int)
    xy[:,0]=np.clip(xy[:,0],0,gray.shape[1]-1); xy[:,1]=np.clip(xy[:,1],0,gray.shape[0]-1)
    return float(np.mean(gray[xy[:,1],xy[:,0]]<=150))

def merge_seed_fragments(stats, centers, gray, max_gap_px=85.0):
    """Join seed fragments only when a dark-metal bridge supports the merge."""
    parent=list(range(len(stats)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def join(i,j):
        i,j=find(i),find(j)
        if i!=j: parent[j]=i
    for i in range(len(stats)):
        for j in range(i+1,len(stats)):
            if box_gap(stats[i],stats[j])<=max_gap_px and dark_line_support(gray,centers[i],centers[j])>=.45:
                join(i,j)
    groups={}
    for i in range(len(stats)): groups.setdefault(find(i),[]).append(i)
    return list(groups.values())

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

def retain_supported_workpiece_components(mask, valid, gray, height, positives):
    """Remove SAM tabletop spill using current-frame RGB-D support.

    The actual metal part can appear as several dark components.  Preserve
    only supported components near one of the positive workpiece prompts,
    rather than retaining the complete connected SAM region through cloth.
    """
    support = (mask.astype(bool) & valid & (gray <= 110) & (height >= .001)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    kept = np.zeros_like(support)
    positive = np.asarray(positives, dtype=float)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        yy, xx = np.nonzero(labels == label)
        pixels = np.column_stack((xx, yy)).astype(float)
        nearest = np.min(np.sum((pixels[:, None, :] - positive[None, :, :]) ** 2, axis=2)) if len(positive) else np.inf
        if nearest <= 25.0 ** 2:
            kept[labels == label] = 1
    # Do not silently erase a valid workpiece if depth is exceptionally sparse.
    return kept if int(np.count_nonzero(kept)) >= 350 else mask.astype(np.uint8)

def main():
 workspace=json.loads((MODULE_ROOT/'config/tabletop_workspace.json').read_text())
 p=argparse.ArgumentParser(); p.add_argument('--capture-dir',type=Path,required=True); p.add_argument('--socket',default=os.environ.get('PARM_SAM2_URL','unix:///home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/sam2-service/runtime/sam2-service.sock')); p.add_argument('--height-mm',type=float,default=3.0); p.add_argument('--max-gray',type=int,default=105); p.add_argument('--min-seed-area',type=int,default=90); p.add_argument('--roi',nargs=4,type=int,default=workspace['safe_roi_xywh'],metavar=('X','Y','W','H')); p.add_argument('--out',type=Path,default=None); a=p.parse_args()
 meta=json.loads((a.capture_dir/'metadata.json').read_text()); rgb=cv2.imread(str(a.capture_dir/'rgb_000.png')); d=cv2.imread(str(a.capture_dir/'depth_aligned_000.png'),cv2.IMREAD_UNCHANGED)
 k=meta['color_intrinsics']; K=np.array([[k['fx'],0,k['ppx']],[0,k['fy'],k['ppy']],[0,0,1.]])
 xyz=depth_to_xyz(d,K,float(meta['depth_scale_m'])); valid=(d>0)&(d<65535); pts=xyz[valid]
 import open3d as o3d
 cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)); plane,_=cloud.voxel_down_sample(.003).segment_plane(.004,3,1200); plane=np.asarray(plane); n=plane[:3]/np.linalg.norm(plane[:3]); c=pts.mean(0)
 if np.dot(n,-c)<0: plane=-plane
 h=(xyz@plane[:3]+plane[3]); gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
 # Depth supplies conservative foreground hints. Dark-metal RGB support
 # suppresses tabletop height ripple before SAM2 grows a complete object.
 seeds=((h>=a.height_mm/1000)&valid&(gray<=a.max_gray)).astype(np.uint8)
 roi=np.zeros_like(seeds); rx,ry,rw,rh=a.roi; roi[max(0,ry):min(seeds.shape[0],ry+rh),max(0,rx):min(seeds.shape[1],rx+rw)]=1; seeds*=roi
 seeds=cv2.morphologyEx(seeds,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); seeds=cv2.dilate(seeds,np.ones((5,5),np.uint8),iterations=1)
 count,lab,stats,centers=cv2.connectedComponentsWithStats(seeds)
 out=a.out or a.capture_dir/'sam2_masks'; out.mkdir(parents=True,exist_ok=True); overlay=rgb.copy(); index=0; masks=[]
 kept=[label for label in range(1,count) if stats[label,cv2.CC_STAT_AREA]>=a.min_seed_area]
 seed_stats=[tuple(map(int,stats[label,:4])) for label in kept]; seed_centers=[tuple(map(int,np.rint(centers[label]))) for label in kept]
 prompt_groups=[]
 for group in merge_seed_fragments(seed_stats,seed_centers,gray):
  members=[kept[i] for i in group]; positives=distributed_positive_points(lab,members)
  boxes=[seed_stats[i] for i in group]; x=min(q[0] for q in boxes); y=min(q[1] for q in boxes); x2=max(q[0]+q[2]-1 for q in boxes); y2=max(q[1]+q[3]-1 for q in boxes)
  pad=45; box=[max(0,x-pad),max(0,y-pad),min(rgb.shape[1]-1,x2+pad),min(rgb.shape[0]-1,y2+pad)]
  # SAM can otherwise grow from the dark metal into the wrinkled tabletop.
  # Negative prompts are only sampled from pixels that are both bright and at
  # table height, so elevated/shiny workpiece surfaces are never negatives.
  bx0,by0,bx1,by1=box; in_box=np.zeros_like(valid,dtype=bool); in_box[by0:by1+1,bx0:bx1+1]=True
  table_negative=in_box & roi.astype(bool) & valid & (np.abs(h)<=.0015) & (gray>=125)
  negatives=distributed_binary_points(table_negative, count=6)
  mask=prompt_segment(rgb,box,positives,negatives,a.socket)
  # A SAM2 prompt is allowed to grow only inside the declared tabletop work
  # zone.  This prevents a metal robot link from becoming a workpiece mask.
  mask=(mask.astype(np.uint8)*roi)
  n,labels=cv2.connectedComponents(mask)
  components={int(labels[py,px]) for px,py in positives if 0<=py<labels.shape[0] and 0<=px<labels.shape[1] and labels[py,px]>0}
  if not components:
   overlap=[np.count_nonzero((labels==j)&np.isin(lab,members)) for j in range(1,n)]; components={1+int(np.argmax(overlap))} if overlap else set()
  mask=np.isin(labels,list(components)).astype(np.uint8) if components else np.zeros_like(mask)
  raw_mask_pixels=int(np.count_nonzero(mask))
  mask=retain_supported_workpiece_components(mask,valid,gray,h,positives)
  if np.count_nonzero(mask)<350: continue
  color=tuple(map(int,np.random.default_rng(index).integers(60,255,3))); overlay[mask>0]=(.55*overlay[mask>0]+.45*np.array(color)).astype(np.uint8); masks.append(mask); index+=1
  prompt_groups.append({'seed_labels':members,'positives_xy':positives,'negatives_xy':negatives,'box_xyxy':box,'auto_negatives':True,'sam_mask_pixels':raw_mask_pixels,'mask_pixels_after_rgbd_refine':int(np.count_nonzero(mask))})
 cv2.imwrite(str(out/'sam2_masks_overlay.png'),overlay); np.savez_compressed(out/'masks.npz',masks=np.asarray(masks,dtype=np.uint8)); (out/'meta.json').write_text(json.dumps({'masks':index,'plane':plane.tolist(),'prompt_groups':prompt_groups},indent=2)); print(out)
if __name__=='__main__': main()
