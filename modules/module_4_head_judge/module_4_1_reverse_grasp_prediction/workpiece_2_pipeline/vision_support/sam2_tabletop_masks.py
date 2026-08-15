#!/usr/bin/env python3
"""Generate one SAM2 mask per elevated tabletop component from an RGB-D capture."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[5]
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

def distributed_positive_points(labels, members, count=3, interior_margin_px=12,
                                core_distance_fraction=.70, min_separation_px=18):
    """Choose a few well-separated prompts only from the component's deep core.

    SAM2 does not need positive points spread across every narrow tab.  Those
    points are fragile because the depth seed can be thin near a metallic edge.
    Use only pixels whose boundary distance is close to the component maximum,
    and return fewer points rather than falling back to edge pixels.
    """
    component=np.isin(labels,members).astype(np.uint8)
    distance=cv2.distanceTransform(component,cv2.DIST_L2,5)
    max_distance=float(distance.max())
    if max_distance <= 0:
        return []
    core_distance=max(float(interior_margin_px), max_distance*float(core_distance_fraction))
    yy,xx=np.nonzero((component.astype(bool)) & (distance>=core_distance))
    points=np.column_stack((xx,yy))
    if len(points)==0:
        y,x=np.unravel_index(int(np.argmax(distance)),distance.shape)
        return [[int(x),int(y)]]
    point_distances=distance[yy,xx]
    chosen=[int(np.argmax(point_distances))]
    while len(chosen)<min(int(count),len(points)):
        nearest=np.min(np.sum((points[:,None,:]-points[chosen][None,:,:])**2,axis=2),axis=1)
        available=np.flatnonzero(nearest>=float(min_separation_px)**2)
        if len(available)==0:
            break
        index=int(available[np.argmax(nearest[available])])
        chosen.append(index)
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


def tabletop_roi_mask(workspace, shape_hw, roi_xywh=None):
    """Build the configured tabletop quadrilateral at the current RGB size.

    A rectangular ``--roi`` remains available as a deliberate debugging
    override, but production calls use the calibrated tabletop polygon.  The
    same mask gates seed generation, SAM2 output, and negative prompts.
    """
    height, width = shape_hw
    if roi_xywh is not None:
        x, y, roi_width, roi_height = roi_xywh
        polygon = np.asarray(((x, y), (x + roi_width - 1, y),
                              (x + roi_width - 1, y + roi_height - 1),
                              (x, y + roi_height - 1)), dtype=np.int32)
        source = "cli_rectangle"
    else:
        try:
            reference_width, reference_height = workspace["reference_color_size_wh"]
            reference_polygon = np.asarray(workspace["safe_roi_polygon_px"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("tabletop_workspace.json 缺少有效的 reference_color_size_wh 或 safe_roi_polygon_px") from exc
        if reference_polygon.ndim != 2 or reference_polygon.shape[0] < 3 or reference_polygon.shape[1] != 2 or reference_width <= 0 or reference_height <= 0:
            raise ValueError("tabletop_workspace.json 的桌面 ROI 必须至少包含三个有效顶点")
        scale = np.asarray((width / float(reference_width), height / float(reference_height)))
        polygon = np.rint(reference_polygon * scale).astype(np.int32)
        source = "workspace_polygon"
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1, lineType=cv2.LINE_8)
    return mask, polygon, source


def save_prompt_visualization(image, roi_polygon, box_xyxy, positives, negatives, path, candidate_index):
    """Save the exact positive/negative prompts passed to one SAM2 request."""
    overlay = image.copy()
    cv2.polylines(overlay, [roi_polygon], True, (255, 255, 0), 2, cv2.LINE_AA)
    bx0, by0, bx1, by1 = box_xyxy
    cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 255, 255), 2)
    for point_index, (x, y) in enumerate(positives, start=1):
        cv2.circle(overlay, (x, y), 8, (0, 0, 0), -1)
        cv2.circle(overlay, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(overlay, f"+{point_index}", (x + 9, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 2, cv2.LINE_AA)
    for point_index, (x, y) in enumerate(negatives, start=1):
        cv2.drawMarker(overlay, (x, y), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 4, cv2.LINE_AA)
        cv2.drawMarker(overlay, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
        cv2.putText(overlay, f"-{point_index}", (x + 9, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 2, cv2.LINE_AA)
    caption = f"SAM2 prompt {candidate_index}: green=positive, red=negative, yellow=box, cyan=table ROI"
    cv2.putText(overlay, caption, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, caption, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"无法写入 SAM2 提示可视化图: {path}")

def main():
 workspace=json.loads((MODULE_ROOT/'config/tabletop_workspace.json').read_text())
 p=argparse.ArgumentParser(); p.add_argument('--capture-dir',type=Path,required=True); p.add_argument('--socket',default='unix:///tmp/sam2-service.sock'); p.add_argument('--height-mm',type=float,default=3.0); p.add_argument('--max-gray',type=int,default=105); p.add_argument('--min-seed-area',type=int,default=90); p.add_argument('--positive-interior-margin-px',type=int,default=12); p.add_argument('--positive-core-distance-fraction',type=float,default=.70); p.add_argument('--positive-min-separation-px',type=int,default=18); p.add_argument('--roi',nargs=4,type=int,default=None,metavar=('X','Y','W','H'),help='debug-only rectangular override; production uses tabletop_workspace.json polygon'); p.add_argument('--out',type=Path,default=None); a=p.parse_args()
 if a.positive_interior_margin_px < 0:
  raise ValueError('--positive-interior-margin-px 必须 >= 0')
 if not 0 < a.positive_core_distance_fraction <= 1:
  raise ValueError('--positive-core-distance-fraction 必须在 (0,1]')
 if a.positive_min_separation_px < 0:
  raise ValueError('--positive-min-separation-px 必须 >= 0')
 meta=json.loads((a.capture_dir/'metadata.json').read_text()); rgb=cv2.imread(str(a.capture_dir/'rgb_000.png')); d=cv2.imread(str(a.capture_dir/'depth_aligned_000.png'),cv2.IMREAD_UNCHANGED)
 if rgb is None or d is None:
  raise RuntimeError("无法读取头部 RGB 或对齐深度图")
 actual_size=[int(rgb.shape[1]),int(rgb.shape[0])]
 k=meta['color_intrinsics']; K=np.array([[k['fx'],0,k['ppx']],[0,k['fy'],k['ppy']],[0,0,1.]])
 xyz=depth_to_xyz(d,K,float(meta['depth_scale_m'])); valid=(d>0)&(d<65535); pts=xyz[valid]
 import open3d as o3d
 cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)); plane,_=cloud.voxel_down_sample(.003).segment_plane(.004,3,1200); plane=np.asarray(plane); n=plane[:3]/np.linalg.norm(plane[:3]); c=pts.mean(0)
 if np.dot(n,-c)<0: plane=-plane
 h=(xyz@plane[:3]+plane[3]); gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
 # Depth supplies conservative foreground hints. Dark-metal RGB support
 # suppresses tabletop height ripple before SAM2 grows a complete object.
 seeds=((h>=a.height_mm/1000)&valid&(gray<=a.max_gray)).astype(np.uint8)
 roi, roi_polygon, roi_source = tabletop_roi_mask(workspace, seeds.shape, a.roi); seeds*=roi
 seeds=cv2.morphologyEx(seeds,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); seeds=cv2.dilate(seeds,np.ones((5,5),np.uint8),iterations=1)
 count,lab,stats,centers=cv2.connectedComponentsWithStats(seeds)
 out=a.out or a.capture_dir/'sam2_masks'; out.mkdir(parents=True,exist_ok=True); overlay=rgb.copy(); index=0; masks=[]
 prompt_dir=out/'prompt_visualizations'; prompt_dir.mkdir(exist_ok=True); prompt_visualizations=[]
 kept=[label for label in range(1,count) if stats[label,cv2.CC_STAT_AREA]>=a.min_seed_area]
 seed_overlay=rgb.copy()
 seed_overlay[seeds>0]=(0,0,255)
 cv2.imwrite(str(out/'seed_foreground_overlay.png'),seed_overlay)
 seed_stats=[tuple(map(int,stats[label,:4])) for label in kept]; seed_centers=[tuple(map(int,np.rint(centers[label]))) for label in kept]
 prompt_groups=[]
 for candidate_index, group in enumerate(merge_seed_fragments(seed_stats,seed_centers,gray)):
  members=[kept[i] for i in group]; positives=distributed_positive_points(lab,members,interior_margin_px=a.positive_interior_margin_px,core_distance_fraction=a.positive_core_distance_fraction,min_separation_px=a.positive_min_separation_px)
  boxes=[seed_stats[i] for i in group]; x=min(q[0] for q in boxes); y=min(q[1] for q in boxes); x2=max(q[0]+q[2]-1 for q in boxes); y2=max(q[1]+q[3]-1 for q in boxes)
  pad=45; box=[max(0,x-pad),max(0,y-pad),min(rgb.shape[1]-1,x2+pad),min(rgb.shape[0]-1,y2+pad)]
  # SAM can otherwise grow from the dark metal into the wrinkled tabletop.
  # Negative prompts are only sampled from pixels that are both bright and at
  # table height, so elevated/shiny workpiece surfaces are never negatives.
  bx0,by0,bx1,by1=box; in_box=np.zeros_like(valid,dtype=bool); in_box[by0:by1+1,bx0:bx1+1]=True
  seed_footprint=cv2.dilate(np.isin(lab,members).astype(np.uint8),np.ones((21,21),np.uint8)).astype(bool)
  # Reflective metal can have tabletop-like depth and be absent from the seed
  # itself.  Exclude the full seed bounding envelope, not only seed pixels.
  envelope=np.zeros_like(valid,dtype=bool)
  envelope_margin=30
  envelope[max(0,y-envelope_margin):min(valid.shape[0],y2+1+envelope_margin),max(0,x-envelope_margin):min(valid.shape[1],x2+1+envelope_margin)]=True
  table_negative=in_box & roi.astype(bool) & valid & (np.abs(h)<=.0015) & (gray>=125) & ~seed_footprint & ~envelope
  negatives=distributed_binary_points(table_negative, count=6)
  prompt_path=prompt_dir/f'sam2_prompt_{candidate_index:03d}.png'
  save_prompt_visualization(rgb, roi_polygon, box, positives, negatives, prompt_path, candidate_index)
  prompt_visualizations.append(str(prompt_path))
  mask=prompt_segment(rgb,box,positives,negatives,a.socket)
  # A SAM2 prompt is allowed to grow only inside the declared tabletop work
  # zone.  This prevents a metal robot link from becoming a workpiece mask.
  mask=(mask.astype(np.uint8)*roi)
  n,labels=cv2.connectedComponents(mask)
  components={int(labels[py,px]) for px,py in positives if 0<=py<labels.shape[0] and 0<=px<labels.shape[1] and labels[py,px]>0}
  if not components:
   overlap=[np.count_nonzero((labels==j)&np.isin(lab,members)) for j in range(1,n)]; components={1+int(np.argmax(overlap))} if overlap else set()
  first_mask=np.isin(labels,list(components)).astype(np.uint8) if components else np.zeros_like(mask)
  # The first pass only turns conservative height/RGB seeds into a full SAM
  # silhouette.  Select final positive prompts from that silhouette's core,
  # then send a second SAM2 request so prompts cannot land on seed fragments.
  core_positives=distributed_positive_points(
      first_mask, [1], interior_margin_px=a.positive_interior_margin_px,
      core_distance_fraction=a.positive_core_distance_fraction,
      min_separation_px=a.positive_min_separation_px,
  )
  if not core_positives:
   continue
  refine_prompt_path=prompt_dir/f'sam2_refine_prompt_{candidate_index:03d}.png'
  first_mask_footprint=cv2.dilate(first_mask.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)
  refine_negative=distributed_binary_points(table_negative & ~first_mask_footprint, count=6)
  save_prompt_visualization(rgb, roi_polygon, box, core_positives, refine_negative, refine_prompt_path, candidate_index)
  prompt_visualizations.append(str(refine_prompt_path))
  refined_mask=prompt_segment(rgb,box,core_positives,refine_negative,a.socket)
  refined_mask=(refined_mask.astype(np.uint8)*roi)
  refined_count,refined_labels=cv2.connectedComponents(refined_mask)
  refined_components={int(refined_labels[py,px]) for px,py in core_positives if 0<=py<refined_labels.shape[0] and 0<=px<refined_labels.shape[1] and refined_labels[py,px]>0}
  if not refined_components:
   overlap=[np.count_nonzero((refined_labels==j)&first_mask.astype(bool)) for j in range(1,refined_count)]
   refined_components={1+int(np.argmax(overlap))} if overlap else set()
  mask=np.isin(refined_labels,list(refined_components)).astype(np.uint8) if refined_components else np.zeros_like(refined_mask)
  raw_mask_pixels=int(np.count_nonzero(mask))
  mask=retain_supported_workpiece_components(mask,valid,gray,h,core_positives)
  if np.count_nonzero(mask)<350: continue
  color=tuple(map(int,np.random.default_rng(index).integers(60,255,3))); overlay[mask>0]=(.55*overlay[mask>0]+.45*np.array(color)).astype(np.uint8); masks.append(mask); index+=1
  prompt_groups.append({'seed_labels':members,'initial_positives_xy':positives,'core_positives_xy':core_positives,'positive_interior_margin_px':a.positive_interior_margin_px,'positive_core_distance_fraction':a.positive_core_distance_fraction,'positive_min_separation_px':a.positive_min_separation_px,'initial_negatives_xy':negatives,'refine_negatives_xy':refine_negative,'box_xyxy':box,'auto_negatives':True,'initial_prompt_visualization':str(prompt_path),'core_prompt_visualization':str(refine_prompt_path),'first_sam_mask_pixels':int(np.count_nonzero(first_mask)),'sam_mask_pixels':raw_mask_pixels,'mask_pixels_after_rgbd_refine':int(np.count_nonzero(mask))})
 cv2.imwrite(str(out/'sam2_masks_overlay.png'),overlay); np.savez_compressed(out/'masks.npz',masks=np.asarray(masks,dtype=np.uint8)); (out/'meta.json').write_text(json.dumps({'masks':index,'plane':plane.tolist(),'prompt_groups':prompt_groups,'prompt_visualizations':prompt_visualizations,'image_size_wh':actual_size,'roi_polygon_px':roi_polygon.tolist(),'roi_source':roi_source,'seed_pixels':int(np.count_nonzero(seeds)),'seed_components_above_min_area':len(kept)},indent=2)); print(out); print(f'SAM2_PROMPT_VISUALIZATIONS={prompt_dir}')
 if not masks:
  raise RuntimeError(
   "SAM2 未生成任何工件掩码；请查看 "
   f"{out/'seed_foreground_overlay.png'} 与 {out/'sam2_masks_overlay.png'}。"
  )
if __name__=='__main__': main()
