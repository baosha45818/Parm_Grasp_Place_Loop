#!/usr/bin/env python3
"""Render RGB-D scene points and a simple two-finger gripper at queued grasp poses."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def box(ax,c,R,size,color):
 s=np.array(size)/2; v=np.array([[x,y,z] for x in(-s[0],s[0]) for y in(-s[1],s[1]) for z in(-s[2],s[2])]); v=v@R.T+c
 faces=[[v[i] for i in q] for q in ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))]
 ax.add_collection3d(Poly3DCollection(faces,facecolors=color,edgecolors='k',linewidths=.3,alpha=.75))
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture-dir',type=Path,required=True);a=p.parse_args(); d=a.capture_dir; m=json.loads((d/'metadata.json').read_text()); poses=json.loads((d/'sam2_masks/grasp_poses.json').read_text()); dep=cv2.imread(str(d/'depth_aligned_000.png'),cv2.IMREAD_UNCHANGED); rgb=cv2.cvtColor(cv2.imread(str(d/'rgb_000.png')),cv2.COLOR_BGR2RGB); k=m['color_intrinsics']; yy,xx=np.indices(dep.shape); z=dep*float(m['depth_scale_m']); ok=(z>.35)&(z<1.2); x=(xx-k['ppx'])*z/k['fx']; y=(yy-k['ppy'])*z/k['fy']; pts=np.c_[x[ok],y[ok],z[ok]]; col=rgb[ok]/255.; take=np.random.default_rng(0).choice(len(pts),min(30000,len(pts)),False)
 fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(111,projection='3d');ax.scatter(pts[take,0],pts[take,1],pts[take,2],c=col[take],s=.25)
 for r in poses:
  R=np.array(r['grasp_pose']['rotation_matrix']); c=np.array(r['grasp_pose']['translation_m']); # Version7: +X approaches down, +Y is jaw opening
  box(ax,c-R[:,0]*.025,R,(.025,.075,.035),'#444444'); box(ax,c+R[:,1]*.028-R[:,0]*.015,R,(.055,.018,.018),'#d8a300'); box(ax,c-R[:,1]*.028-R[:,0]*.015,R,(.055,.018,.018),'#d8a300')
  for i,co in enumerate(('r','g','b')): ax.quiver(*c,*R[:,i],length=.06,color=co,arrow_length_ratio=.15)
 ax.set_xlabel('camera X (m)');ax.set_ylabel('camera Y (m)');ax.set_zlabel('depth Z (m)');ax.view_init(28,-72);ax.set_box_aspect((1,1,.55));plt.tight_layout();out=d/'sam2_masks/grasp_3d.png';plt.savefig(out,dpi=180);print(out)
if __name__=='__main__':main()
