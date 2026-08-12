#!/usr/bin/env python3
"""Desktop-grasp end-to-end entry point; perception by default, motion only with --execute."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from capture_head_rgbd import capture, DEFAULT_HEAD_SERIAL, DEFAULT_OUTPUT_DIR

ROOT=Path(__file__).resolve().parent
MODULE_ROOT=ROOT.parent
PROJECT_ROOT=MODULE_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
# Use the caller's environment, rather than a workstation-specific venv path.
PYTHON=Path(sys.executable)
def run(script,*args): subprocess.run([str(PYTHON),str(ROOT/script),*map(str,args)],check=True)
def ensure_sam2():
 from modules.module_0_prepare.ensure_sam2_service import require_sam2_service
 require_sam2_service()
def move_head_to_calibrated_capture_pose(body_head,head_host,settle_s):
 calibration=json.loads(body_head.read_text())
 pose=calibration.get('head_pose_deg') or {}
 pitch=float(pose['pitch']); yaw=float(pose['yaw'])
 from parm_sdk import Head
 head=Head(host=head_host)
 try:
  print(f'Head capture pose: pitch={pitch:.1f}deg, yaw={yaw:.1f}deg')
  head.move_to(pitch=pitch,yaw=yaw)
  deadline=time.monotonic()+max(5.,float(settle_s)+4.)
  while time.monotonic()<deadline:
   state=head.get_state()
   try:
    actual_pitch=-float(state['pitch']['position']); actual_yaw=float(state['yaw']['position'])
   except (KeyError,TypeError,ValueError):
    time.sleep(.05); continue
   if abs(actual_pitch-pitch)<=.5 and abs(actual_yaw-yaw)<=.5:
    if settle_s: time.sleep(float(settle_s))
    print(f'Head capture pose reached: pitch={actual_pitch:.2f}deg, yaw={actual_yaw:.2f}deg')
    return
   time.sleep(.05)
  raise RuntimeError(f'head did not reach calibrated capture pose pitch={pitch:.1f}deg, yaw={yaw:.1f}deg')
 finally: head.close()
def run_perception(d, stl, orientation_config):
 from modules.module_0_prepare.ensure_sam2_service import sam2_url
 run('sam2_tabletop_masks.py','--capture-dir',d,'--socket',sam2_url())
 run('sam2_icp_grasp.py','--capture-dir',d,'--stl',stl,'--orientation-config',orientation_config)
 run('visualize_grasp_3d.py','--capture-dir',d)
def move_left_ready_after_capture(d,planner_host,arm_host,speed):
 """Execute the configured Planner RRTStar joint move and wait for arm feedback."""
 sys.path[:0]=[str(PROJECT_ROOT),str(PROJECT_ROOT/'tool')]
 from planner_runtime import PlannerRuntime
 from loop_config import LoopConfig
 from parm_sdk import Arm
 from modules.reverse_grasp_handoff_positions import load_workpiece_handoff_profile
 ready=load_workpiece_handoff_profile('1')['left_reverse_grasp']; target=list(ready['ready_joints_deg'])
 runtime=PlannerRuntime(LoopConfig(planner_host=planner_host,execute_url='http://127.0.0.1:5667/middle/waypoint/execute',base_frame='body_base',left_group='left_arm',right_group='right_arm',left_tool_frame='left_eef',right_tool_frame='right_hand_transfer_end',max_velocity=speed,max_acceleration=speed,idle_poll_seconds=.2,live=True,poses={}))
 event={'motion':'planner_rrtstar','target_joint_angles_deg':target,'started_at':time.time(),'planner_host':planner_host}
 try:
  runtime.open(required_frames={'body_base','left_eef'}); print('Post-capture Planner RRTStar target (left deg):',target); runtime.move_left_joints(target,planner='rrtstar')
  deadline=time.monotonic()+float(ready['arrival_timeout_s']); tolerance=float(ready['arrival_tolerance_deg']); last=None
  with Arm(host=arm_host) as arm:
   while time.monotonic()<deadline:
    state=arm.get_state() or {}; actual=((state.get('state') or {}).get('left') or {}).get('position') or []
    if len(actual)==7:
     delta=(np.asarray(actual,float)-target+180.)%360.-180.; last=float(np.max(np.abs(delta)))
     if last<=tolerance: event.update({'status':'reached','max_joint_error_deg':last}); print(f'Post-capture move_j reached; max joint error={last:.3f}deg'); return event
    time.sleep(.05)
  raise RuntimeError(f'post-capture move_j did not arrive within {ready["arrival_timeout_s"]}s; max joint error={last}')
 except Exception as exc:
  event.update({'status':'failed','error':str(exc)}); raise
 finally:
  event['finished_at']=time.time(); (d/'motion_after_capture.json').write_text(json.dumps(event,indent=2)); runtime.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture-dir',type=Path,help='reuse a prior RGB-D capture');p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true',help='plan all enabled approach candidates but never execute');p.add_argument('--body-head',type=Path,default=MODULE_ROOT/'config/body_head_calibration.json');p.add_argument('--stl',type=Path,default=PROJECT_ROOT/'tool/welding_up_side_1118_only_2_small_screws_removed.stl');p.add_argument('--orientation-config',type=Path,default=MODULE_ROOT/'config/cad_orientation.json');p.add_argument('--planner-host',default='127.0.0.1');p.add_argument('--arm-host',default='192.168.88.1');p.add_argument('--head-host',default='127.0.0.1');p.add_argument('--head-settle-s',type=float,default=1.0);p.add_argument('--speed',type=float,default=.20);p.add_argument('--approach-angles-deg',nargs='+',type=float,default=[90.,15.,-15.],help='table-relative approach angles; 90=vertical down; default: 90 15 -15 (up to +/-75deg side tilt)');p.add_argument('--body-z-down-m',type=float,default=.005,help='additional body_base -Z bias for both pregrasp and grasp (default: 0.005 m)');a=p.parse_args()
 if a.execute and a.preflight_only: p.error('--execute and --preflight-only are mutually exclusive')
 ensure_sam2()
 fresh_capture=not bool(a.capture_dir)
 if a.capture_dir: d=a.capture_dir
 else:
  move_head_to_calibrated_capture_pose(a.body_head,a.head_host,a.head_settle_s)
  d=capture(SimpleNamespace(serial=DEFAULT_HEAD_SERIAL,output_dir=DEFAULT_OUTPUT_DIR,frames=1,width=1280,height=720,fps=30,warmup_frames=30,timeout_ms=5000,depth_fuse_frames=15,preview_min_m=.20,preview_max_m=2.50))
 # Finish perception from the fixed RGB-D capture before any arm motion.
 # This guarantees that SAM2/ICP never runs while the left arm can occlude the
 # tabletop, and avoids move_j when no unique reverse workpiece was found.
 run_perception(d,a.stl,a.orientation_config)
 records=json.loads((d/'sam2_masks/grasp_poses.json').read_text())
 if a.execute and fresh_capture:
  if len(records)==1: move_left_ready_after_capture(d,a.planner_host,a.arm_host,a.speed)
  else: print(f'Post-capture move_j skipped: expected one reverse grasp candidate, got {len(records)}.')
 print(f'2D: {d}/sam2_masks/grasp_overlay.png\n3D: {d}/sam2_masks/grasp_3d.png')
 if not a.execute and not a.preflight_only: print('Perception complete. Inspect visualisations; add --execute only after approval.'); return
 if len(records)!=1:
  mode='Preflight' if a.preflight_only else 'Execution'
  print(f'{mode} not run: expected exactly one reverse-side grasp candidate, got {len(records)}. No robot, gripper, or Planner trajectory command was sent.')
  if a.execute or a.preflight_only: raise RuntimeError(f'expected exactly one reverse-side grasp candidate, got {len(records)}')
  return
 mode='--execute' if a.execute else '--preflight-only'
 run('execute_confirmed_grasp.py','--grasp-json',d/'sam2_masks/grasp_poses.json','--body-head',a.body_head,'--planner-host',a.planner_host,'--arm-host',a.arm_host,'--speed',a.speed,'--approach-angles-deg',*a.approach_angles_deg,'--body-z-down-m',a.body_z_down_m,mode)
if __name__=='__main__':main()
