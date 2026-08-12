#!/usr/bin/env python3
"""Automatically execute a reviewed grasp after all Planner checks pass.

Default is dry-run.  --execute is required for any planner motion.
"""
from __future__ import annotations
import argparse,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4]; MODULE_ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(MODULE_ROOT));sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tool'))
from planner_runtime import PlannerRuntime
from loop_config import LoopConfig
from loop_types import ObjectPose
from modules.module_0_prepare.gripper_state_compat import is_gripper_usable
from modules.reverse_grasp_handoff_positions import load_workpiece_handoff_profile
from parm_sdk.api.exceptions import ParmAPIError
from execution_timing import ExecutionTiming

NO_GRASP_EXIT_CODE=20
REVERSE_GRASP_CONFIG=load_workpiece_handoff_profile('1')['left_reverse_grasp']
POST_CLOSE_LIFT_M=0.03
# 反抓到工件后，新左夹爪必须保持至少 35% 的开度；接近全闭合不视为可靠持件。
LEFT_GRIPPER_HELD_POSITION_MIN=35.0

def pose(T):
 q=Rotation.from_matrix(T[:3,:3]).as_quat() # x,y,z,w
 return ObjectPose(frame_id='body_base',position=T[:3,3].tolist(),orientation=q.tolist())
def pose_debug(T):
 r=Rotation.from_matrix(T[:3,:3]); return {'position_m':np.asarray(T[:3,3]).round(7).tolist(),'rpy_xyz_deg':r.as_euler('xyz',degrees=True).round(3).tolist(),'quaternion_xyzw':r.as_quat().round(8).tolist(),'rotation_matrix':np.asarray(T[:3,:3]).round(8).tolist()}
def targets_for_grasp(Tg, Tb, above_m, body_z_down_m):
 """Build direct ``left_eef`` Planner targets for one grasp orientation."""
 target=Tb@Tg
 Tg_above=Tg.copy(); Tg_above[:3,3]-=Tg[:3,0]*float(above_m)
 above=Tb@Tg_above
 # Keep the 3 cm pregrasp separation while applying the requested physical-Z bias.
 target[2,3]-=body_z_down_m; above[2,3]-=body_z_down_m
 return target, above, Tg_above

def tilt_from_table_angle_deg(angle_deg):
 """Map a signed table-relative approach angle to local jaw-axis tilt.

 90 deg is vertical down.  +/-60 and +/-30 deg are respectively 30 and
 60 degrees of tilt to the two sides about local +Y (jaw-opening axis).
 """
 angle=float(angle_deg)
 return np.copysign(90.0-abs(angle), angle)
def wait_for_left_joint_target(arm_host,target_deg,label,tolerance_deg=.5,timeout_s=90.):
 """The planner executor is asynchronous; only continue after live left-arm feedback arrives."""
 from parm_sdk import Arm
 deadline=time.monotonic()+float(timeout_s);last=float('inf');target=np.asarray(target_deg,float)
 with Arm(host=arm_host) as arm:
  while time.monotonic()<deadline:
   state=arm.get_state() or {};actual=np.asarray(((state.get('state') or {}).get('left') or {}).get('position') or [],float)
   if actual.shape==(7,):
    delta=(actual-target+180.)%360.-180.;last=float(np.max(np.abs(delta)))
    if last<=float(tolerance_deg): print(f'{label} reached; max joint error={last:.3f}deg');return
   time.sleep(.05)
 raise RuntimeError(f'{label} did not arrive within {timeout_s:.1f}s; max joint error={last:.3f}deg')
def main():
 p=argparse.ArgumentParser();p.add_argument('--grasp-json',type=Path,required=True);p.add_argument('--body-head',type=Path,default=MODULE_ROOT/'config/body_head_calibration.json',help='body_base <- head camera calibration');p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true',help='plan all enabled approach candidates but never execute');p.add_argument('--speed',type=float,default=.20);p.add_argument('--above-m',type=float,default=.03);p.add_argument('--approach-angles-deg',nargs='+',type=float,default=[90.,15.,-15.],help='table-relative approach angles; 90=vertical down; default: 90 15 -15 (up to +/-75deg side tilt)');p.add_argument('--body-z-down-m',type=float,default=.005,help='additional body_z_down_m');p.add_argument('--lift-m',type=float,default=float(REVERSE_GRASP_CONFIG['post_close_lift_m']),help='固定为 0.03m：闭夹爪后上提3cm');p.add_argument('--gripper-host',default='127.0.0.1');p.add_argument('--planner-host',default='127.0.0.1');p.add_argument('--arm-host',default='192.168.88.1');p.add_argument('--execute-url',default='http://127.0.0.1:5667/middle/waypoint/execute');p.add_argument('--timing-file',type=Path,default=None);a=p.parse_args()
 if a.execute and a.preflight_only: p.error('--execute and --preflight-only are mutually exclusive')
 if a.above_m<=0 or abs(a.lift_m-POST_CLOSE_LIFT_M)>1e-9 or a.body_z_down_m<0 or not a.approach_angles_deg or any(not 0<abs(x)<=90 for x in a.approach_angles_deg): p.error(f'--above-m 必须为正，--lift-m 必须固定为 {POST_CLOSE_LIFT_M:.2f}m；--body-z-down-m 必须非负；每个 --approach-angles-deg 必须在 [-90,90] 且非0')
 records=json.loads(a.grasp_json.read_text());
 if len(records) != 1:
  raise RuntimeError(f'execution blocked: expected exactly one reverse-side grasp candidate, got {len(records)}; inspect grasp_overlay.png')
 r=records[0]; Tg=np.eye(4);Tg[:3,:3]=r['grasp_pose']['rotation_matrix'];Tg[:3,3]=r['grasp_pose']['translation_m']
 Tb=np.asarray(json.loads(a.body_head.read_text())['T'],float)
 timing=ExecutionTiming();timing_path=a.timing_file or a.grasp_json.parent/'reverse_grasp_execution_timing.json';outcome='FAILURE'
 # ``left_eef`` is the URDF open-jaw grasp centre: target it directly and do
 # not apply the retired grasp-TCP calibration or Planner-tool bridge.
 live_planner=a.execute or a.preflight_only
 cfg=LoopConfig(planner_host=a.planner_host,execute_url=a.execute_url,base_frame='body_base',left_group='left_arm',right_group='right_arm',left_tool_frame='left_eef',right_tool_frame='right_hand_transfer_end',max_velocity=a.speed,max_acceleration=a.speed,idle_poll_seconds=.2,live=live_planner,poses={})
 planner=PlannerRuntime(cfg)
 try:
  planner.open(required_frames={'body_base','left_eef'})
  # Higher table-relative angle has priority: first 90 (top-down), then 60,
  # then 30.  For equal absolute angles, preserve the user-provided sign order.
  approach_angles=sorted(dict.fromkeys(a.approach_angles_deg),key=lambda x:-abs(x))
  candidate_groups=[]
  for approach_deg in approach_angles:
   tilt_deg=tilt_from_table_angle_deg(approach_deg)
   tilt=np.eye(4); tilt[:3,:3]=Rotation.from_rotvec([0,np.deg2rad(tilt_deg),0]).as_matrix()
   candidate_groups.append([(approach_deg,tilt_deg,0.0,Tg@tilt)])
  reachable=[]; preflight=[]
  def preflight_group(group):
   # Use one independent Planner client per angular group.  The three groups
   # can therefore be evaluated together without sharing an SDK connection.
   group_planner=None
   if live_planner:
    group_planner=PlannerRuntime(cfg); group_planner.open(required_frames={'body_base','left_eef'})
   result=[]
   try:
    for approach_deg,tilt_deg,degrees,Tg_candidate in group:
     target,above,Tg_above=targets_for_grasp(Tg_candidate,Tb,a.above_m,a.body_z_down_m)
     item={'approach_angle_from_table_deg':approach_deg,'side_approach_tilt_deg':tilt_deg,'left_eef_head_camera':pose_debug(Tg_candidate),'pregrasp_left_eef_head_camera':pose_debug(Tg_above),'left_eef_target_body_base':pose_debug(target),'above_target_body_base':pose_debug(above)}
     candidate_started=time.perf_counter()
     if not live_planner:
      item['reachable']=None;item['planning_duration_seconds']=time.perf_counter()-candidate_started;result.append((approach_deg,tilt_deg,degrees,target,above,Tg_candidate,None,item)); continue
     # The transit to pregrasp is collision-checked by Planner but need not be
     # a Cartesian straight line.  RRTstar provides obstacle-avoiding joint-space freedom.
     try: plan=group_planner.plan_cartesian_pose('left',pose(above),planner='rrtstar')
     except ParmAPIError as exc: item['reachable']=False; item['reason']=str(exc);item['planning_duration_seconds']=time.perf_counter()-candidate_started; result.append((approach_deg,tilt_deg,degrees,target,above,Tg_candidate,None,item)); continue
     # Checking only the pregrasp target can leave the robot parked above the
     # part when the final RRTstar descent is infeasible.  Validate the descent
     # from the exact joint endpoint that Planner chose for the pregrasp.
     pregrasp_joints=(plan.get('data') or {}).get('trajectory') or []
     if not pregrasp_joints or len(pregrasp_joints[-1]) != 7:
      item['reachable']=False; item['reason']='pregrasp planner returned no seven-joint endpoint for RRTstar-descent validation';item['planning_duration_seconds']=time.perf_counter()-candidate_started; result.append((approach_deg,tilt_deg,degrees,target,above,Tg_candidate,None,item)); continue
     item['pregrasp_joint_target_deg']=[float(v) for v in pregrasp_joints[-1]]
     try:
      group_planner.plan_cartesian_pose('left',pose(target),planner='rrtstar',start_joints_deg=item['pregrasp_joint_target_deg'])
     except (ParmAPIError, RuntimeError) as exc:
      item['reachable']=False; item['descent_reachable']=False; item['reason']=f'pregrasp is reachable, but RRTstar descent from its endpoint is not: {exc}';item['planning_duration_seconds']=time.perf_counter()-candidate_started; result.append((approach_deg,tilt_deg,degrees,target,above,Tg_candidate,None,item)); continue
     item['reachable']=True; item['descent_reachable']=True;item['planning_duration_seconds']=time.perf_counter()-candidate_started; result.append((approach_deg,tilt_deg,degrees,target,above,Tg_candidate,plan,item))
   finally:
    if group_planner is not None: group_planner.close()
   return result
  timing.start('planner_preflight_parallel')
  if live_planner:
   # 90, +60 and -60 start concurrently, one grasp orientation per angle.
   with ThreadPoolExecutor(max_workers=len(candidate_groups)) as pool: grouped_results=list(pool.map(preflight_group,candidate_groups))
  else: grouped_results=[preflight_group(group) for group in candidate_groups]
  for group_result in grouped_results:
   for approach_deg,tilt_deg,degrees,target,above,Tg_candidate,plan,item in group_result:
    preflight.append(item)
    if item['reachable']:
     reachable.append((approach_deg,tilt_deg,degrees,target,above,Tg_candidate,plan,item)); print(f'Preflight approach={approach_deg:.0f}deg tilt={tilt_deg:.0f}deg: reachable (not executed)')
    elif live_planner: print(f"Preflight approach={approach_deg:.0f}deg tilt={tilt_deg:.0f}deg: unreachable: {item['reason']}")
  timing.finish('planner_preflight_parallel')
  debug={'approach_preflight':preflight,'body_z_down_m':a.body_z_down_m,'planner_frame':'left_eef','grasp_target_mode':'direct_urdf_left_eef','grasp_frame_convention':r.get('grasp_frame_convention','legacy_unknown'),'pregrasp_planner_mode':'rrtstar','grasp_descent_planner_mode':'rrtstar','post_grasp_lift_planner_mode':'rrtstar'}
  debug_path=a.grasp_json.parent/'planner_pose_debug.json'; debug_path.write_text(json.dumps(debug,indent=2)); print(json.dumps(debug,indent=2)); print('Pose debug:',debug_path)
  if not reachable: raise RuntimeError('execution blocked: no configured approach angle has a reachable pregrasp plan; inspect planner_pose_debug.json')
  approach_deg,tilt_deg,degrees,target,above,Tg_selected,pregrasp_plan,_=reachable[0]
  if a.preflight_only: outcome='SUCCESS_PREFLIGHT';print('preflight complete; no trajectory was executed');return
  if not a.execute: outcome='SUCCESS_DRY_RUN';print('dry-run complete; live Planner preflight was not requested');return
  print(f'Selected table approach={approach_deg:.0f}deg (side tilt={tilt_deg:.0f}deg); above target (body_base):',above[:3,3].round(4).tolist())
  timing.start('pregrasp_motion')
  planner.execute_planned_result(pregrasp_plan)
  wait_for_left_joint_target(a.arm_host,(pregrasp_plan.get('data') or {}).get('trajectory',[])[-1],'left pregrasp')
  timing.finish('pregrasp_motion')
  print('At pregrasp; automatically descending with RRTstar.')
  timing.start('grasp_descent_plan_and_motion')
  grasp_plan=planner.plan_cartesian_pose('left',pose(target),planner='rrtstar')
  if grasp_plan is None: raise RuntimeError('grasp descent planner returned no trajectory')
  planner.execute_planned_result(grasp_plan)
  wait_for_left_joint_target(a.arm_host,(grasp_plan.get('data') or {}).get('trajectory',[])[-1],'left grasp pose')
  timing.finish('grasp_descent_plan_and_motion')
  from parm_sdk import Gripper
  print('At grasp pose; closing left gripper synchronously...')
  timing.start('gripper_close')
  with Gripper(host=a.gripper_host,status_port=7702) as gripper:
   state=gripper.get_state().get(1)
   if not is_gripper_usable(state): raise RuntimeError('left gripper hand=1 is unavailable or not initialized')
   response=gripper.move(1,0.0,force=100.0,speed=80.0,mode='sync',preempt=False,timeout=60.0)
  # 新 driver 的 sync 返回包已包含本次命令的权威终态；7702 PUB 帧可能仍是
  # 上一时刻的 executing，不能立刻用它否定已成功的同步响应。
  payload=response.get('data') or response
  device=(payload.get('devices') or {}).get('left_gripper') or {}
  device_state=str(device.get('device_state','')).strip().lower()
  if response.get('status') != 'success' or device_state not in {'arrived','gripping'}:
   raise RuntimeError(f'left gripper close sync response is invalid: {response}')
  timing.finish('gripper_close')
  # ``arrived`` only confirms that the close command completed.  The actual
  # hold decision is deliberately deferred until after the 3 cm lift and must
  # come from the subscribed gripper state: only ``gripping`` is a contact.
  print('Gripper close complete; automatically lifting along body_base +Z by 3 cm before checking gripper hold state.')
  timing.start('post_grasp_lift_and_hold_check')
  check_lift=target.copy();check_lift[2,3]+=POST_CLOSE_LIFT_M
  print('Check lift target (body_base):',check_lift[:3,3].round(4).tolist())
  lift_plan=planner.plan_cartesian_pose('left',pose(check_lift),planner='rrtstar')
  if lift_plan is None: raise RuntimeError('post-grasp lift planner returned no trajectory')
  planner.execute_planned_result(lift_plan)
  wait_for_left_joint_target(a.arm_host,(lift_plan.get('data') or {}).get('trajectory',[])[-1],'left 3cm post-grasp lift')
  with Gripper(host=a.gripper_host,status_port=7702) as gripper:
   hold_state=gripper.get_state().get(1) or {}
  motion_state=str(hold_state.get('motion_state','')).strip().lower()
  position=hold_state.get('position')
  try: position=float(position)
  except (TypeError,ValueError): position=None
  held=motion_state=='gripping' and position is not None and position>LEFT_GRIPPER_HELD_POSITION_MIN
  print(f'[反抓夹爪] 上提3cm后状态: motion_state={motion_state!r} position={position}; held={held} (要求 gripping 且 position > {LEFT_GRIPPER_HELD_POSITION_MIN:.1f}%)',flush=True)
  timing.finish('post_grasp_lift_and_hold_check')
  if not held:
   with Gripper(host=a.gripper_host,status_port=7702) as gripper:
    gripper.move(1,100.0,force=100.0,speed=80.0,mode='sync',preempt=False,timeout=60.0)
   print('NO_GRASP: 上提3cm后左夹爪未报告 gripping 或闭合位置不满足；已打开左夹爪，返回预备位后重拍。',flush=True)
   raise SystemExit(NO_GRASP_EXIT_CODE)
  print('grasp-and-lift complete (held confirmed by post-lift left gripper state)');outcome='SUCCESS'
 finally:
  planner.close();timing.write(timing_path,outcome);print(f'REVERSE_GRASP_EXECUTION_TIMING={timing_path}',flush=True)
if __name__=='__main__':main()
