"""总控统一停止链：先取消两臂 SDK 任务，再由 runner 结束子进程组。"""
from __future__ import annotations


def cancel_both_arms(arm_host: str, reason: str) -> None:
    print(f"[controller/stop] {reason}；发送左/右 arm.cancel_task()", flush=True)
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(f"[controller/stop] 无法导入 parm_sdk: {exc}", flush=True)
        return
    for arm_id, label in ((1, "左"), (2, "右")):
        arm = None
        try:
            arm = Arm(host=arm_host, status_port=None)
            arm.cancel_task(arm_id)
            print(f"[controller/stop] {label}臂 cancel_task({arm_id}) 已发送", flush=True)
        except Exception as exc:
            print(f"[controller/stop] {label}臂 cancel_task 失败: {exc}", flush=True)
        finally:
            if arm is not None:
                try:
                    arm.close()
                except Exception:
                    pass
