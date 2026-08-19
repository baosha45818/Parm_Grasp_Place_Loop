"""连接真实 ForceSensor，交互式验证抓空/单件/多件重量判定。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from tools.grasp_weight_check import (
    ForceDataError,
    GraspClass,
    WeightThresholds,
    collect_force_window,
    verify_grasp_after_lift,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--hand", type=int, choices=(1, 2), default=1)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--empty-max-g", type=float, required=True)
    parser.add_argument("--single-min-g", type=float, required=True)
    parser.add_argument("--single-max-g", type=float, required=True)
    parser.add_argument("--multiple-min-g", type=float, required=True)
    parser.add_argument("--max-axis-mad-n", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = WeightThresholds(
        empty_max_g=args.empty_max_g,
        single_min_g=args.single_min_g,
        single_max_g=args.single_max_g,
        multiple_min_g=args.multiple_min_g,
        max_axis_mad_n=args.max_axis_mad_n,
    )

    try:
        from parm_sdk import ForceSensor
    except ImportError as exc:
        print(f"无法导入 parm_sdk：{exc}")
        return 5

    try:
        with ForceSensor(host=args.host, hand=args.hand) as sensor:
            sensor.wait_first(timeout=args.timeout)
            input("将空夹爪移到抬升检查位姿并保持静止，按 Enter 采空载基线...")
            baseline = collect_force_window(
                sensor, sample_count=args.samples, timeout_s=args.timeout
            )
            print(f"空载中位力: {baseline.median_force_n} N")

            input(
                "完成抓取并垂直上移离开桌面，回到抬升检查位姿后按 Enter 检测..."
            )
            result = verify_grasp_after_lift(
                sensor,
                baseline,
                thresholds,
                sample_count=args.samples,
                timeout_s=args.timeout,
            )
    except (ForceDataError, TimeoutError) as exc:
        print(f"采样失败：{exc}")
        return 5
    except Exception as exc:
        # 真实 SDK 的连接/协议异常类型可能随版本变化，手工测试统一转为故障退出码。
        print(f"力传感器访问失败：{type(exc).__name__}: {exc}")
        return 5

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return {
        GraspClass.SINGLE: 0,
        GraspClass.EMPTY: 2,
        GraspClass.MULTIPLE: 3,
        GraspClass.UNKNOWN: 4,
    }[result.grasp_class]


if __name__ == "__main__":
    raise SystemExit(main())
