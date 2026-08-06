#!/usr/bin/env python3
"""Create offline force/torque charts from a Hole-A placement JSONL log.

The x axis is the approximate distance (mm) commanded along the search axis.
Force values are sensor-frame changes relative to the force baseline, not
world-frame force components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # This is an offline tool; do not require a desktop.
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
COLORS = ("tab:red", "tab:green", "tab:blue")
FORCE_LABELS = ("dFx", "dFy", "dFz")
TORQUE_LABELS = ("dTx", "dTy", "dTz")


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"日志文件不存在：{path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"日志第 {line_number} 行不是合法 JSON：{exc}") from exc
        if isinstance(event, dict):
            events.append(event)
    return events


def numeric_vector(event: dict[str, Any], key: str) -> list[float] | None:
    value = event.get(key)
    if not isinstance(value, list) or len(value) != 3:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return [float(item) for item in value]


def continuous_samples(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for event in events:
        # Prefer the 50-Hz data.  Older non-continuous runs have force_sample.
        if event.get("event") not in {"continuous_force_monitor_tick", "force_sample"}:
            continue
        if not isinstance(event.get("back_mm"), (int, float)):
            continue
        force = numeric_vector(event, "delta_force_n")
        torque = numeric_vector(event, "delta_torque_nm")
        if force is None or torque is None:
            continue
        samples.append({
            "back_mm": float(event["back_mm"]),
            "force": force,
            "torque": torque,
            "force_norm": float(event.get("delta_force_norm_n", sum(v * v for v in force) ** 0.5)),
            "torque_norm": float(event.get("delta_torque_norm_nm", sum(v * v for v in torque) ** 0.5)),
        })
    return samples


def event_markers(events: list[dict[str, Any]]) -> list[tuple[float, str, str]]:
    markers: list[tuple[float, str, str]] = []
    release_starts = [
        event for event in events
        if event.get("event") == "continuous_release_state"
        and event.get("release_frames") == 1
        and isinstance(event.get("back_mm"), (int, float))
    ]
    if release_starts:
        markers.append((float(release_starts[0]["back_mm"]), "release start", "tab:green"))
    decisions = [
        event for event in events
        if event.get("event") == "insertion_decision"
        and isinstance(event.get("back_mm"), (int, float))
        and (str(event.get("result", "")).startswith("entered_") or "contact_limit" in str(event.get("result", "")))
    ]
    if decisions:
        markers.append((float(decisions[0]["back_mm"]), str(decisions[0].get("result")), "tab:purple"))
    return markers


def decorate_axis(axis: plt.Axes, markers: list[tuple[float, str, str]]) -> None:
    for position, label, color in markers:
        axis.axvline(position, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=label)
    axis.grid(True, alpha=0.25)
    axis.set_xlabel("Search distance (mm)")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=8)


def save_force_axes(x: list[float], force: list[list[float]], markers: list[tuple[float, str, str]], out: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6), layout="constrained")
    for index, label in enumerate(FORCE_LABELS):
        axis.plot(x, [value[index] for value in force], label=label, color=COLORS[index], linewidth=1.5)
    axis.set_title("Force components relative to baseline")
    axis.set_ylabel("Force change (N)")
    decorate_axis(axis, markers)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def save_torque_axes(x: list[float], torque: list[list[float]], markers: list[tuple[float, str, str]], out: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6), layout="constrained")
    for index, label in enumerate(TORQUE_LABELS):
        axis.plot(x, [value[index] for value in torque], label=label, color=COLORS[index], linewidth=1.5)
    axis.set_title("Torque components relative to baseline")
    axis.set_ylabel("Torque change (N m)")
    decorate_axis(axis, markers)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def save_norms(x: list[float], force_norm: list[float], torque_norm: list[float], markers: list[tuple[float, str, str]], out: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    axes[0].plot(x, force_norm, color="black", label="|dF|", linewidth=1.7)
    axes[0].set_title("Resultant force")
    axes[0].set_ylabel("|dF| (N)")
    axes[1].plot(x, torque_norm, color="tab:orange", label="|dT|", linewidth=1.7)
    axes[1].set_title("Resultant torque")
    axes[1].set_ylabel("|dT| (N m)")
    for axis in axes:
        decorate_axis(axis, markers)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def save_overview(x: list[float], force: list[list[float]], torque: list[list[float]], force_norm: list[float], torque_norm: list[float], markers: list[tuple[float, str, str]], out: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    for index, label in enumerate(FORCE_LABELS):
        axes[0, 0].plot(x, [value[index] for value in force], label=label, color=COLORS[index])
    axes[0, 0].set_title("Force components")
    axes[0, 0].set_ylabel("N")
    for index, label in enumerate(TORQUE_LABELS):
        axes[0, 1].plot(x, [value[index] for value in torque], label=label, color=COLORS[index])
    axes[0, 1].set_title("Torque components")
    axes[0, 1].set_ylabel("N m")
    axes[1, 0].plot(x, force_norm, color="black", label="|dF|")
    axes[1, 0].set_title("Resultant force")
    axes[1, 0].set_ylabel("N")
    axes[1, 1].plot(x, torque_norm, color="tab:orange", label="|dT|")
    axes[1, 1].set_title("Resultant torque")
    axes[1, 1].set_ylabel("N m")
    for axis in axes.flat:
        decorate_axis(axis, markers)
    figure.suptitle("Hole-A placement force/torque overview", fontsize=14)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def latest_log(workpiece: int | None) -> Path:
    pattern = f"hole_a_workpiece_{workpiece}_*.log" if workpiece is not None else "hole_a_workpiece_*.log"
    paths = sorted(LOG_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise ValueError(f"未找到日志：{LOG_DIR / pattern}")
    return paths[-1]


def all_logs(workpiece: int) -> list[Path]:
    paths = sorted(LOG_DIR.glob(f"hole_a_workpiece_{workpiece}_*.log"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise ValueError(f"未找到工件 {workpiece} 的日志：{LOG_DIR}")
    return paths


def trajectory_label(path: Path, events: list[dict[str, Any]]) -> str:
    decision = next(
        (
            event.get("result") for event in events
            if event.get("event") == "insertion_decision"
            and str(event.get("result", "")).startswith("entered_")
        ),
        "no_terminal_decision",
    )
    timestamp = path.stem.removeprefix("hole_a_workpiece_").split("_", 1)[-1]
    return f"{timestamp} ({decision})"


def save_component_comparison(
    trajectories: list[tuple[str, list[dict[str, Any]]]],
    *,
    value_key: str,
    labels: tuple[str, str, str],
    unit: str,
    title: str,
    out: Path,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, layout="constrained")
    for axis_index, axis in enumerate(axes):
        for trajectory_index, (label, samples) in enumerate(trajectories):
            axis.plot(
                [sample["back_mm"] for sample in samples],
                [sample[value_key][axis_index] for sample in samples],
                label=label,
                linewidth=1.4,
            )
        axis.set_title(labels[axis_index])
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("Search distance (mm)")
    figure.suptitle(title, fontsize=14)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def save_norm_comparison(
    trajectories: list[tuple[str, list[dict[str, Any]]]], out: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout="constrained")
    for label, samples in trajectories:
        x = [sample["back_mm"] for sample in samples]
        axes[0].plot(x, [sample["force_norm"] for sample in samples], label=label, linewidth=1.4)
        axes[1].plot(x, [sample["torque_norm"] for sample in samples], label=label, linewidth=1.4)
    axes[0].set_title("Resultant force across trajectories")
    axes[0].set_ylabel("|dF| (N)")
    axes[1].set_title("Resultant torque across trajectories")
    axes[1].set_ylabel("|dT| (N m)")
    axes[1].set_xlabel("Search distance (mm)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=7)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def plot_all_trajectories(paths: list[Path], output_dir: Path) -> None:
    trajectories: list[tuple[str, list[dict[str, Any]]]] = []
    for path in paths:
        events = load_events(path)
        samples = continuous_samples(events)
        if samples:
            trajectories.append((trajectory_label(path, events), samples))
    if not trajectories:
        raise ValueError("所选日志中没有可绘制的连续力数据")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_component_comparison(
        trajectories,
        value_key="force",
        labels=FORCE_LABELS,
        unit="Force change (N)",
        title="Force components: all placement trajectories",
        out=output_dir / "01_all_trajectories_force_components.png",
    )
    save_component_comparison(
        trajectories,
        value_key="torque",
        labels=TORQUE_LABELS,
        unit="Torque change (N m)",
        title="Torque components: all placement trajectories",
        out=output_dir / "02_all_trajectories_torque_components.png",
    )
    save_norm_comparison(trajectories, output_dir / "03_all_trajectories_resultants.png")
    print(f"叠加轨迹数：{len(trajectories)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, help="要绘制的 JSONL 日志文件")
    source.add_argument("--latest", action="store_true", help="自动选择 logs 中最新的一份日志")
    source.add_argument("--all", action="store_true", help="叠加绘制指定工件的全部日志")
    parser.add_argument("--workpiece", type=int, choices=(1, 2), help="与 --latest 配合，仅选择指定工件")
    parser.add_argument("--output-dir", type=Path, help="图片输出目录；默认与日志同名的 .plots 目录")
    args = parser.parse_args()
    if args.workpiece is not None and not (args.latest or args.all):
        parser.error("--workpiece 只能与 --latest 或 --all 一起使用")
    if args.all and args.workpiece is None:
        parser.error("--all 必须指定 --workpiece 1 或 --workpiece 2")
    try:
        if args.all:
            paths = all_logs(args.workpiece)
            output_dir = (
                args.output_dir.resolve()
                if args.output_dir else LOG_DIR / f"workpiece_{args.workpiece}_trajectory_comparison.plots"
            )
            plot_all_trajectories(paths, output_dir)
            print("已生成：")
            for path in sorted(output_dir.glob("*.png")):
                print(f"  {path}")
            return 0
        log_path = latest_log(args.workpiece) if args.latest else args.log.resolve()
        events = load_events(log_path)
        samples = continuous_samples(events)
        if not samples:
            raise ValueError("日志中没有可绘制的 continuous_force_monitor_tick 或 force_sample 数据")
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir.resolve() if args.output_dir else log_path.with_suffix(".plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    x = [sample["back_mm"] for sample in samples]
    force = [sample["force"] for sample in samples]
    torque = [sample["torque"] for sample in samples]
    force_norm = [sample["force_norm"] for sample in samples]
    torque_norm = [sample["torque_norm"] for sample in samples]
    markers = event_markers(events)

    save_force_axes(x, force, markers, output_dir / "01_force_components.png")
    save_torque_axes(x, torque, markers, output_dir / "02_torque_components.png")
    save_norms(x, force_norm, torque_norm, markers, output_dir / "03_resultant_force_torque.png")
    save_overview(x, force, torque, force_norm, torque_norm, markers, output_dir / "04_overview.png")

    print(f"日志：{log_path}")
    print(f"采样点：{len(samples)}；横轴为沿搜索轴的近似移动距离。")
    print("已生成：")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
