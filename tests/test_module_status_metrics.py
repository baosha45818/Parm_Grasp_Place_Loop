#!/usr/bin/env python3
"""离线验证模块耗时、任务成功率和异常回退记录。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_types import ResultStatus
from modules.module_status import report, report_exception


class ModuleStatusMetricsTest(unittest.TestCase):
    def _environment(self, directory: str) -> dict[str, str]:
        return {
            "PARM_MODULE_METRICS_DIR": directory,
            "PARM_MODULE_STATUS_FILE": str(Path(directory) / "latest.json"),
            "PARM_LOOP_RUN_ID": "offline-test",
        }

    def test_task_success_rate_and_partial_rollback_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, self._environment(temporary), clear=False
        ):
            report(
                "module_1_to_3_left_pipeline",
                ResultStatus.WAIT,
                "开始任务1",
            )
            time.sleep(0.001)
            report(
                "module_1_to_3_left_pipeline",
                ResultStatus.SUCCESS,
                "左侧链完成，返回分支=BRANCH_1_FRONT",
            )
            report("module_10_right_idle", ResultStatus.WAIT, "开始右臂收尾")
            time.sleep(0.001)
            report(
                "module_10_right_idle",
                ResultStatus.SUCCESS,
                "右臂已放置、开爪并回 Idle",
            )
            report(
                "module_1_to_3_left_pipeline",
                ResultStatus.WAIT,
                "开始任务2",
            )
            time.sleep(0.001)
            report(
                "module_1_to_3_left_pipeline",
                ResultStatus.RETRY,
                "未预测到抓取姿态",
                reason_code="NO_GRASP_PREDICTED",
            )

            summary = json.loads(
                (Path(temporary) / "summary.json").read_text(encoding="utf-8")
            )
            events = [
                json.loads(line)
                for line in (Path(temporary) / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        task = summary["task_execution"]
        self.assertEqual(task["attempts"], 2)
        self.assertEqual(task["successes"], 1)
        self.assertEqual(task["failures"], 1)
        self.assertEqual(task["recovered_failures"], 1)
        self.assertEqual(task["success_rate"], 0.5)
        self.assertGreater(events[-1]["duration_seconds"], 0.0)
        self.assertEqual(
            events[-1]["recovery"]["target_module"],
            "module_1",
        )

    def test_fatal_loop_error_closes_active_module_and_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, self._environment(temporary), clear=False
        ):
            report(
                "modular_loop",
                ResultStatus.WAIT,
                "状态机启动",
            )
            report(
                "module_1_to_3_left_pipeline",
                ResultStatus.WAIT,
                "开始任务",
            )
            report(
                "module_3_left_place",
                ResultStatus.WAIT,
                "开始固定 Buffer 放置",
            )
            report_exception(
                "modular_loop",
                RuntimeError("worker failed"),
                action="cancel 两臂并保留现场",
                target_module="initial_state",
                recovery_level="SAFE_STOP",
            )
            summary = json.loads(
                (Path(temporary) / "summary.json").read_text(encoding="utf-8")
            )
            last = summary["last_event"]

        self.assertEqual(summary["task_execution"]["in_progress"], 0)
        self.assertEqual(summary["task_execution"]["failures"], 1)
        self.assertEqual(
            summary["modules"]["module_3_left_place"]["aborted"],
            1,
        )
        self.assertEqual(
            last["recovery"]["target_module"],
            "initial_state",
        )
        self.assertIn(
            "module_3_left_place",
            {
                item["module"]
                for item in last["closed_active_modules"]
            },
        )


if __name__ == "__main__":
    unittest.main()
