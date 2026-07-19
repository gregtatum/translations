#!/usr/bin/env python3
"""
Run the inference-rs check tasks with concise, task-focused feedback.

Instead of streaming every tool log, this renders a compact pass/fail summary.
Light checks run in parallel; the CPU-heavy ones are serialized (see
MAX_CONCURRENT_HEAVY). Interactive terminals get a live-updating summary grouped
into "Parallel" and "Sequential" sections, each with its own wall-time total,
while non-TTY runs report the first completed failure and stop the rest.

Each entry in CHECKS is a task in this directory's Taskfile, namespaced under
`rs:`. Invoked via `task rs:check`.

The presentation (colors, the group-tree renderer, the failure-log dump) lives
in the shared `statusboard` module so this board and the publish orchestrator
read as one family; this file keeps the scheduler and the CHECKS list.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

from statusboard import (
    IS_INTERACTIVE,
    InteractiveRenderer,
    Group,
    Row,
    color,
    format_duration,
    print_failures,
    print_results,
    status_glyph,
    strip_ansi,
)

CHECKS = [
    {"task": "rs:lint-black", "label": "Lint Black"},
    {"task": "rs:lint-rust", "label": "Lint Rust"},
    {"task": "rs:lint-ci", "label": "Lint CI"},
    {"task": "rs:test", "label": "Rust Tests", "heavy": True},
    {"task": "rs:conformance", "label": "CLI Conformance", "heavy": True},
    {"task": "inference-build", "label": "Build Inference Engine", "heavy": True},
]

# The heavy checks (Rust Tests, CLI Conformance, Build Inference Engine) each
# spin up a build/test that saturates every core (cargo-nextest, cargo + wasm
# builds, cmake -j). Running them concurrently oversubscribes the CPU and thrashes:
# on a 6-performance-core machine, the three heavy checks measured ~19s racing
# against each other vs ~12s serialized. So we cap heavy checks to one at a time
# while letting the light (IO/startup-bound) checks run fully in parallel.
MAX_CONCURRENT_HEAVY = 1
_heavy_slots = threading.Semaphore(MAX_CONCURRENT_HEAVY)

LABEL_WIDTH = max(len(check["label"]) for check in CHECKS)

# The interactive summary is grouped by how the checks are scheduled: the light
# checks fan out in parallel, the heavy ones run one at a time. Each group shows
# its own wall-time total (the parallel group's ≈ its slowest check; the
# sequential group's ≈ the sum of its checks).
GROUPS = [
    ("Parallel", [check for check in CHECKS if not check.get("heavy")]),
    ("Sequential", [check for check in CHECKS if check.get("heavy")]),
]
# Column where the duration starts: tree connector ("├─ ") + glyph ("✓ ") + label.
DURATION_COL = 5 + LABEL_WIDTH


def format_row(check: dict, result: Optional[dict], is_running: bool = False) -> str:
    if not IS_INTERACTIVE:
        return format_plain_row(check, result, is_running)

    label = check["label"].ljust(LABEL_WIDTH)
    suffix = ""
    if is_running:
        suffix = " running..."
    elif result:
        suffix = f" {format_duration(result['duration_ms'])}"

    return f"{status_glyph(result, is_running)} {label}{suffix}"


def format_plain_row(check: dict, result: Optional[dict], is_running: bool = False) -> str:
    label = check["label"].ljust(LABEL_WIDTH)

    if is_running:
        return f"RUN  {label}"

    if not result:
        return f"WAIT {label}"

    status = "PASS" if result["exit_code"] == 0 else "FAIL"
    return f"{status} {label} {format_duration(result['duration_ms'])}"


def check_env() -> dict:
    env = dict(os.environ)
    env.setdefault("TERM", "xterm-256color")

    if IS_INTERACTIVE:
        env["FORCE_COLOR"] = "1"
        env.pop("NO_COLOR", None)

    return env


class CheckProcess:
    """A single check running as `task --silent --exit-code <task>`.

    Started in its own process group so the whole subtree can be killed when a sibling
    check fails first in a non-interactive run.
    """

    def __init__(self, check: dict):
        self.check = check
        self._started_at = time.monotonic()
        self._result: Optional[dict] = None
        self._proc = subprocess.Popen(
            ["task", "--silent", "--exit-code", check["task"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=check_env(),
            start_new_session=True,
        )

    def wait(self) -> dict:
        """Block until the check finishes and return its result (memoized)."""
        if self._result is None:
            try:
                output, _ = self._proc.communicate()
            except Exception as error:  # noqa: BLE001 - surface any spawn/IO failure as output
                output = f"{error}\n"
            ended_at = time.monotonic()
            exit_code = self._proc.returncode
            self._result = {
                **self.check,
                "started_at": self._started_at,
                "ended_at": ended_at,
                "duration_ms": (ended_at - self._started_at) * 1000,
                "exit_code": exit_code if exit_code is not None else 1,
                "output": output or "",
            }
        return self._result

    def kill(self) -> None:
        if self._proc.poll() is not None:
            return
        self._signal(signal.SIGTERM)

        def force_kill() -> None:
            if self._proc.poll() is None:
                self._signal(signal.SIGKILL)

        timer = threading.Timer(1.0, force_kill)
        timer.daemon = True
        timer.start()

    def _signal(self, sig: int) -> None:
        try:
            os.killpg(os.getpgid(self._proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            self._proc.send_signal(sig)


def group_duration_text(checks: list, results: dict, started_at: dict, now: float) -> str:
    starts = [started_at[check["task"]] for check in checks if check["task"] in started_at]
    if not starts:
        return ""
    group_start = min(starts)
    if all(check["task"] in results for check in checks):
        group_end = max(results[check["task"]]["ended_at"] for check in checks)
        return format_duration((group_end - group_start) * 1000)
    # Still running: elapsed wall time so far, dimmed to read as provisional.
    return color(format_duration((now - group_start) * 1000), "dim")


def child_right_text(result: Optional[dict], is_running: bool) -> str:
    if is_running:
        return color("running...", "dim")
    if result:
        return format_duration(result["duration_ms"])
    return ""


def board_snapshot(results: dict, running_tasks: set, started_at: dict) -> tuple:
    """The current group tree for the shared interactive renderer."""
    now = time.monotonic()
    groups = []
    for title, checks in GROUPS:
        rows = []
        for check in checks:
            result = results.get(check["task"])
            is_running = check["task"] in running_tasks
            rows.append(
                Row(
                    glyph=status_glyph(result, is_running),
                    label=check["label"],
                    right=child_right_text(result, is_running),
                )
            )
        groups.append(
            Group(
                title=title,
                rows=rows,
                right=group_duration_text(checks, results, started_at, now),
            )
        )
    return groups, LABEL_WIDTH, DURATION_COL


def clean_task_output(result: dict) -> str:
    prefix = f'task: Failed to run task "{result["task"]}"'
    lines = [
        line for line in result["output"].split("\n") if not strip_ansi(line).startswith(prefix)
    ]
    return "\n".join(lines)


def run_interactive_checks(results_by_task: dict) -> None:
    # A heavy check appears as pending (not "running...") until it holds a heavy
    # slot and has started its subprocess; light checks start immediately, so they
    # are seeded here as running with a shared start time for the group total.
    start_time = time.monotonic()
    running_tasks: set = {check["task"] for check in CHECKS if not check.get("heavy")}
    started_at: dict = {task: start_time for task in running_tasks}
    render_lock = threading.Lock()

    renderer = InteractiveRenderer(
        lambda: board_snapshot(results_by_task, running_tasks, started_at)
    )
    renderer.render()

    def watch(check: dict) -> None:
        if check.get("heavy"):
            _heavy_slots.acquire()
        try:
            with render_lock:
                running_tasks.add(check["task"])
                started_at.setdefault(check["task"], time.monotonic())
                renderer.render()
            result = CheckProcess(check).wait()
        finally:
            if check.get("heavy"):
                _heavy_slots.release()
        with render_lock:
            results_by_task[check["task"]] = result
            running_tasks.discard(check["task"])
            renderer.render()

    join_all(threading.Thread(target=watch, args=(check,)) for check in CHECKS)

    if renderer.has_rendered:
        print()


def run_non_interactive_checks(results_by_task: dict) -> None:
    state_lock = threading.Lock()
    started_processes: list = []
    first_failure: dict = {}
    successful_results: list = []
    # Set on the first failure to stop queued heavy checks from ever starting
    # their subprocess (they check it before and after acquiring the semaphore).
    abort = threading.Event()

    def watch(check: dict) -> None:
        if check.get("heavy"):
            if abort.is_set():
                return
            _heavy_slots.acquire()
        try:
            if abort.is_set():
                return
            print(format_row(check, None, is_running=True), flush=True)
            with state_lock:
                check_process = CheckProcess(check)
                started_processes.append(check_process)
            result = check_process.wait()
        finally:
            if check.get("heavy"):
                _heavy_slots.release()

        with state_lock:
            if result["exit_code"] != 0:
                if not first_failure:
                    first_failure["result"] = result
                    results_by_task[result["task"]] = result
                    print(format_row(result, result), flush=True)
                    abort.set()
                    for other in started_processes:
                        if other is not check_process:
                            other.kill()
                return
            if not first_failure:
                successful_results.append(result)

    join_all(threading.Thread(target=watch, args=(check,)) for check in CHECKS)

    if not first_failure:
        for result in successful_results:
            results_by_task[result["task"]] = result
            print(format_row(result, result), flush=True)


def join_all(threads) -> None:
    threads = list(threads)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main() -> None:
    results_by_task: dict = {}

    if IS_INTERACTIVE:
        run_interactive_checks(results_by_task)
    else:
        run_non_interactive_checks(results_by_task)

    results = [
        results_by_task[check["task"]] for check in CHECKS if check["task"] in results_by_task
    ]

    print_failures(results, clean_output=clean_task_output)
    print_results(results, lambda result: format_row(result, result))

    if any(result["exit_code"] != 0 for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
