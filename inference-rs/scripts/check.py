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
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

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

IS_INTERACTIVE = sys.stdout.isatty()
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
# Lines the live summary redraws in place: one header per group plus every check.
RENDER_LINE_COUNT = len(GROUPS) + len(CHECKS)

COLORS = {
    "bold": "\x1b[1m",
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "cyan": "\x1b[36m",
    "dim": "\x1b[2m",
    "reset": "\x1b[0m",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def format_duration(duration_ms: float) -> str:
    return f"{duration_ms / 1000:.1f}s"


def color(text: str, color_name: str) -> str:
    if not IS_INTERACTIVE:
        return text
    return f"{COLORS[color_name]}{text}{COLORS['reset']}"


def styled(text: str, *color_names: str) -> str:
    if not IS_INTERACTIVE:
        return text
    prefix = "".join(COLORS[name] for name in color_names)
    return f"{prefix}{text}{COLORS['reset']}"


def status_glyph(result: Optional[dict], is_running: bool = False) -> str:
    if not result:
        return color("•", "dim") if is_running else color("·", "dim")
    return color("✓", "green") if result["exit_code"] == 0 else color("x", "red")


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


_has_rendered = False


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


def render_group_header(
    title: str, checks: list, results: dict, started_at: dict, now: float
) -> str:
    left = title.ljust(DURATION_COL)
    duration = group_duration_text(checks, results, started_at, now)
    return f"{styled(left, 'bold')} {duration}".rstrip()


def render_group_child(
    check: dict, result: Optional[dict], is_running: bool, is_last: bool
) -> str:
    connector = "└─" if is_last else "├─"
    label = check["label"].ljust(LABEL_WIDTH)
    if is_running:
        duration = color("running...", "dim")
    elif result:
        duration = format_duration(result["duration_ms"])
    else:
        duration = ""
    left = f"{color(connector, 'dim')} {status_glyph(result, is_running)} {label}"
    return f"{left} {duration}".rstrip()


def render_interactive(results: dict, running_tasks: set, started_at: dict) -> None:
    global _has_rendered

    if _has_rendered:
        # Move the cursor back up to the top of the summary block to redraw it in place.
        sys.stdout.write(f"\x1b[{RENDER_LINE_COUNT}F")

    now = time.monotonic()
    for title, checks in GROUPS:
        header = render_group_header(title, checks, results, started_at, now)
        sys.stdout.write(f"\x1b[2K{header}\n")
        for index, check in enumerate(checks):
            result = results.get(check["task"])
            is_running = check["task"] in running_tasks
            child = render_group_child(check, result, is_running, is_last=index == len(checks) - 1)
            sys.stdout.write(f"\x1b[2K{child}\n")

    sys.stdout.flush()
    _has_rendered = True


def clean_task_output(result: dict) -> str:
    prefix = f'task: Failed to run task "{result["task"]}"'
    lines = [
        line for line in result["output"].split("\n") if not strip_ansi(line).startswith(prefix)
    ]
    return "\n".join(lines)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def print_failures(results: list) -> None:
    failures = [result for result in results if result["exit_code"] != 0]

    if not failures:
        return

    if IS_INTERACTIVE:
        print(styled("✖ Failures", "bold", "red"))
        print(styled("──────────", "red"))
    else:
        print("FAILURES")

    for result in failures:
        label = result["label"]
        task = result["task"]
        exit_code = result["exit_code"]
        cmd = f"task --silent --exit-code {task}"
        print()
        if IS_INTERACTIVE:
            print(
                f"{styled('┌─', 'red')} {styled(label, 'bold', 'red')} failed "
                f"{styled(f'exit {exit_code}', 'red')}  "
                f"{styled(f'run: task {task}', 'dim')}"
            )
            print(f"{styled('│', 'red')} {styled(cmd, 'dim')}")
            print(styled("└─ output", "red"))
        else:
            print(f"FAIL {label} exit {exit_code} | run: task {task}")
            print(f"cmd: {cmd}")
            print("output:")
        output = clean_task_output(result)
        sys.stdout.write(output or "(no output)\n")
        if output and not output.endswith("\n"):
            sys.stdout.write("\n")

    print()


def print_results(results: list) -> None:
    if not IS_INTERACTIVE or all(result["exit_code"] == 0 for result in results):
        return

    print(styled("◆ Results", "bold", "cyan"))
    print(styled("─────────", "cyan"))

    for result in results:
        retry = "" if result["exit_code"] == 0 else f"  run: task {result['task']}"
        print(f"{format_row(result, result)}{retry}")


def run_interactive_checks(results_by_task: dict) -> None:
    # A heavy check appears as pending (not "running...") until it holds a heavy
    # slot and has started its subprocess; light checks start immediately, so they
    # are seeded here as running with a shared start time for the group total.
    start_time = time.monotonic()
    running_tasks: set = {check["task"] for check in CHECKS if not check.get("heavy")}
    started_at: dict = {task: start_time for task in running_tasks}
    render_lock = threading.Lock()

    render_interactive(results_by_task, running_tasks, started_at)

    def watch(check: dict) -> None:
        if check.get("heavy"):
            _heavy_slots.acquire()
        try:
            with render_lock:
                running_tasks.add(check["task"])
                started_at.setdefault(check["task"], time.monotonic())
                render_interactive(results_by_task, running_tasks, started_at)
            result = CheckProcess(check).wait()
        finally:
            if check.get("heavy"):
                _heavy_slots.release()
        with render_lock:
            results_by_task[check["task"]] = result
            running_tasks.discard(check["task"])
            render_interactive(results_by_task, running_tasks, started_at)

    join_all(threading.Thread(target=watch, args=(check,)) for check in CHECKS)

    if _has_rendered:
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

    print_failures(results)
    print_results(results)

    if any(result["exit_code"] != 0 for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
