#!/usr/bin/env python3
"""
Shared presentation layer for the inference-rs status boards.

Both `task rs:check` and the publish orchestrator render the same visual
language: hidden stdout, a compact live-updating tree grouped into sections with
✓/x/·/• glyphs and per-group wall-time, and a full captured-log dump only for
failures. That rendering lives here so the two callers stay byte-for-byte
identical instead of forking the ANSI/tree/redraw code.

The module is deliberately caller-agnostic. `check.py` fans out homogeneous
parallel checks keyed by task name; the publisher walks ordered Step objects. So
the renderer takes plain data — `Group`/`Row` value objects and right-hand
strings the caller has already computed — rather than reaching into either
caller's domain model.
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

COLORS = {
    "bold": "\x1b[1m",
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "cyan": "\x1b[36m",
    "dim": "\x1b[2m",
    "reset": "\x1b[0m",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Colors and cursor control only make sense on a real terminal; a piped/CI run
# gets plain text. Callers can override for tests, but the default matches what
# both boards want.
IS_INTERACTIVE = sys.stdout.isatty()


def color(text: str, color_name: str) -> str:
    if not IS_INTERACTIVE:
        return text
    return f"{COLORS[color_name]}{text}{COLORS['reset']}"


def styled(text: str, *color_names: str) -> str:
    if not IS_INTERACTIVE:
        return text
    prefix = "".join(COLORS[name] for name in color_names)
    return f"{prefix}{text}{COLORS['reset']}"


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def format_duration(duration_ms: float) -> str:
    return f"{duration_ms / 1000:.1f}s"


def status_glyph(result: Optional[dict], is_running: bool = False) -> str:
    """Glyph for a check/step result: pending dot, running dot, ✓ or x.

    `result` is any mapping with an "exit_code"; None means not started.
    """
    if not result:
        return color("•", "dim") if is_running else color("·", "dim")
    return color("✓", "green") if result["exit_code"] == 0 else color("x", "red")


@dataclass
class Row:
    """A single line under a group header: a glyph, a label, and right text.

    `right` is the trailing column (a duration, "running...", a detail note, or
    empty); the caller formats it so a check can show timings and the publisher
    can show "0.4.2 live". The renderer only lays out the tree connector, glyph,
    padded label, and right text.
    """

    glyph: str
    label: str
    right: str = ""


@dataclass
class Group:
    """A titled section of the board: a header (title + right text) and its rows."""

    title: str
    rows: list = field(default_factory=list)
    right: str = ""


def render_group_header(group: Group, duration_col: int) -> str:
    left = group.title.ljust(duration_col)
    return f"{styled(left, 'bold')} {group.right}".rstrip()


def render_group_child(row: Row, label_width: int, is_last: bool) -> str:
    connector = "└─" if is_last else "├─"
    label = row.label.ljust(label_width)
    left = f"{color(connector, 'dim')} {row.glyph} {label}"
    return f"{left} {row.right}".rstrip()


def render_group_block(group: Group, label_width: int, duration_col: int) -> list:
    """The rendered lines for one group: its header followed by each child row."""
    lines = [render_group_header(group, duration_col)]
    for index, row in enumerate(group.rows):
        lines.append(render_group_child(row, label_width, is_last=index == len(group.rows) - 1))
    return lines


class InteractiveRenderer:
    """Redraws a group tree in place on each update.

    The first render prints the block; every later render moves the cursor back
    up over the block it drew last time (`\\x1b[<n>F`) and clears each line
    (`\\x1b[2K`) before rewriting, so the summary updates without scrolling. The
    line count is tracked from the last frame so groups can grow or shrink
    between frames.

    `snapshot` is a caller-supplied callable returning the current
    `(groups, label_width, duration_col)`; the driver calls it each frame so it
    never has to know how the caller derives its rows.
    """

    def __init__(self, snapshot: Callable[[], tuple]):
        self._snapshot = snapshot
        self._rendered_lines = 0

    def render(self) -> None:
        if self._rendered_lines:
            # Move the cursor back up to the top of the summary block to redraw it in place.
            sys.stdout.write(f"\x1b[{self._rendered_lines}F")

        groups, label_width, duration_col = self._snapshot()
        lines: list = []
        for group in groups:
            lines.extend(render_group_block(group, label_width, duration_col))

        for line in lines:
            sys.stdout.write(f"\x1b[2K{line}\n")

        sys.stdout.flush()
        self._rendered_lines = len(lines)

    def reset(self) -> None:
        """Forget the last frame's line count so the NEXT render prints a fresh block
        instead of moving the cursor up to redraw in place. Call this after something
        (an interactive prompt, say) has written below the board and moved the cursor,
        so the in-place redraw would otherwise land in the wrong spot."""
        self._rendered_lines = 0

    @property
    def has_rendered(self) -> bool:
        return self._rendered_lines > 0


def print_failures(
    results: list,
    clean_output: Optional[Callable[[dict], str]] = None,
) -> None:
    """Dump the full captured log for every failing result.

    Each result is a mapping with "label", "task", "exit_code", and "output".
    `clean_output` optionally post-processes a result's raw output before it is
    printed (check.py strips its task-runner "Failed to run task" preamble); it
    defaults to the untouched output so non-task callers need not pass anything.
    """
    if clean_output is None:
        clean_output = lambda result: result["output"]

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
        output = clean_output(result)
        sys.stdout.write(output or "(no output)\n")
        if output and not output.endswith("\n"):
            sys.stdout.write("\n")

    print()


def print_results(results: list, format_row: Callable[[dict], str]) -> None:
    """Interactive-only recap listing every result with a retry hint on failures.

    `format_row` renders one result to its summary line; the caller owns that
    formatting since row layout differs between the boards.
    """
    if not IS_INTERACTIVE or all(result["exit_code"] == 0 for result in results):
        return

    print(styled("◆ Results", "bold", "cyan"))
    print(styled("─────────", "cyan"))

    for result in results:
        retry = "" if result["exit_code"] == 0 else f"  run: task {result['task']}"
        print(f"{format_row(result)}{retry}")
