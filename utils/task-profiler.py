import argparse
import http.server
import json
import os
import sys
import urllib
import re
from datetime import datetime
import socket
import time
from typing import NamedTuple, Optional
import webbrowser

import requests

"""
Perform pre-flight checks for a training run. This script will exercise the taskgraph
and output information about the training steps, and how they are organized. This is
helpful when debugging and understanding pipeline changes, or when writing a training
config.

Usage:

    task task-profiler -- --log_path path/to/live.log

    task task-profiler -- --task_id FOquY6dBSySnNpIotuJpOQ

"""

class LogRow:
    """
    A row from the log
    """
    component: str
    time: Optional[datetime]
    message: str

    def __init__(self, component: str, time: Optional[datetime], message: str) -> None:
        self.component = component
        self.time = time
        self.message = message

    def unix_milliseconds(self) -> int:
        """The profiler requires the Unix epoch milliseconds"""
        return int(time.mktime(self.time.timetuple()) * 1000.0)

class UniqueStringArray:
    """
    This ported from the profiler.
    """

    def __init__(self, original_array: Optional[list[str]] = None):
        if original_array is None:
            original_array = []
        self._array: list[str] = original_array[:]
        self._string_to_index: dict[str, int] = {string: i for i, string in enumerate(original_array)}

    def get_string(self, index: int, els: Optional[str] = None) -> str:
        """Get the string at the given index."""
        if not self.has_index(index):
            if els:
                print(f"index {index} not in UniqueStringArray")
                return els
            raise ValueError(f"index {index} not in UniqueStringArray")
        return self._array[index]

    def has_index(self, index: int) -> bool:
        """Check if the given index exists in the array."""
        return index < len(self._array)

    def has_string(self, s: str) -> bool:
        """Check if the given string exists in the array."""
        return s in self._string_to_index

    def index_for_string(self, s: str) -> int:
        """Get the index for the given string, adding it if it doesn't exist."""
        index = self._string_to_index.get(s)
        if index is None:
            index = len(self._array)
            self._string_to_index[s] = index
            self._array.append(s)
        return index

    def serialize_to_array(self) -> list[str]:
        """Serialize the array to a new list."""
        return self._array[:]

def read_log_file(lines) -> list[LogRow]:
    log_pattern = re.compile(r'''
        \[                               # [taskcluster:warn 2024-05-20T14:40:11.353Z]
            (?P<component>\w+)           # Capture the component name, here "taskcluster"
            (:(?P<log_level>\w+))?       # An optional log level, like "warn"
            \s*                          # Ignore whitespace
            (?P<time>[\d\-T:.Z]+)        # Capture the timestamp
        \]                               #
        \s*                              # Ignore whitespace
        (?P<message>.*)                  # Capture the rest as the message
    ''', re.VERBOSE)

    log_rows = []

    for line in lines:
        if not line.strip():
            continue

        match = log_pattern.match(line)
        if match:
            log_rows.append(LogRow(
                component=match.group("component"),
                time=datetime.strptime(match.group("time"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                message=match.group("message")
            ))
        else:
            log_rows.append(LogRow(
                component="",
                time=None,
                message=line
            ))

    return log_rows

def fixup_log_rows(log_rows: list[LogRow]) -> list[LogRow]:
    "[2024-05-20 14:42:21]"
    # Remove any extra datetimes.
    # "[2024-05-20 15:04:26] Ep. 1 : Up. 12 : Sen. 24,225 : ..."
    #  ^^^^^^^^^^^^^^^^^^^^^
    regex = r"^\s*\[[\d\-T:.Z]\]\s*"

    for log_row in log_rows:
        # Remove the date-time part from the log string
        message = re.sub(regex, "", log_row.message)
        log_row.message = message


def get_empty_profile():
    return {
        "meta": {
            "interval": 1,
            "startTime": int(time.time() * 1000),
            "abi": "",
            "misc": "",
            "oscpu": "",
            "platform": "",
            "processType": 0,
            "extensions": {"id": [], "name": [], "baseURL": [], "length": 0},
            "categories": get_categories(),
            "product": "Taskcluster Task",
            "stackwalk": 0,
            "toolkit": "",
            "version": 29,
            "preprocessedProfileVersion": 48,
            "appBuildID": "",
            "sourceURL": "",
            "physicalCPUs": 0,
            "logicalCPUs": 0,
            "CPUName": "",
            "symbolicated": True,
            "markerSchema": [
                get_task_schema()
            ],
        },
        "libs": [],
        "pages": [],
        "threads": [],
    }

def get_categories():
    """
    Colors are listed here:
    https://github.com/firefox-devtools/profiler/blob/ffe2b6af0fbf4f91a389cc31fd7df776bb198034/src/utils/colors.js#L96
    """
    return [
        {
        "name": 'none',
        "color": 'grey',
        "subcategories": ['Other'],
        },
        {
        "name": 'fetches',
        "color": 'purple',
        "subcategories": ['Other'],
        },
        {
        "name": 'vcs',
        "color": 'orange',
        "subcategories": ['Other'],
        },
        {
        "name": 'setup',
        "color": 'lightblue',
        "subcategories": ['Other'],
        },
        {
        "name": 'taskcluster',
        "color": 'green',
        "subcategories": ['Other'],
        },
    ]

def get_empty_thread():
    """
    https://github.com/firefox-devtools/profiler/blob/ffe2b6af0fbf4f91a389cc31fd7df776bb198034/src/profile-logic/data-structures.js#L358
    """
    return {
        "processType": "default",
        "processStartupTime": 0,
        "processShutdownTime": None,
        "registerTime": 0,
        "unregisterTime": None,
        "pausedRanges": [],
        "name": "Empty",
        "isMainThread": True,
        "pid": "0",
        "tid": 0,
        "samples": {
            "weightType": "tracing-ms",
            "weight": [],
            "stack": [],
            "time": [],
            "length": 0,
        },
        "markers": {
            "data": [],
            "name": [],
            "startTime": [],
            "endTime": [],
            "phase": [],
            "category": [],
            "length": 0,
        },
        "stackTable": {"frame": [], "prefix": [], "category": [], "subcategory": [], "length": 0},
        "frameTable": {
            "address": [],
            "inlineDepth": [],
            "category": [],
            "subcategory": [],
            "func": [],
            "nativeSymbol": [],
            "innerWindowID": [],
            "implementation": [],
            "line": [],
            "column": [],
            "length": 0,
        },
        "stringArray": [],
        "funcTable": {
            "isJS": [],
            "relevantForJS": [],
            "name": [],
            "resource": [],
            "fileName": [],
            "lineNumber": [],
            "columnNumber": [],
            "length": 0,
        },
        "resourceTable": {"lib": [], "name": [], "host": [], "type": [], "length": 0},
        "nativeSymbols": {
            "libIndex": [],
            "address": [],
            "name": [],
            "functionSize": [],
            "length": 0,
        },
    }

def get_task_schema():
    """
    This is documented in the profiler:
    Markers: https://github.com/firefox-devtools/profiler/src/types/markers.js
    Schema: https://github.com/firefox-devtools/profiler/blob/df32b2d320cb4c9bc7b4ee988a291afa33daff71/src/types/markers.js#L100
    """
    return {
        "name": 'LiveLogRow',
        "tooltipLabel": '{marker.data.message}',
        "tableLabel": '{marker.data.message}',
        "chartLabel": '{marker.data.message}',
        "display": ['marker-chart', 'marker-table', 'timeline-overview'],
        "data": [
            {
                "key": 'startTime',
                "label": 'Start time',
                "format": 'string',
            },
            {
                "key": 'message',
                "label": 'Log Message',
                "format": 'string',
                "searchable": "true",
            },
            {
                "key": 'hour',
                "label": 'Hour',
                "format": 'string',
            },
            {
                "key": 'date',
                "label": 'Date',
                "format": 'string',
            },
            {
                "key": 'time',
                "label": 'Time',
                "format": 'time',
            },
        ]
    }

Profile = dict[str, any]

def build_profile(log_rows: list[LogRow]) -> Profile:
    profile = get_empty_profile()

    # Compute and save the profilel start time.
    profile_start_time = 0
    for log_row in log_rows:
        if log_row.time:
            profile_start_time = log_row.unix_milliseconds()
            profile["meta"]["startTime"] = profile_start_time

    # Create the thread that we'll attach the markers to.
    thread = get_empty_thread()
    thread["name"] = "Live Log"
    profile["threads"].append(thread)
    thread["isMainThread"] = True
    markers = thread["markers"]

    # Map a category name to its index.
    category_index_dict = { category["name"]:index for index, category in enumerate(profile["meta"]["categories"])}
    string_array = UniqueStringArray()

    # run_end = profile_start_time
    for log_row in log_rows:
        if not log_row.time:
            continue
        run_start = log_row.unix_milliseconds()
        instant_marker = 0
        markers["startTime"].append(run_start - profile_start_time)

        markers["endTime"].append(None)
        markers["phase"].append(instant_marker)

        # Code to add a duration marker:
        # duration_marker = 1
        # markers["endTime"].append(run_end - profile_start_time)
        # markers["phase"].append(duration_marker)

        markers["category"].append(category_index_dict.get(log_row.component, 0))
        markers["name"].append(string_array.index_for_string(log_row.component))

        markers["data"].append({
          "type": 'LiveLogRow',
          "name": "LiveLogRow",
          "message": log_row.message,
          "hour": log_row.time.strftime('%H:%M:%S'),
          "date": log_row.time.strftime("%Y-%m-%d"),
        #   "url": `https://${url.host}/tasks/groups/${taskGroup.taskGroupId}`,
        })

        markers["length"] += 1

    thread["stringArray"] = string_array.serialize_to_array()

    return profile

waiting_for_request = True
profile_data = None

def open_profile(profile):
    global profile_data
    profile_data = json.dumps(profile).encode('utf-8')

    port = get_free_port()
    json_url = f"http://localhost:{port}"

    webbrowser.open("https://profiler.firefox.com/from-url/" + urllib.parse.quote(json_url, safe='') + "?name=" + urllib.parse.quote("Live Log", safe=''))
    server = http.server.HTTPServer(("", port), ServeFile)

    while waiting_for_request:
        server.handle_request()

class ServeFile(http.server.BaseHTTPRequestHandler):
    """Creates a one-time server that just serves one file."""

    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def log_message(self, *args):
        # Disable server logging.
        pass

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        self._set_headers()
        try:
            global profile_data
            self.wfile.write(profile_data)
        except Exception as exception:
            print("Failed to serve the file", exception)
            pass
        global waiting_for_request
        waiting_for_request = False

def get_free_port() -> int:
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("--task_id", metavar="ID", type=str, help="The id of the task")
    parser.add_argument("--log_path", metavar="ID", type=str, help="The path to a local log file")
    args = parser.parse_args()

    if args.task_id and args.log_path:
        print("A --task_id and --log_path were both provided, only one is needed")
        sys.exit(1)

    if args.log_path:
        with open(args.log_path, 'r') as file:
            print("Reading the local live log")
            log_rows = read_log_file(file.readlines())
    elif args.task_id:
        url = f"https://firefoxci.taskcluster-artifacts.net/{args.task_id}/0/public/logs/live_backing.log"
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure we notice bad responses
        log_rows = read_log_file(response.iter_lines(decode_unicode=True))
    else:
        print("A --task_id or --log_path must be provided")
        sys.exit(1)

    fixup_log_rows(log_rows)

    print("Build the profile")
    profile = build_profile(log_rows)
    open_profile(profile)

if __name__ == "__main__":
    main()
