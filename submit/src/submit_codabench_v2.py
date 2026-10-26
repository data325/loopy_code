#!/usr/bin/env python3
"""Upload and poll a submission using Codabench's current dataset API."""

from __future__ import annotations

import argparse
import getpass
import os
import hashlib
import json
import time
from pathlib import Path

import requests


BASE = "https://www.codabench.org"


def checked(response, action):
    if response.ok:
        return response
    raise SystemExit(f"{action} failed: HTTP {response.status_code}: {response.text[:1000]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--task", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=1200)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    if not bundle.is_file():
        raise SystemExit(f"Bundle not found: {bundle}")
    password = os.environ.get("CODABENCH_PASSWORD") or getpass.getpass("Codabench password: ")
    auth = checked(
        requests.post(
            f"{BASE}/api/api-token-auth/",
            json={"username": args.username, "password": password},
            timeout=30,
        ),
        "Authentication",
    ).json()
    password = None
    headers = {"Authorization": f"Token {auth['token']}"}
    quota = checked(
        requests.get(f"{BASE}/api/can_make_submission/{args.phase}/", headers=headers, timeout=30),
        "Quota check",
    ).json()
    print("Quota check:", json.dumps(quota), flush=True)
    if not quota.get("can"):
        raise SystemExit("Codabench reports that a submission cannot be made")

    size = bundle.stat().st_size
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    record = checked(
        requests.post(
            f"{BASE}/api/datasets/",
            headers=headers,
            json={
                "type": "submission",
                "request_sassy_file_name": bundle.name,
                "file_name": bundle.name,
                "file_size": size,
                "name": bundle.name,
                "is_public": False,
            },
            timeout=30,
        ),
        "Create upload record",
    ).json()
    key = record["key"]
    signed_url = record.get("sassy_url") or record.get("data_file")
    if not signed_url:
        raise SystemExit(f"Upload record lacks signed URL; fields={sorted(record)}")
    print(f"Upload record: key={key}, bytes={size}, sha256={digest}", flush=True)
    with bundle.open("rb") as stream:
        checked(
            requests.put(
                signed_url,
                data=stream,
                headers={"Content-Type": "application/zip", "Content-Length": str(size)},
                timeout=300,
            ),
            "Upload bundle",
        )
    checked(
        requests.put(f"{BASE}/api/datasets/completed/{key}/", headers=headers, timeout=30),
        "Finalize upload",
    )

    body = {"data": key, "phase": args.phase}
    if args.task:
        body["tasks"] = args.task
    submission = checked(
        requests.post(f"{BASE}/api/submissions/", headers=headers, json=body, timeout=30),
        "Create submission",
    ).json()
    submission_id = submission["id"]
    print(f"Submission created: id={submission_id}, status={submission.get('status')}", flush=True)

    deadline = time.monotonic() + args.poll_seconds
    previous = None
    while time.monotonic() < deadline:
        current = checked(
            requests.get(f"{BASE}/api/submissions/{submission_id}/", headers=headers, timeout=30),
            "Poll submission",
        ).json()
        status = str(current.get("status", "")).lower()
        if status != previous:
            print(f"Submission {submission_id}: {current.get('status')}", flush=True)
            previous = status
        if status in {"finished", "failed", "cancelled"}:
            print(json.dumps(current, indent=2), flush=True)
            return
        time.sleep(5)
    print(f"Submission {submission_id}: still running", flush=True)


if __name__ == "__main__":
    main()
