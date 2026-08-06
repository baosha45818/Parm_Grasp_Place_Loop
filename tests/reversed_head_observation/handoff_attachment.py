"""Host-side request/ack handshake for the MoveIt attached workpiece owner."""

from __future__ import annotations

import json
from pathlib import Path
import time
from uuid import uuid4


VALID_OWNERS = {"left", "right", "none"}


def write_attachment_request(path: Path, owner: str) -> str:
    if owner not in VALID_OWNERS:
        raise ValueError(f"invalid attachment owner: {owner!r}")
    state = json.loads(path.read_text(encoding="utf-8"))
    request_id = uuid4().hex
    state["requested_owner"] = owner
    state["request_id"] = request_id
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return request_id


def attachment_request_applied(path: Path, owner: str, request_id: str) -> bool:
    state = json.loads(path.read_text(encoding="utf-8"))
    applied_owner = None if owner == "none" else owner
    return (
        state.get("applied_owner") == applied_owner
        and state.get("applied_request_id") == request_id
    )


def request_attachment_owner(
    path: Path, owner: str, timeout_s: float = 5.0
) -> dict:
    if timeout_s <= 0.0:
        raise ValueError("attachment acknowledgement timeout must be positive")
    request_id = write_attachment_request(path, owner)
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            if attachment_request_applied(path, owner, request_id):
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(
        f"MoveIt attachment publisher did not acknowledge owner={owner!r}, "
        f"request_id={request_id}; last_error={last_error}"
    )
