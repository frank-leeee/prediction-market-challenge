from __future__ import annotations

import json
import sys
import traceback

from orderbook_pm_challenge.loader import load_strategy_factory
from orderbook_pm_challenge.sandbox import (
    install_builtin_restrictions,
    install_import_restrictions,
    load_strategy_factory_in_sandbox,
)
from orderbook_pm_challenge.strategy_host import deserialize_state, serialize_action


def _emit(payload: dict[str, object], *, stream) -> None:
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def _emit_error(message: str, *, stream) -> None:
    _emit({"success": False, "error": message}, stream=stream)


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    init_line = sys.stdin.readline()
    if not init_line:
        _emit_error("Strategy host did not receive an init payload", stream=protocol_out)
        return 1

    try:
        init_payload = json.loads(init_line)
    except json.JSONDecodeError as exc:
        _emit_error(f"Invalid init payload: {exc}", stream=protocol_out)
        return 1

    sandbox = bool(init_payload.get("sandbox"))
    strategy_path = str(init_payload["strategy_path"])

    try:
        if sandbox:
            install_import_restrictions()
            factory = load_strategy_factory_in_sandbox(strategy_path)
            install_builtin_restrictions()
        else:
            factory = load_strategy_factory(strategy_path)
        strategy = factory()
    except Exception:
        _emit_error(traceback.format_exc(), stream=protocol_out)
        return 1

    _emit({"success": True}, stream=protocol_out)

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            op = request.get("op")
            if op == "on_step":
                state = deserialize_state(request["state"])
                actions = strategy.on_step(state)
                payloads = [serialize_action(action) for action in actions]
                _emit({"success": True, "actions": payloads}, stream=protocol_out)
            elif op == "close":
                _emit({"success": True}, stream=protocol_out)
                return 0
            else:
                _emit_error(f"Unknown strategy host operation: {op!r}", stream=protocol_out)
                return 1
        except Exception:
            _emit_error(traceback.format_exc(), stream=protocol_out)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
