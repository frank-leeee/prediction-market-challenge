from __future__ import annotations

import json
import math
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .types import CancelAll, CancelOrder, OwnOrderView, PlaceOrder, Side, StepState


def serialize_state(state: StepState) -> dict[str, object]:
    return {
        "step": state.step,
        "steps_remaining": state.steps_remaining,
        "yes_inventory": state.yes_inventory,
        "no_inventory": state.no_inventory,
        "cash": state.cash,
        "reserved_cash": state.reserved_cash,
        "free_cash": state.free_cash,
        "competitor_best_bid_ticks": state.competitor_best_bid_ticks,
        "competitor_best_ask_ticks": state.competitor_best_ask_ticks,
        "buy_filled_quantity": state.buy_filled_quantity,
        "sell_filled_quantity": state.sell_filled_quantity,
        "own_orders": [
            {
                "order_id": order.order_id,
                "side": order.side.value,
                "price_ticks": order.price_ticks,
                "remaining_quantity": order.remaining_quantity,
                "submitted_step": order.submitted_step,
            }
            for order in state.own_orders
        ],
    }


def deserialize_state(payload: dict[str, object]) -> StepState:
    own_orders = tuple(
        OwnOrderView(
            order_id=str(order["order_id"]),
            side=Side(str(order["side"])),
            price_ticks=int(order["price_ticks"]),
            remaining_quantity=float(order["remaining_quantity"]),
            submitted_step=int(order["submitted_step"]),
        )
        for order in payload["own_orders"]
    )
    return StepState(
        step=int(payload["step"]),
        steps_remaining=int(payload["steps_remaining"]),
        yes_inventory=float(payload["yes_inventory"]),
        no_inventory=float(payload["no_inventory"]),
        cash=float(payload["cash"]),
        reserved_cash=float(payload["reserved_cash"]),
        free_cash=float(payload["free_cash"]),
        competitor_best_bid_ticks=(
            None if payload["competitor_best_bid_ticks"] is None else int(payload["competitor_best_bid_ticks"])
        ),
        competitor_best_ask_ticks=(
            None if payload["competitor_best_ask_ticks"] is None else int(payload["competitor_best_ask_ticks"])
        ),
        buy_filled_quantity=float(payload["buy_filled_quantity"]),
        sell_filled_quantity=float(payload["sell_filled_quantity"]),
        own_orders=own_orders,
    )


def serialize_action(action: object) -> dict[str, object]:
    if isinstance(action, CancelAll):
        return {"type": "cancel_all"}
    if isinstance(action, CancelOrder):
        return {"type": "cancel_order", "order_id": action.order_id}
    if isinstance(action, PlaceOrder):
        return {
            "type": "place_order",
            "side": action.side.value,
            "price_ticks": action.price_ticks,
            "quantity": action.quantity,
            "client_order_id": action.client_order_id,
        }
    raise TypeError(f"Unsupported action: {action!r}")


def deserialize_actions(payloads: list[object]) -> list[object]:
    actions: list[object] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError(f"Strategy action payload must be an object, got {payload!r}")
        action_type = payload.get("type")
        if action_type == "cancel_all":
            actions.append(CancelAll())
        elif action_type == "cancel_order":
            actions.append(CancelOrder(order_id=str(payload["order_id"])))
        elif action_type == "place_order":
            actions.append(
                PlaceOrder(
                    side=Side(str(payload["side"])),
                    price_ticks=int(payload["price_ticks"]),
                    quantity=float(payload["quantity"]),
                    client_order_id=(
                        None if payload.get("client_order_id") is None else str(payload["client_order_id"])
                    ),
                )
            )
        else:
            raise TypeError(f"Unsupported action payload: {payload!r}")
    return actions


class SubprocessStrategyProxy:
    def __init__(
        self,
        strategy_path: str,
        *,
        sandbox: bool = False,
        nsjail_path: str | None = None,
        timeout: int | None = None,
        max_output_bytes: int | None = None,
    ) -> None:
        self._strategy_path = str(Path(strategy_path).expanduser().resolve())
        self._sandbox = sandbox
        self._timeout = timeout
        self._deadline = None if timeout is None else time.monotonic() + timeout
        self._max_output_bytes = max_output_bytes
        self._stderr_file = tempfile.TemporaryFile()
        self._nsjail_config_path: str | None = None
        self._closed = False

        command = self._build_command(nsjail_path=nsjail_path)
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
        )

        try:
            response = self._round_trip(
                {
                    "op": "init",
                    "strategy_path": self._strategy_path,
                    "sandbox": self._sandbox,
                }
            )
        except Exception:
            self.close()
            raise

        if not response.get("success"):
            self.close()
            raise RuntimeError(str(response.get("error", "Strategy host failed to initialize")))

    def _build_command(self, *, nsjail_path: str | None) -> list[str]:
        python_bin = sys.executable
        worker_module = "orderbook_pm_challenge._strategy_worker"
        command = [python_bin, "-m", worker_module]
        if not (self._sandbox and nsjail_path):
            return command

        from .sandbox import _generate_nsjail_config

        package_path = str(Path(__file__).parent.resolve())
        time_limit = max(1, int(math.ceil(self._timeout or 300)))
        nsjail_cfg = _generate_nsjail_config(
            python_bin,
            self._strategy_path,
            package_path,
            time_limit=time_limit,
        )
        cfg_fd, cfg_path = tempfile.mkstemp(suffix=".cfg", prefix="nsjail_pm_strategy_")
        with os.fdopen(cfg_fd, "w") as handle:
            handle.write(nsjail_cfg)
        self._nsjail_config_path = cfg_path
        return [nsjail_path, "--config", cfg_path, "--", *command]

    def on_step(self, state: StepState) -> list[object]:
        response = self._round_trip({"op": "on_step", "state": serialize_state(state)})
        if not response.get("success"):
            raise RuntimeError(str(response.get("error", "Strategy callback failed")))
        payloads = response.get("actions", [])
        if not isinstance(payloads, list):
            raise RuntimeError(f"Strategy returned invalid action payload: {payloads!r}")
        return deserialize_actions(payloads)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self._proc.poll() is None:
                try:
                    self._send({"op": "close"})
                except Exception:
                    pass
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=1)
        finally:
            for stream in (self._proc.stdin, self._proc.stdout):
                if stream is not None:
                    stream.close()
            self._stderr_file.close()
            if self._nsjail_config_path is not None and os.path.exists(self._nsjail_config_path):
                os.unlink(self._nsjail_config_path)

    def _round_trip(self, payload: dict[str, object]) -> dict[str, object]:
        self._enforce_output_limit()
        self._send(payload)
        response = self._receive()
        self._enforce_output_limit()
        return response

    def _send(self, payload: dict[str, object]) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("Strategy host stdin is unavailable")
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._unexpected_exit_message()) from exc

    def _receive(self) -> dict[str, object]:
        if self._proc.stdout is None:
            raise RuntimeError("Strategy host stdout is unavailable")

        timeout = self._remaining_timeout()
        if timeout is None:
            line = self._proc.stdout.readline()
        else:
            ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
            if not ready:
                self._proc.kill()
                self._proc.wait(timeout=1)
                raise RuntimeError(self._timeout_message())
            line = self._proc.stdout.readline()

        if not line:
            raise RuntimeError(self._unexpected_exit_message())

        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Strategy host returned invalid JSON: {line!r}") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"Strategy host returned invalid response: {response!r}")
        return response

    def _remaining_timeout(self) -> float | None:
        if self._deadline is None:
            return None
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self._proc.kill()
            self._proc.wait(timeout=1)
            raise RuntimeError(self._timeout_message())
        return remaining

    def _enforce_output_limit(self) -> None:
        if self._max_output_bytes is None:
            return
        self._stderr_file.flush()
        self._stderr_file.seek(0, os.SEEK_END)
        size = self._stderr_file.tell()
        if size <= self._max_output_bytes:
            return
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=1)
        raise RuntimeError(
            "Strategy host exceeded output limit "
            f"of {self._max_output_bytes} bytes ({size} bytes written)"
        )

    def _timeout_message(self) -> str:
        if self._timeout is None:
            return "Strategy host timed out"
        return f"Strategy host timed out after {self._timeout} seconds"

    def _unexpected_exit_message(self) -> str:
        stderr = self._stderr_text()
        if stderr:
            return stderr
        return f"Strategy host exited with code {self._proc.returncode}"

    def _stderr_text(self) -> str:
        self._stderr_file.flush()
        self._stderr_file.seek(0)
        data = self._stderr_file.read()
        if not data:
            return ""
        return data.decode("utf-8", errors="replace").strip()


def load_subprocess_strategy_factory(
    strategy_path: str,
    *,
    sandbox: bool = False,
    nsjail_path: str | None = None,
    timeout: int | None = None,
    max_output_bytes: int | None = None,
):
    resolved_path = str(Path(strategy_path).expanduser().resolve())

    def factory():
        return SubprocessStrategyProxy(
            resolved_path,
            sandbox=sandbox,
            nsjail_path=nsjail_path,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
        )

    return factory
