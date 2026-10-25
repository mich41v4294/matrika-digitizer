import asyncio
import json
import time
from typing import Any, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event_seq: int = 0
        self.latest_state: dict[str, Any] = {
            "active_stage": "idle",
            "active_model": None,
            "current_scan_id": None,
            "current_row_id": None,
            "current_row_index": None,
            "total_rows": None,
            "current_sheet_id": None,
            "current_slovakiana_key": None,
            "current_sheet_label": None,
            "job_id": None,
            "job_sheet_index": None,
            "job_total_sheets": None,
            "progress_pct": 0,
            "started_at": None,
            "updated_at": None,
            "message_sk": "Systém je pripravený.",
            "message_en": "Pipeline idle and ready.",
            "last_completed": None,
        }
        self.recent_completions: list[dict[str, Any]] = []

    async def connect(self, websocket: WebSocket) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        await websocket.accept()
        self.active_connections.add(websocket)
        await websocket.send_text(json.dumps({"type": "state", "data": self.latest_state}, default=str))

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if event_type == "state":
            now = time.time()
            prev_stage = self.latest_state.get("active_stage", "idle")
            new_stage = payload.get("active_stage", prev_stage)
            if new_stage == "idle":
                self.latest_state["current_row_id"] = payload.get("current_row_id")
                self.latest_state["current_row_index"] = payload.get("current_row_index")
                self.latest_state["total_rows"] = payload.get("total_rows")
                self.latest_state["started_at"] = None
                self.latest_state["updated_at"] = now
                if "job_id" not in payload:
                    self.latest_state["job_id"] = None
                    self.latest_state["job_sheet_index"] = None
                    self.latest_state["job_total_sheets"] = None
            else:
                if prev_stage == "idle" or self.latest_state.get("started_at") is None:
                    self.latest_state["started_at"] = now
                self.latest_state["updated_at"] = now
            self.latest_state.update(payload)
            payload = dict(self.latest_state)
        message = json.dumps({"type": event_type, "data": payload}, default=str)
        stale: list[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                stale.append(connection)
        for s in stale:
            self.active_connections.discard(s)

    async def emit_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._event_seq += 1
        event: dict[str, Any] = {
            "event_id": self._event_seq,
            "timestamp": time.time(),
            "status": payload.get("status", "completed"),
            **payload,
        }
        self.latest_state["last_completed"] = event
        self.recent_completions.append(event)
        if len(self.recent_completions) > 25:
            self.recent_completions = self.recent_completions[-25:]
        await self.broadcast("ocr_completed", event)
        return event

    def broadcast_sync(self, event_type: str, payload: dict[str, Any]) -> None:
        """Thread-safe fire-and-forget broadcast from sync endpoints or worker threads."""
        if not self.active_connections and event_type != "state":
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(event_type, payload))
            return
        except RuntimeError:
            pass
        if self._loop is not None and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.broadcast(event_type, payload), self._loop)
            except Exception:
                pass


ws_manager = ConnectionManager()


@router.websocket("/ws/pipeline")
async def pipeline_websocket(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "data": ws_manager.latest_state}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
