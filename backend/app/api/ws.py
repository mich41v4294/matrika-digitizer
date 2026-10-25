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
            "queued_count": 0,
            "active_task": None,
            "queued_tasks": [],
        }
        self.recent_completions: list[dict[str, Any]] = []

    async def connect(self, websocket: WebSocket) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        await websocket.accept()
        self.active_connections.add(websocket)
        try:
            from backend.app.pipeline.task_queue import task_queue

            snap = task_queue.get_snapshot()
            self.latest_state["queued_count"] = snap["counts"]["queued"]
            self.latest_state["active_task"] = snap["active_task"]
            self.latest_state["queued_tasks"] = snap["queued_tasks"]
        except Exception:
            snap = None
        await websocket.send_text(json.dumps({"type": "state", "data": self.latest_state}, default=str))
        if snap is not None:
            await websocket.send_text(json.dumps({"type": "task_queue", "data": snap}, default=str))
        try:
            from backend.app.config import settings

            await websocket.send_text(
                json.dumps({"type": "feature_flags", "data": settings.get_feature_flags()}, default=str)
            )
        except Exception:
            pass

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if event_type == "state":
            payload = dict(payload)
            from_queue = bool(payload.pop("_from_task_queue", False))
            try:
                from backend.app.pipeline.task_queue import task_queue

                if (
                    task_queue.has_running_task()
                    and not from_queue
                    and not task_queue.is_inside_active_task()
                ):
                    return
                task_queue.on_pipeline_state_update(payload)
                if payload.get("active_stage") == "idle" and not from_queue:
                    other_running = task_queue.get_other_running_task()
                    if other_running is not None:
                        payload.update(
                            {
                                "active_stage": (
                                    other_running.active_stage
                                    if other_running.active_stage not in ("queued", "idle")
                                    else "ocr"
                                ),
                                "active_model": other_running.active_model,
                                "current_scan_id": other_running.metadata.get("current_scan_id")
                                or other_running.metadata.get("scan_id"),
                                "current_row_id": other_running.metadata.get("current_row_id")
                                or other_running.metadata.get("row_id"),
                                "current_sheet_id": other_running.metadata.get("current_sheet_id")
                                or other_running.metadata.get("sheet_id"),
                                "current_slovakiana_key": other_running.metadata.get(
                                    "current_slovakiana_key"
                                )
                                or other_running.metadata.get("slovakiana_key"),
                                "current_sheet_label": other_running.metadata.get(
                                    "current_sheet_label"
                                )
                                or other_running.metadata.get("sheet_label"),
                                "job_id": other_running.metadata.get("job_id"),
                                "progress_pct": max(5, other_running.progress_pct),
                                "message_sk": other_running.message_sk,
                                "message_en": other_running.message_en,
                            }
                        )
                snap = task_queue.get_snapshot()
                payload["queued_count"] = snap["counts"]["queued"]
                payload["active_task"] = snap["active_task"]
                payload["running_tasks"] = snap["running_tasks"]
                payload["queued_tasks"] = snap["queued_tasks"]
            except Exception:
                snap = None

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
