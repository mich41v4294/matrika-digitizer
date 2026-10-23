import asyncio
import json
from typing import Any, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.latest_state: dict[str, Any] = {
            "active_stage": "idle",
            "active_model": None,
            "current_scan_id": None,
            "current_row_id": None,
            "progress_pct": 0,
            "message_sk": "Systém je pripravený.",
            "message_en": "Pipeline idle and ready.",
        }

    async def connect(self, websocket: WebSocket) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        await websocket.accept()
        self.active_connections.add(websocket)
        await websocket.send_text(json.dumps({"type": "state", "data": self.latest_state}))

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if event_type == "state":
            self.latest_state.update(payload)
        message = json.dumps({"type": event_type, "data": payload}, default=str)
        stale: list[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                stale.append(connection)
        for s in stale:
            self.active_connections.discard(s)

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
