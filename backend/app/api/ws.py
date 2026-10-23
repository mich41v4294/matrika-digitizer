import json
from typing import Any, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
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
        await websocket.accept()
        self.active_connections.add(websocket)
        await websocket.send_text(json.dumps({"type": "state", "data": self.latest_state}))

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, event_type: str, payload: dict[str, Any]) -> None:
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
