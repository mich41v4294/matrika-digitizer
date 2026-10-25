from __future__ import annotations

import asyncio
from contextvars import ContextVar
import time
from typing import Any, Awaitable, Callable

from backend.app.config import settings

_ACTIVE_QUEUE_TASK_ID: ContextVar[int | None] = ContextVar("_ACTIVE_QUEUE_TASK_ID", default=None)


def _resolve_task_provider(
    *,
    explicit_provider: str | None = None,
    category: str = "ocr",
    initial_stage: str = "ocr",
    active_model: str | None = None,
) -> str:
    if explicit_provider in ("freellm", "local"):
        return explicit_provider
    mod_str = str(active_model or "").strip().lower()
    if mod_str.startswith("freellm:"):
        return "freellm"
    if mod_str.startswith("local:"):
        return "local"
    if category in ("ocr", "debug"):
        try:
            if initial_stage in ("extract", "ocr", "decision"):
                resolved = settings.resolve_stage_ai(initial_stage, requested_model=active_model)
                return str(resolved.get("provider") or "local")
            # Full scan/sheet/batch pipeline task: check extract stage (and overall provider)
            ext_res = settings.resolve_stage_ai("extract")
            if ext_res.get("provider") == "freellm":
                return "freellm"
        except Exception:
            pass
    return "local"


class TaskQueueItem:
    def __init__(
        self,
        *,
        task_num: int,
        task_type: str,
        category: str,
        title_sk: str,
        title_en: str,
        provider: str = "local",
        initial_stage: str = "queued",
        active_model: str | None = None,
        target_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        priority: int = 10,
    ) -> None:
        self.id: int = task_num
        self.task_id: str = f"task-{task_num}"
        self.task_type: str = task_type
        self.category: str = category
        self.provider: str = provider
        self.title_sk: str = title_sk
        self.title_en: str = title_en
        self.status: str = "queued"  # queued, running, completed, cancelled, paused, error
        self.priority: int = priority
        self.queue_position: int | None = None
        self.progress_pct: int = 0
        self.active_stage: str = initial_stage
        self.active_model: str | None = active_model
        self.message_sk: str = "Čaká vo fronte na uvoľnenie kapacity lokálneho modelu…"
        self.message_en: str = "Waiting in queue for local model concurrency slot…"
        self.target_url: str | None = target_url
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.created_at: float = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.duration_ms: int | None = None
        self.error: str | None = None
        self.cancel_requested: bool = False

        self._turn_event: asyncio.Event | None = None
        self._asyncio_task: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def to_dict(self, queue_position: int | None = None) -> dict[str, Any]:
        now = time.time()
        dur_ms = self.duration_ms
        if dur_ms is None and self.started_at is not None:
            end_ts = self.finished_at or now
            dur_ms = max(0, int((end_ts - self.started_at) * 1000))
        q_pos = queue_position if queue_position is not None else self.queue_position
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "category": self.category,
            "provider": self.provider,
            "title_sk": self.title_sk,
            "title_en": self.title_en,
            "status": self.status,
            "priority": self.priority,
            "queue_position": q_pos,
            "progress_pct": self.progress_pct,
            "active_stage": self.active_stage,
            "active_model": self.active_model,
            "message_sk": self.message_sk,
            "message_en": self.message_en,
            "target_url": self.target_url,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": dur_ms,
            "error": self.error,
        }


class TaskQueueManager:
    """
    Provider-aware task scheduler & queue:
    - FreeLLM (`provider == "freellm"`) requests/tasks are NEVER queued: they start immediately
      as they come and execute concurrently.
    - Local model (`provider == "local"`) requests/tasks are bounded by `settings.local_max_concurrent`
      (default 5 concurrent requests), queueing excess tasks in FIFO / priority order.
    """

    def __init__(self) -> None:
        self._seq: int = 0
        self._running_ids: list[int] = []
        self._items_by_id: dict[int, TaskQueueItem] = {}
        self._waiting_ids: list[int] = []
        self._history: list[TaskQueueItem] = []
        self._max_history: int = 50

    def _get_local_limit(self) -> int:
        return max(1, int(getattr(settings, "local_max_concurrent", 5) or 5))

    def _running_local_count(self) -> int:
        cnt = 0
        for tid in self._running_ids:
            it = self._items_by_id.get(tid)
            if it and it.provider != "freellm":
                cnt += 1
        return cnt

    def is_inside_active_task(self) -> bool:
        cur_id = _ACTIVE_QUEUE_TASK_ID.get()
        if cur_id is None:
            return False
        self._cleanup_stale_loop_state()
        return cur_id in self._running_ids

    @property
    def running_task_id(self) -> int | None:
        self._cleanup_stale_loop_state()
        return self._running_ids[0] if self._running_ids else None

    def has_running_task(self) -> bool:
        self._cleanup_stale_loop_state()
        return len(self._running_ids) > 0

    def get_other_running_task(self) -> TaskQueueItem | None:
        """
        Returns another currently running task (different from the calling task's ContextVar, if any),
        so `ws_manager` does not flip to `idle` when one task finishes while another is still running.
        """
        self._cleanup_stale_loop_state()
        cur_id = _ACTIVE_QUEUE_TASK_ID.get()
        for tid in self._running_ids:
            if tid != cur_id and tid in self._items_by_id:
                return self._items_by_id[tid]
        return None

    def is_current_task_cancelled(self) -> bool:
        cur_id = _ACTIVE_QUEUE_TASK_ID.get()
        if cur_id is None:
            return False
        item = self._items_by_id.get(cur_id)
        return bool(item and item.cancel_requested)

    def _cleanup_stale_loop_state(self) -> None:
        """
        In unit tests that call `asyncio.run()` multiple times, a previous event loop
        may have closed. Clean up any orphaned state bound to a closed loop.
        """
        for r_id in list(self._running_ids):
            r_item = self._items_by_id.get(r_id)
            if r_item is None:
                self._running_ids.remove(r_id)
            elif r_item._loop is not None and r_item._loop.is_closed():
                self._items_by_id.pop(r_id, None)
                self._running_ids.remove(r_id)
                r_item.status = "cancelled"
                r_item.finished_at = time.time()
                self._append_history(r_item)
        for wid in list(self._waiting_ids):
            w_item = self._items_by_id.get(wid)
            if w_item is None:
                self._waiting_ids.remove(wid)
            elif w_item._loop is not None and w_item._loop.is_closed():
                self._items_by_id.pop(wid, None)
                self._waiting_ids.remove(wid)
                w_item.status = "cancelled"
                w_item.finished_at = time.time()
                self._append_history(w_item)

    def _append_history(self, item: TaskQueueItem) -> None:
        item.queue_position = None
        self._history.insert(0, item)
        if len(self._history) > self._max_history:
            self._history = self._history[: self._max_history]

    def _sort_waiting_ids(self) -> None:
        self._waiting_ids.sort(
            key=lambda tid: (
                self._items_by_id[tid].priority if tid in self._items_by_id else 999,
                tid,
            )
        )
        for idx, tid in enumerate(self._waiting_ids, start=1):
            if tid in self._items_by_id:
                self._items_by_id[tid].queue_position = idx

    def get_queued_count(self) -> int:
        self._cleanup_stale_loop_state()
        return len(self._waiting_ids)

    def get_running_tasks_list(self) -> list[dict[str, Any]]:
        self._cleanup_stale_loop_state()
        out: list[dict[str, Any]] = []
        for tid in self._running_ids:
            it = self._items_by_id.get(tid)
            if it:
                out.append(it.to_dict(queue_position=0))
        return out

    def get_active_task_summary(self) -> dict[str, Any] | None:
        running = self.get_running_tasks_list()
        return running[0] if running else None

    def get_queued_tasks_list(self) -> list[dict[str, Any]]:
        self._cleanup_stale_loop_state()
        self._sort_waiting_ids()
        out: list[dict[str, Any]] = []
        for idx, tid in enumerate(self._waiting_ids, start=1):
            it = self._items_by_id.get(tid)
            if it:
                out.append(it.to_dict(queue_position=idx))
        return out

    def is_census_job_active_or_queued(self, job_id: int) -> bool:
        self._cleanup_stale_loop_state()
        for item in self._items_by_id.values():
            if item.metadata.get("job_id") == job_id and item.status in ("queued", "running"):
                return True
        return False

    def get_census_job_queue_info(self, job_id: int) -> dict[str, Any] | None:
        self._cleanup_stale_loop_state()
        self._sort_waiting_ids()
        for item in self._items_by_id.values():
            if item.metadata.get("job_id") == job_id and item.status in ("queued", "running"):
                return {
                    "task_id": item.task_id,
                    "queue_status": item.status,
                    "queue_position": item.queue_position,
                }
        return None

    def get_familysearch_queue_info(self) -> dict[str, Any]:
        self._cleanup_stale_loop_state()
        self._sort_waiting_ids()
        queued_fs = [
            self._items_by_id[tid]
            for tid in self._waiting_ids
            if tid in self._items_by_id and self._items_by_id[tid].category == "familysearch"
        ]
        is_running_fs = any(
            tid in self._items_by_id and self._items_by_id[tid].category == "familysearch"
            for tid in self._running_ids
        )
        return {
            "queued": len(queued_fs) > 0,
            "queued_count": len(queued_fs),
            "queue_position": queued_fs[0].queue_position if queued_fs else None,
            "running_in_queue": is_running_fs,
        }

    def get_snapshot(self) -> dict[str, Any]:
        self._cleanup_stale_loop_state()
        self._sort_waiting_ids()
        running_list = self.get_running_tasks_list()
        active_dict = running_list[0] if running_list else None
        queued_list = self.get_queued_tasks_list()
        history_list = [h.to_dict(queue_position=None) for h in self._history]

        completed_cnt = sum(1 for h in self._history if h.status == "completed")
        cancelled_cnt = sum(1 for h in self._history if h.status in ("cancelled", "paused"))
        error_cnt = sum(1 for h in self._history if h.status == "error")

        ai_conc: dict[str, Any] = {
            "local_max_concurrent": self._get_local_limit(),
            "active_local_requests": 0,
            "waiting_local_requests": 0,
            "active_freellm_requests": 0,
            "freellm_queued_requests": 0,
            "freellm_concurrent_unlimited": True,
            "in_flight": [],
            "waiting": [],
        }
        try:
            from backend.app.pipeline.ai_client import ai_client

            ai_conc = ai_client.get_concurrency_state()
        except Exception:
            pass

        return {
            "active_task": active_dict,
            "running_tasks": running_list,
            "queued_tasks": queued_list,
            "history": history_list,
            "concurrency": ai_conc,
            "counts": {
                "running": len(running_list),
                "queued": len(queued_list),
                "completed": completed_cnt,
                "cancelled": cancelled_cnt,
                "error": error_cnt,
                "total_active_and_queued": len(running_list) + len(queued_list),
                "local_max_concurrent": ai_conc["local_max_concurrent"],
                "active_local_requests": ai_conc["active_local_requests"],
                "waiting_local_requests": ai_conc["waiting_local_requests"],
                "active_freellm_requests": ai_conc["active_freellm_requests"],
            },
        }

    def on_pipeline_state_update(self, payload: dict[str, Any]) -> None:
        """Called by `ws_manager.broadcast('state', payload)` to keep the running task synced."""
        self._cleanup_stale_loop_state()
        if not self._running_ids:
            return
        cur_id = _ACTIVE_QUEUE_TASK_ID.get()
        item: TaskQueueItem | None = None
        if cur_id is not None and cur_id in self._items_by_id:
            item = self._items_by_id[cur_id]
        else:
            # Match by sheet_id / scan_id / job_id if available, else primary running task
            p_sheet = payload.get("current_sheet_id")
            p_scan = payload.get("current_scan_id")
            p_job = payload.get("job_id")
            for tid in self._running_ids:
                cand = self._items_by_id.get(tid)
                if not cand:
                    continue
                if (
                    (p_job is not None and cand.metadata.get("job_id") == p_job)
                    or (p_sheet is not None and cand.metadata.get("sheet_id") == p_sheet)
                    or (p_scan is not None and cand.metadata.get("scan_id") == p_scan)
                ):
                    item = cand
                    break
            if item is None:
                item = self._items_by_id.get(self._running_ids[0])
        if not item:
            return

        stage = payload.get("active_stage")
        if stage and stage != "idle":
            item.active_stage = str(stage)
        if payload.get("active_model") is not None:
            item.active_model = payload.get("active_model")
        if payload.get("progress_pct") is not None:
            try:
                pct = int(payload["progress_pct"])
                if stage != "idle" or pct >= 100:
                    item.progress_pct = max(0, min(100, pct))
            except (TypeError, ValueError):
                pass
        if payload.get("message_sk"):
            item.message_sk = str(payload["message_sk"])
        if payload.get("message_en"):
            item.message_en = str(payload["message_en"])
        for meta_key in (
            "current_scan_id",
            "current_row_id",
            "current_row_index",
            "total_rows",
            "current_sheet_id",
            "current_slovakiana_key",
            "current_sheet_label",
            "job_id",
            "job_sheet_index",
            "job_total_sheets",
        ):
            if meta_key in payload and payload[meta_key] is not None:
                item.metadata[meta_key] = payload[meta_key]

    def _broadcast_queue_sync(self) -> None:
        try:
            from backend.app.api.ws import ws_manager

            snap = self.get_snapshot()
            ws_manager.latest_state["queued_count"] = snap["counts"]["queued"]
            ws_manager.latest_state["active_task"] = snap["active_task"]
            ws_manager.latest_state["running_tasks"] = snap["running_tasks"]
            ws_manager.latest_state["queued_tasks"] = snap["queued_tasks"]
            ws_manager.broadcast_sync("task_queue", snap)
        except Exception:
            pass

    def _register_task_item(
        self,
        *,
        task_type: str,
        category: str,
        title_sk: str,
        title_en: str,
        provider: str | None = None,
        initial_stage: str = "ocr",
        active_model: str | None = None,
        target_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        priority: int = 10,
    ) -> TaskQueueItem:
        self._cleanup_stale_loop_state()
        resolved_provider = _resolve_task_provider(
            explicit_provider=provider,
            category=category,
            initial_stage=initial_stage,
            active_model=active_model,
        )
        self._seq += 1
        item = TaskQueueItem(
            task_num=self._seq,
            task_type=task_type,
            category=category,
            title_sk=title_sk,
            title_en=title_en,
            provider=resolved_provider,
            initial_stage=initial_stage,
            active_model=active_model,
            target_url=target_url,
            metadata=metadata,
            priority=priority,
        )
        try:
            loop = asyncio.get_running_loop()
            item._loop = loop
            item._turn_event = asyncio.Event()
        except RuntimeError:
            loop = None

        self._items_by_id[item.id] = item
        local_limit = self._get_local_limit()
        can_start_immediately = (
            resolved_provider == "freellm"
            or (self._running_local_count() < local_limit and not self._waiting_ids)
        )

        if can_start_immediately:
            self._running_ids.append(item.id)
            item.status = "running"
            item.queue_position = 0
            item.started_at = time.time()
            item.active_stage = initial_stage
            item.progress_pct = 5
            item.message_sk = f"Spúšťam: {title_sk}…"
            item.message_en = f"Starting: {title_en}…"
            if item._turn_event is not None:
                item._turn_event.set()
        else:
            item.status = "queued"
            item.active_stage = "queued"
            item.progress_pct = 0
            self._waiting_ids.append(item.id)
            self._sort_waiting_ids()
            pos = item.queue_position or len(self._waiting_ids)
            item.message_sk = (
                f"Vo fronte (#{pos}) – čaká na uvoľnenie slotu lokálneho modelu (limit {local_limit})…"
            )
            item.message_en = (
                f"Queued (#{pos}) – waiting for local model concurrency slot (limit {local_limit})…"
            )

        self._broadcast_queue_sync()
        if item.category == "familysearch" and item.status == "queued":
            self._notify_fs_queued_sync()
        return item

    def _notify_fs_queued_sync(self) -> None:
        try:
            from backend.app.api.ws import ws_manager
            from backend.app.mcp.familysearch_client import fs_mcp_manager

            state = fs_mcp_manager.get_progress_state(refresh_counts=False)
            ws_manager.broadcast_sync("fs_match_progress", state)
        except Exception:
            pass

    async def _execute_registered_item(
        self,
        item: TaskQueueItem,
        coro_fn: Callable[[], Awaitable[Any]],
        initial_stage: str,
    ) -> Any:
        loop = asyncio.get_running_loop()
        item._loop = loop
        if item._turn_event is None:
            item._turn_event = asyncio.Event()
            if item.status == "running":
                item._turn_event.set()

        try:
            if item.status == "queued":
                await item._turn_event.wait()

            if item.cancel_requested or item.status == "cancelled":
                raise asyncio.CancelledError(f"Task {item.task_id} was cancelled")

            # Transition from queued -> running
            if item.id not in self._running_ids:
                self._running_ids.append(item.id)
            item.status = "running"
            item.queue_position = 0
            if item.started_at is None:
                item.started_at = time.time()
            item.active_stage = initial_stage
            item.progress_pct = max(item.progress_pct, 5)
            item.message_sk = f"Spúšťam: {item.title_sk}…"
            item.message_en = f"Running: {item.title_en}…"
            self._broadcast_queue_sync()

            if self._running_ids and self._running_ids[0] == item.id:
                try:
                    from backend.app.api.ws import ws_manager

                    await ws_manager.broadcast(
                        "state",
                        {
                            "_from_task_queue": True,
                            "active_stage": initial_stage,
                            "active_model": item.active_model,
                            "current_scan_id": item.metadata.get("scan_id"),
                            "current_row_id": item.metadata.get("row_id"),
                            "current_row_index": None,
                            "total_rows": None,
                            "current_sheet_id": item.metadata.get("sheet_id"),
                            "current_slovakiana_key": item.metadata.get("slovakiana_key"),
                            "current_sheet_label": item.metadata.get("sheet_label"),
                            "job_id": item.metadata.get("job_id"),
                            "job_sheet_index": None,
                            "job_total_sheets": None,
                            "progress_pct": item.progress_pct,
                            "message_sk": item.message_sk,
                            "message_en": item.message_en,
                        },
                    )
                except Exception:
                    pass

            token = _ACTIVE_QUEUE_TASK_ID.set(item.id)
            try:
                result = await coro_fn()
            finally:
                _ACTIVE_QUEUE_TASK_ID.reset(token)

            now = time.time()
            item.finished_at = now
            item.duration_ms = max(1, int((now - (item.started_at or now)) * 1000))

            if item.cancel_requested:
                item.status = "cancelled"
                item.active_stage = "cancelled"
                item.message_sk = f"Zrušené: {item.title_sk}"
                item.message_en = f"Cancelled: {item.title_en}"
            elif isinstance(result, dict) and result.get("status") == "paused":
                item.status = "paused"
                item.active_stage = "paused"
                item.message_sk = f"Pozastavené: {item.title_sk}"
                item.message_en = f"Paused: {item.title_en}"
            elif isinstance(result, dict) and (
                result.get("status") == "error" or result.get("error")
            ):
                item.status = "error"
                item.active_stage = "error"
                err_txt = str(result.get("error") or "Error")
                item.error = err_txt
                item.message_sk = f"Chyba: {err_txt}"
                item.message_en = f"Error: {err_txt}"
            else:
                item.status = "completed"
                item.active_stage = "completed"
                item.progress_pct = 100
                if item.message_sk.startswith("Spúšťam:") or item.message_sk.startswith("Vo fronte"):
                    item.message_sk = f"Hotovo: {item.title_sk}"
                    item.message_en = f"Completed: {item.title_en}"

            return result

        except asyncio.CancelledError:
            now = time.time()
            item.status = "cancelled"
            item.active_stage = "cancelled"
            item.finished_at = now
            if item.started_at is not None:
                item.duration_ms = max(1, int((now - item.started_at) * 1000))
            item.message_sk = f"Úloha zrušená: {item.title_sk}"
            item.message_en = f"Task cancelled: {item.title_en}"
            raise
        except Exception as exc:
            now = time.time()
            item.status = "error"
            item.active_stage = "error"
            item.error = str(exc)[:500]
            item.finished_at = now
            if item.started_at is not None:
                item.duration_ms = max(1, int((now - item.started_at) * 1000))
            item.message_sk = f"Chyba ({item.title_sk}): {exc}"
            item.message_en = f"Error ({item.title_en}): {exc}"
            raise
        finally:
            self._items_by_id.pop(item.id, None)
            if item.id in self._waiting_ids:
                self._waiting_ids.remove(item.id)
            if item.id in self._running_ids:
                self._running_ids.remove(item.id)
            self._append_history(item)

            self._advance_next_waiting_task()
            self._broadcast_queue_sync()

    def _advance_next_waiting_task(self) -> None:
        self._sort_waiting_ids()
        local_limit = self._get_local_limit()
        while self._waiting_ids:
            # Find the first waiting task that can run (freellm tasks can always run; local tasks respect local_limit)
            picked_idx: int | None = None
            for idx, cand_id in enumerate(self._waiting_ids):
                cand = self._items_by_id.get(cand_id)
                if not cand or cand.cancel_requested or cand.status == "cancelled":
                    picked_idx = idx
                    break
                if cand.provider == "freellm" or self._running_local_count() < local_limit:
                    picked_idx = idx
                    break

            if picked_idx is None:
                break

            next_id = self._waiting_ids.pop(picked_idx)
            next_item = self._items_by_id.get(next_id)
            if not next_item or next_item.cancel_requested or next_item.status == "cancelled":
                if next_item:
                    next_item.status = "cancelled"
                    next_item.finished_at = time.time()
                    self._items_by_id.pop(next_id, None)
                    self._append_history(next_item)
                continue

            if next_item.id not in self._running_ids:
                self._running_ids.append(next_item.id)
            next_item.status = "running"
            next_item.queue_position = 0
            next_item.started_at = time.time()
            self._sort_waiting_ids()
            if next_item._turn_event is not None:
                next_item._turn_event.set()

        # If another task is still running, keep ws_manager reflecting it; otherwise mark idle
        try:
            from backend.app.api.ws import ws_manager

            if self._running_ids:
                rem = self._items_by_id.get(self._running_ids[0])
                if rem:
                    ws_manager.broadcast_sync(
                        "state",
                        {
                            "_from_task_queue": True,
                            "active_stage": rem.active_stage if rem.active_stage != "queued" else "ocr",
                            "active_model": rem.active_model,
                            "current_scan_id": rem.metadata.get("current_scan_id")
                            or rem.metadata.get("scan_id"),
                            "current_row_id": rem.metadata.get("current_row_id")
                            or rem.metadata.get("row_id"),
                            "current_sheet_id": rem.metadata.get("current_sheet_id")
                            or rem.metadata.get("sheet_id"),
                            "current_slovakiana_key": rem.metadata.get("current_slovakiana_key")
                            or rem.metadata.get("slovakiana_key"),
                            "current_sheet_label": rem.metadata.get("current_sheet_label")
                            or rem.metadata.get("sheet_label"),
                            "job_id": rem.metadata.get("job_id"),
                            "progress_pct": max(5, rem.progress_pct),
                            "message_sk": rem.message_sk,
                            "message_en": rem.message_en,
                        },
                    )
            elif not self._waiting_ids and ws_manager.latest_state.get("active_stage") not in (None, "idle"):
                ws_manager.broadcast_sync(
                    "state",
                    {
                        "_from_task_queue": True,
                        "active_stage": "idle",
                        "active_model": None,
                        "progress_pct": 100,
                        "message_sk": "Všetky úlohy vo fronte boli dokončené.",
                        "message_en": "All queued tasks have been completed.",
                    },
                )
        except Exception:
            pass

    def on_concurrency_limit_changed(self) -> None:
        self._cleanup_stale_loop_state()
        self._advance_next_waiting_task()
        self._broadcast_queue_sync()

    async def run_or_enqueue(
        self,
        coro_fn: Callable[[], Awaitable[Any]],
        *,
        task_type: str,
        category: str = "ocr",
        title_sk: str,
        title_en: str,
        provider: str | None = None,
        initial_stage: str = "ocr",
        active_model: str | None = None,
        target_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        priority: int = 10,
    ) -> Any:
        """
        If called from inside an already-running task on this async context, runs `coro_fn` inline.
        Otherwise registers a new task:
        - `freellm` tasks start immediately and concurrently without queueing.
        - `local` tasks run concurrently up to `settings.local_max_concurrent` (default 5) and queue beyond that.
        """
        if self.is_inside_active_task():
            return await coro_fn()

        item = self._register_task_item(
            task_type=task_type,
            category=category,
            title_sk=title_sk,
            title_en=title_en,
            provider=provider,
            initial_stage=initial_stage,
            active_model=active_model,
            target_url=target_url,
            metadata=metadata,
            priority=priority,
        )
        loop = asyncio.get_running_loop()
        bg_task = loop.create_task(
            self._execute_registered_item(item, coro_fn, initial_stage)
        )
        item._asyncio_task = bg_task
        return await asyncio.shield(bg_task)

    def enqueue_background(
        self,
        coro_fn: Callable[[], Awaitable[Any]],
        *,
        task_type: str,
        category: str = "ocr",
        title_sk: str,
        title_en: str,
        provider: str | None = None,
        initial_stage: str = "ocr",
        active_model: str | None = None,
        target_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        priority: int = 10,
    ) -> dict[str, Any]:
        """
        Synchronously registers a task (so it is immediately visible in `/api/debug/tasks`
        and `/api/status`) and schedules its execution in the background event loop.
        """
        item = self._register_task_item(
            task_type=task_type,
            category=category,
            title_sk=title_sk,
            title_en=title_en,
            provider=provider,
            initial_stage=initial_stage,
            active_model=active_model,
            target_url=target_url,
            metadata=metadata,
            priority=priority,
        )

        async def _runner() -> None:
            try:
                await self._execute_registered_item(item, coro_fn, initial_stage)
            except BaseException:
                pass

        try:
            loop = asyncio.get_running_loop()
            item._asyncio_task = loop.create_task(_runner())
        except RuntimeError:
            try:
                from backend.app.api.ws import ws_manager

                if ws_manager._loop is not None and ws_manager._loop.is_running():
                    ws_manager._loop.call_soon_threadsafe(
                        lambda: setattr(item, "_asyncio_task", ws_manager._loop.create_task(_runner()))  # type: ignore[union-attr]
                    )
            except Exception:
                pass

        return item.to_dict()

    def _parse_task_id(self, raw_id: int | str) -> int | None:
        if isinstance(raw_id, int):
            return raw_id
        s = str(raw_id).strip()
        if s.startswith("task-"):
            s = s[5:]
        try:
            return int(s)
        except ValueError:
            return None

    def cancel_task(self, raw_id: int | str) -> dict[str, Any] | None:
        """Cancels a queued or currently running task by ID."""
        self._cleanup_stale_loop_state()
        tid = self._parse_task_id(raw_id)
        if tid is None:
            return None
        item = self._items_by_id.get(tid)
        if not item:
            return None

        item.cancel_requested = True
        job_id = item.metadata.get("job_id")
        if job_id is not None:
            try:
                from backend.app.census.worker import request_job_pause

                request_job_pause(int(job_id))
            except Exception:
                pass

        if item.status == "queued":
            item.status = "cancelled"
            item.active_stage = "cancelled"
            item.finished_at = time.time()
            item.message_sk = f"Úloha bola zrušená vo fronte: {item.title_sk}"
            item.message_en = f"Task cancelled while queued: {item.title_en}"
            if tid in self._waiting_ids:
                self._waiting_ids.remove(tid)
            self._items_by_id.pop(tid, None)
            self._append_history(item)
            self._sort_waiting_ids()
            if item._turn_event is not None:
                item._turn_event.set()
            if item._asyncio_task is not None and not item._asyncio_task.done():
                item._asyncio_task.cancel()
            self._broadcast_queue_sync()
            return item.to_dict()

        if item.status == "running":
            item.message_sk = f"Zastavujem úlohu: {item.title_sk}…"
            item.message_en = f"Stopping task: {item.title_en}…"
            if item._asyncio_task is not None and not item._asyncio_task.done():
                item._asyncio_task.cancel()
            self._broadcast_queue_sync()
            return item.to_dict()

        return item.to_dict()

    def cancel_tasks_for_census_job(self, job_id: int) -> int:
        self._cleanup_stale_loop_state()
        matching_ids = [
            tid
            for tid, it in list(self._items_by_id.items())
            if it.metadata.get("job_id") == job_id
        ]
        for tid in matching_ids:
            self.cancel_task(tid)
        return len(matching_ids)

    def prioritize_task(self, raw_id: int | str) -> dict[str, Any] | None:
        """Moves a queued task to position #1 in the waiting queue."""
        self._cleanup_stale_loop_state()
        tid = self._parse_task_id(raw_id)
        if tid is None:
            return None
        item = self._items_by_id.get(tid)
        if not item or item.status != "queued":
            return item.to_dict() if item else None

        min_prio = min(
            (self._items_by_id[w].priority for w in self._waiting_ids if w in self._items_by_id),
            default=10,
        )
        item.priority = min_prio - 1
        self._sort_waiting_ids()
        pos = item.queue_position or 1
        item.message_sk = f"Vo fronte (#{pos}, prednostne) – čaká na uvoľnenie slotu…"
        item.message_en = f"Queued (#{pos}, prioritized) – waiting for slot…"
        self._broadcast_queue_sync()
        return item.to_dict()

    def clear_queued_tasks(self) -> int:
        """Cancels all waiting queued tasks without stopping currently running tasks."""
        self._cleanup_stale_loop_state()
        waiting_copy = list(self._waiting_ids)
        for tid in waiting_copy:
            self.cancel_task(tid)
        return len(waiting_copy)

    def clear_history(self) -> int:
        cnt = len(self._history)
        self._history.clear()
        self._broadcast_queue_sync()
        return cnt


task_queue = TaskQueueManager()
