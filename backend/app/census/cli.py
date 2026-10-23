"""
Command-line interface for bulk harvesting, downloading, processing, and indexing
Slovakiana Census Sheets (Sčítacie hárky 1930/1939/1940).

Usage examples:
  python -m backend.app.census.cli facets
  python -m backend.app.census.cli harvest-catalog --years 1930 1940 --max-pages 10
  python -m backend.app.census.cli download-scans --municipality-key 7236 --limit 5 --ocr
  python -m backend.app.census.cli reindex
"""

from __future__ import annotations

import argparse
import asyncio
import json
from sqlmodel import Session

from backend.app.census.slovakiana_client import slovakiana_client
from backend.app.census.worker import (
    download_and_process_single_census_sheet,
    run_catalog_harvest_job,
    run_download_and_pipeline_job,
)
from backend.app.db.models import CensusDownloadJob
from backend.app.db.session import engine, init_db
from backend.app.search.fts import rebuild_all_fts


async def _main_async() -> None:
    init_db()
    parser = argparse.ArgumentParser(description="Slovakiana Census Bulk Harvester & Pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("facets", help="Fetch and cache Slovakiana years, districts, and municipalities")

    p_harvest = sub.add_parser("harvest-catalog", help="Run Tier-1 catalog harvester into SQLite + FTS5")
    p_harvest.add_argument("--years", type=int, nargs="+", default=[1930, 1940])
    p_harvest.add_argument("--category-keys", nargs="*", default=[])
    p_harvest.add_argument("--search", type=str, default="")
    p_harvest.add_argument("--max-pages", type=int, default=None)
    p_harvest.add_argument("--page-size", type=int, default=200)
    p_harvest.add_argument("--only-unblackened", action="store_true")

    p_dl = sub.add_parser("download-scans", help="Run Tier-2 scan downloader & optional OCR pipeline")
    p_dl.add_argument("--years", type=int, nargs="+", default=[1930, 1940])
    p_dl.add_argument("--category-keys", nargs="*", default=[])
    p_dl.add_argument("--search", type=str, default="")
    p_dl.add_argument("--limit", type=int, default=20)
    p_dl.add_argument("--ocr", action="store_true", help="Run fine-tuned census OCR & extraction pipeline")
    p_dl.add_argument("--no-mcp", action="store_true", help="Skip FamilySearch MCP cross-matching")

    p_sheet = sub.add_parser("process-sheet", help="Download and process a single CensusSheet by DB ID")
    p_sheet.add_argument("sheet_id", type=int)
    p_sheet.add_argument("--no-ocr", action="store_true")

    sub.add_parser("reindex", help="Rebuild SQLite FTS5 full-text search index")

    args = parser.parse_args()

    if args.cmd == "facets":
        facets = await slovakiana_client.fetch_facets(only_unblackened=False, force_refresh=True)
        print(
            json.dumps(
                {
                    "years_count": len(facets.get("years", [])),
                    "districts_count": len(facets.get("districts", [])),
                    "municipalities_count": len(facets.get("municipalities", [])),
                },
                indent=2,
            )
        )
    elif args.cmd == "harvest-catalog":
        with Session(engine) as session:
            job = CensusDownloadJob(
                job_type="catalog_harvest",
                filter_json=json.dumps(
                    {
                        "years": args.years,
                        "category_keys": args.category_keys,
                        "search_text": args.search,
                        "include_blackened": not args.only_unblackened,
                    }
                ),
                status="running",
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id or 1
        res = await run_catalog_harvest_job(job_id, max_pages=args.max_pages, page_size=args.page_size)
        print(json.dumps(res, indent=2))
    elif args.cmd == "download-scans":
        with Session(engine) as session:
            job = CensusDownloadJob(
                job_type="full_pipeline" if args.ocr else "image_download",
                filter_json=json.dumps(
                    {
                        "years": args.years,
                        "category_keys": args.category_keys,
                        "search_text": args.search,
                        "include_blackened": True,
                        "limit_sheets": args.limit,
                        "auto_ocr": args.ocr,
                        "run_mcp": not args.no_mcp,
                    }
                ),
                status="running",
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id or 1
        res = await run_download_and_pipeline_job(job_id)
        print(json.dumps(res, indent=2))
    elif args.cmd == "process-sheet":
        res = await download_and_process_single_census_sheet(
            args.sheet_id,
            run_pipeline=not args.no_ocr,
        )
        print(json.dumps(res, indent=2, default=str))
    elif args.cmd == "reindex":
        with Session(engine) as session:
            res = rebuild_all_fts(session)
            print(json.dumps(res, indent=2))


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
