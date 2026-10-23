"""Entry point: ``python -m familysearch_mcp`` / the ``familysearch-mcp`` script."""

from __future__ import annotations

from .server import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
