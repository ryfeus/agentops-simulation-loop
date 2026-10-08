"""Restore the canonical disputed-refund state in Phase 3 DSQL."""

import asyncio

from scripts.dsql_admin import reset_demo

if __name__ == "__main__":
    asyncio.run(reset_demo())
