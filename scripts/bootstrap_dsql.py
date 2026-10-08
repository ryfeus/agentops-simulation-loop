"""Bootstrap the Phase 3 DSQL schema, role mapping, grants, and demo state."""

import asyncio

from scripts.dsql_admin import bootstrap

if __name__ == "__main__":
    asyncio.run(bootstrap())
