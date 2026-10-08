"""Small Harbor EC2 compatibility layer for HTTPS-only Ubuntu package access."""

from __future__ import annotations

from typing import override

from harbor.environments.ec2 import EC2Environment

_APT_HTTPS_SOURCES = r"""
set -e
for source in /etc/apt/sources.list \
  /etc/apt/sources.list.d/*.list \
  /etc/apt/sources.list.d/*.sources; do
  [ -f "$source" ] || continue
  sudo sed -i -E \
    -e 's#http://([A-Za-z0-9.-]*archive\.ubuntu\.com)#https://\1#g' \
    -e 's#http://security\.ubuntu\.com#https://security.ubuntu.com#g' \
    "$source"
done
"""


class HttpsBootstrapEC2Environment(EC2Environment):
    """Use Harbor's native EC2 environment after making Ubuntu sources TLS-only."""

    @override
    async def _bootstrap_docker(self) -> None:
        if self.bootstrap_docker:
            result = await self._ssh_exec(_APT_HTTPS_SOURCES, timeout_sec=60)
            if result.return_code != 0:
                raise RuntimeError(
                    "failed to configure HTTPS Ubuntu package sources before Docker bootstrap: "
                    f"{result.stdout} {result.stderr}"
                )
        await super()._bootstrap_docker()
