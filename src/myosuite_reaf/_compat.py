# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform compatibility shims for upstream ``mujoco_playground``.

``playground==1.0.1`` imports ``fcntl`` unconditionally at module import time in
``mujoco_playground._src.sim.asset_loader``, which raises ``ModuleNotFoundError``
on Windows (where the POSIX ``fcntl`` stdlib module does not exist). Calling
:func:`ensure_windows_fcntl_shim` before importing ``mujoco_playground``
installs a minimal ``msvcrt``-backed ``flock`` shim on ``win32`` and is a no-op
on POSIX platforms.
"""

from __future__ import annotations

import os
import sys
import types
from typing import IO, Any


def ensure_windows_fcntl_shim() -> bool:
    """Install a minimal ``fcntl`` module shim on Windows if missing.

    Returns:
        ``True`` if the shim was installed, ``False`` if ``fcntl`` was already
        available or the current platform is not Windows.
    """
    if sys.platform != "win32" or "fcntl" in sys.modules:
        return False

    import msvcrt

    shim = types.ModuleType("fcntl")
    shim.LOCK_SH = 1  # type: ignore[attr-defined]
    shim.LOCK_EX = 2  # type: ignore[attr-defined]
    shim.LOCK_NB = 4  # type: ignore[attr-defined]
    shim.LOCK_UN = 8  # type: ignore[attr-defined]

    def _flock(fd_or_file: int | IO[Any], operation: int) -> None:
        fd = fd_or_file if isinstance(fd_or_file, int) else fd_or_file.fileno()
        os.lseek(fd, 0, os.SEEK_SET)
        if operation & shim.LOCK_UN:  # type: ignore[attr-defined]
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            mode = (
                msvcrt.LK_NBLCK
                if (operation & shim.LOCK_NB)  # type: ignore[attr-defined]
                else msvcrt.LK_LOCK
            )
            try:
                msvcrt.locking(fd, mode, 1)
            except OSError:
                pass

    shim.flock = _flock  # type: ignore[attr-defined]
    sys.modules["fcntl"] = shim
    return True
