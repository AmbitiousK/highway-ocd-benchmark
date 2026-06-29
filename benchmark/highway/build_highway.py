# benchmark/highway/build_highway.py
# -*- coding: utf-8 -*-
"""
Build the vendored Highway C++ backend (portable; mirrors data_module.tools.build_lfr).

  python -m benchmark.highway.build_highway

Compiles highway_cpp/ (CMake, C++17, -O3, optional OpenMP) to
  benchmark/highway/highway_cpp/build/highway
The binary is built on demand and **not committed** (see highway_cpp/.gitignore).

No hardcoded absolute paths: everything is resolved relative to this file, so the
module stays self-contained / independently deliverable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CPP = _HERE / "highway_cpp"
_BUILD = _CPP / "build"
_BIN = _BUILD / "highway"


def binary_path() -> Path:
    return _BIN


def is_built() -> bool:
    return _BIN.is_file()


def build(force: bool = False) -> Path:
    if is_built() and not force:
        print(f"[build_highway] already built: {_BIN}")
        return _BIN
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake not found on PATH; install CMake to build Highway.")

    _BUILD.mkdir(parents=True, exist_ok=True)
    print(f"[build_highway] configuring: {_CPP}")
    subprocess.run(["cmake", "-S", str(_CPP), "-B", str(_BUILD)], check=True)
    print("[build_highway] compiling (-O3)...")
    subprocess.run(["cmake", "--build", str(_BUILD), "--config", "Release", "-j"], check=True)
    if not _BIN.is_file():
        raise RuntimeError(f"build finished but binary missing: {_BIN}")
    print(f"[build_highway] OK: {_BIN}")
    return _BIN


if __name__ == "__main__":
    build(force="--force" in sys.argv)
