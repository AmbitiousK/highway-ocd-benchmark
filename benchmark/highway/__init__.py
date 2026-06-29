# benchmark/highway/__init__.py
"""Vendored canonical Highway backend (Python wrapper + C++17 source).

Public API:
    highway_nx(G, ...)  -> List[List[node]]   # overlapping communities
    build()             -> Path               # compile the C++ backend on demand
    is_built()          -> bool

The compiled binary lives at ``highway_cpp/build/highway`` (built on demand, not
committed). See ``VENDOR.md`` for provenance.
"""
from __future__ import annotations

from .highway import highway_nx
from .build_highway import build, is_built, binary_path

__all__ = ["highway_nx", "build", "is_built", "binary_path"]
