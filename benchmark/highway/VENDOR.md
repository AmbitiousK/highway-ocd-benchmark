# Vendored Highway C++ (canonical)

Source: `~/Downloads/cdlib-add-highway-cpp/cdlib/algorithms/internal/`
(cdlib fork with the Highway C++ backend added; LICENSE = BSD-2 as in that repo).

Vendored here so `performance_module` is self-contained & independently deliverable
(mirrors `data_module` vendoring its generators):

  algorithms/
    highway.py            # portable wrapper: highway_nx(G, ...) → List[List[node]]
    highway_cpp/          # C++17 source (src/ + include/ + CMakeLists.txt)
      build/              # built on demand (gitignored, NOT committed)

Binary path is resolved **relative** to highway.py:
  highway_cpp/build/highway   (Path(__file__).parent / "highway_cpp" / "build" / "highway")

Build:  python -m benchmark.highway.build_highway
This replaces the scattered, now-missing `highway_modularity_final*/build/...` paths
that the frozen old `performance/algorithms_registry.py` pointed at (the cause of the
failed legacy real-world run).
