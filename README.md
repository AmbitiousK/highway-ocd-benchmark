# Highway OCD Benchmark

A small, self-contained harness to **compare overlapping community detection (OCD)
algorithms** — the Highway method, a set of baselines, and *your own* algorithm — on the
same graphs and the same metrics.

Built so a third party can drop in a new algorithm and get an apples-to-apples comparison
in a few lines. No hidden absolute paths, no notebook archaeology.

```python
import pickle
from benchmark import compare_algorithms

G = pickle.load(open("data/synthetic/lfr_N200/graph.gpickle", "rb"))
df = compare_algorithms(G)          # baselines + Highway, ground truth read from the graph
print(df)
```

---

## What you get

- **One call to benchmark everything:** `compare_algorithms(G)` runs all algorithms and
  returns a tidy `pandas.DataFrame` (status, runtime, #communities, metric panel).
- **One line to add your algorithm:** `register_algorithm("my_ocd", fn)` — then it appears
  in every comparison, scored identically.
- **The paper's metric panel:** `extended_modularity` (Q_ov, unsupervised) + `fri`,
  `cover_sim_czekanowski` (Dice), `fstar_wo` (F*), `onmi` (ONMI) when ground truth exists.
- **Highway as portable C++:** vendored C++17 source, compiled on demand — no prebuilt
  binary, no machine-specific path.

## Built-in algorithms (zero external dependencies)

| name | source |
|---|---|
| `highway` | vendored Highway C++ backend |
| `slpa`, `demon`, `kclique`, `walkscan`, `conga`, `congo`, `lais2`, `lfm` | [cdlib](https://cdlib.readthedocs.io) baselines |

> **Optional external baselines (COPRA, BigClam)** need third-party compiled binaries and
> are *not* built in. Enable one by registering a thin wrapper that shells out to your
> binary — see [Optional external baselines](#optional-external-baselines).

---

## Install

```bash
pip install -r requirements.txt
# To use Highway, also have cmake (>=3.10) and a C++17 compiler on PATH.
```

The Highway C++ backend builds automatically the first time you run `highway`. To build it
ahead of time:

```bash
python -m benchmark.highway.build_highway
```

---

## Add your own algorithm

Your algorithm only needs this contract:

```
fn(G: networkx.Graph) -> List[List[node]]      # a list of communities; may overlap
```

```python
import networkx as nx
from benchmark import compare_algorithms, register_algorithm

def my_ocd(G):
    # ... your method ...
    return [[1, 2, 3], [3, 4, 5]]               # node labels are the graph's own labels

register_algorithm("my_ocd", my_ocd)
compare_algorithms(G, algos=["highway", "slpa", "my_ocd"])
```

A ready-to-edit template lives in [`examples/add_your_algorithm.py`](examples/add_your_algorithm.py).

---

## One-click runs

```bash
# quick: score every algorithm on one graph, full 5-metric panel
python examples/run_synthetic.py

# notebook walkthrough
jupyter notebook examples/quickstart.ipynb
```

## Experiment suite (figures + tables)

The four paper experiments, each running **all algorithms live** over a graph
corpus and rendering the paper-style figure plus the numeric tables. Your
registered algorithm is included automatically.

```bash
python examples/run_experiments.py                       # all four, shipped corpora
python examples/run_experiments.py --only performance,overlap
python examples/run_experiments.py --algos highway,slpa,demon,lfm --timeout 120
python examples/run_experiments.py --only realworld --realworld /path/to/realworld_nx
```

| experiment | what it produces | figure | tables |
|---|---|---|---|
| **performance** | 5-metric panel vs mixing `muw` (LFR) / `xi` (ABCD+o²) | `performance_<bmk>_1x5.pdf` | best-baseline, seed dispersion |
| **scalability** | runtime vs graph size, log-y | `scalability_<bmk>_runtime_logy.pdf` | runtime by size bin |
| **overlap** | 4-metric panel vs overlap `eta`, with seed error bands | `overlap_<bmk>_1x4.pdf` | overlap stability |
| **realworld** | Q_ov + runtime grouped bars | `realworld_qov_runtime.pdf` | real-world main |

Outputs land in `results/experiments/{plots,tables}/` as **pdf + png** figures and
**csv + tex** tables (the LaTeX is paper-ready). Visual style (Highway in crimson,
large fonts, 1×N panels, error bands) is vendored from the paper's figure code, so
figures match. Metric computation (ONMI/F\*) dominates runtime — budget a few
seconds per graph; a hard per-algorithm `--timeout` bounds any pathological case.

```bash
# score one real-world graph directly (Q_ov + runtime)
python examples/run_realworld.py --graph /path/to/graph.gpickle --timeout 300
```

---

## Data

### Graph contract

A graph is a pickled `networkx.Graph`:

- edges may carry a `weight` attribute;
- **ground truth** (synthetic graphs) is stored per node as
  `G.nodes[v]["communities"] = [community_id, ...]` (a node may belong to several).

`compare_algorithms` reads ground truth from this attribute automatically. Pass
`ground_truth=None` to force unsupervised-only scoring (Q_ov), or pass your own cover.

### Shipped examples (`data/synthetic/`)

- `lfr_N200` — LFR benchmark, single-membership ground truth.
- `abcdo2_N200` — ABCD+o² benchmark, **overlapping** ground truth.

### Real-world data

The paper's real-world graphs (SNAP Amazon / DBLP / YouTube, top-5000-community subsets) are
large and not committed here. Obtain them from the
[SNAP collection](https://snap.stanford.edu/data/#communities), build a `networkx.Graph`,
pickle it, and point `run_realworld.py --graph` at it. Real-world graphs typically have no
ground truth, so only Q_ov + runtime are reported.

---

## Optional external baselines

COPRA and BigClam are strong baselines but ship as third-party compiled binaries. To include
one, register a wrapper that runs your binary and returns the communities:

```python
import subprocess, networkx as nx
from benchmark import register_algorithm

def copra(G):
    # write G to the format your COPRA build expects, run it, parse its output ...
    out = subprocess.run(["/path/to/copra", "graph.txt", "-v", "8"], ...)
    return parse_copra(out)          # -> List[List[node]]

register_algorithm("copra", copra)
```

---

## Layout

```
benchmark/
  __init__.py            # public API: compare_algorithms, register_algorithm, evaluate
  compare.py             # compare_algorithms(G, ...) -> DataFrame (+ optional timeout)
  algorithms.py          # registry + register_algorithm + cdlib baselines + highway
  metrics.py             # evaluate() + ground_truth_cover(): the metric panel
  metrics_impl.py        # frozen metric implementations (Q_ov, FRI, Dice, F*, ONMI)
  make_full_coverage.py  # coverage helper used by Q_ov
  highway/               # vendored Highway: python wrapper + C++17 source + build script
data/synthetic/          # small LFR + ABCD+o2 graphs with ground truth
examples/                # run_synthetic.py, run_realworld.py, add_your_algorithm.py, quickstart.ipynb
```

## License

BSD 2-Clause — see [LICENSE](LICENSE).
>>>>>>> b0fa463 (Initial commit: Highway OCD benchmark harness)
