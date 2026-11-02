# PXDesign on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-LittleMonk/PXDesign/blob/main/PXDesign_colab.ipynb)

A Colab notebook that runs PXDesignBench on current Colab. Clean template — no
outputs, no run history, parameters at their defaults.

Built on [PXDesignBench](https://github.com/bytedance/PXDesignBench) by ByteDance —
this repo is a personal wrapper around their work, not a replacement for it, and not
a reimplementation of any part of it. See [Credit](#credit).

**This repo is the only current version.** Check
[the releases page](https://github.com/a-LittleMonk/PXDesign/releases) for the newest
number; the notebook prints its own version in the first cell, so any copy can be
checked against that page at a glance.

Colab opens a GitHub notebook **read-only**. Use *File > Save a copy in Drive* to get
a runnable one. That copy is a snapshot — it keeps the form values you type into it
(chains, sequence counts, filters), and it never picks up later fixes. Come back here
for those, which is also why the repo stays a clean template.

## What it does

PXDesignBench is the evaluation half of PXDesign (Ren *et al.*, bioRxiv 2025 — the
full citation is in
[upstream's README](https://github.com/bytedance/PXDesignBench#-citing-related-work)):
it does not generate backbones, it judges them. Give it a folder of designed structures and it

1. puts sequences on each backbone with **ProteinMPNN**,
2. folds those sequences back with **AlphaFold2** and **Protenix** (binder designs) or
   **ESMFold** (monomers), knowing only the sequence,
3. measures how far the refold drifts from the design, and how confident the folding
   model is about the interface,
4. writes one CSV row per sequence plus a summary, and counts how many designs pass
   each filter from the paper.

That is the number you want before ordering anything: whether a design's own sequence
folds back into the thing you designed.

| Task | Sequences from | Refolded by |
|---|---|---|
| `binder` | ProteinMPNN | AlphaFold2 (complex + binder alone), Protenix |
| `monomer` | ProteinMPNN | ESMFold |

The notebook changes none of this. It installs the stack, fetches weights, and calls
`pxdbench/run.py` or `pxdbench/run_monomer.py` with the same arguments upstream's demo
scripts pass. Every setting in the run cell is an upstream config key under its own
name, and the defaults are upstream's defaults.

## Why this exists

Upstream installs through `install.sh` (conda, CUDA >= 12.1) or Docker with
`--gpus all`. Neither applies on Colab, and the stack is pinned hard enough —
PyTorch 2.3.1, JAX 0.4.29, NumPy 1.26.3 — that installing it over Colab's own Python
breaks both it and Colab. So the notebook installs Python 3.11 into
`/content/pxdenv`, builds the pinned stack there, and runs the evaluation as a
subprocess, leaving Colab's Python alone.

Three details that are easy to get wrong and are handled for you:

- Upstream launches ProteinMPNN, AlphaFold2 and Protenix as subprocesses that call
  plain `python3`, so `python3` has to resolve to the isolated environment. It does.
- Colab exports `PYTHONPATH=/env/python`. Left alone, that leaks Colab's NumPy 2 and
  its own torch into the environment. It is replaced for every subprocess.
- Upstream's argument parser re-splits the command line on whitespace, so no path may
  contain a space. `My Drive` does. Everything therefore runs out of `/content`, and
  the notebook copies to and from Drive itself.

## Running it

Set the runtime to a GPU, then run cells 1–7 in order.

| Cell | | Time |
|---|---|---|
| 1 | Check the GPU | — |
| 2 | Mount your Google Drive | — |
| 3 | Install Python 3.11 | ~1 min |
| 4 | Build the environment | ~12 min, ~12 GB |
| 5 | Compile the fused LayerNorm kernel | ~3 min |
| 6 | Download model weights | 1–8 GB |
| 7 | Run the evaluation | depends |
| 8 | Results | — |
| 9 | View a prediction | — |
| 10 | Save results to Drive | — |

Cell 4 checks itself and raises if NumPy got pulled up to 2.x, if torch is not 2.3.1,
if torch cannot see the GPU, or if JAX landed on CPU — that last one is the quiet
killer, because AlphaFold2 on CPU looks like a hang rather than an error. Cell 6
raises if a weights file did not arrive. Both failures are terminal: see *If something
fails* in the notebook, and do not re-run the cell in place.

Cell 5 is allowed to fail. It builds Protenix's fused LayerNorm kernel, which is a
speed optimisation over `torch.nn.LayerNorm` and nothing more; if it does not compile,
cell 7 leaves `LAYERNORM_TYPE` unset and runs a little slower.

## Run all

Two things need you; everything else, including saving back to Drive, is unattended.

1. **Cell 2 waits for a Google Drive authorisation popup.** Run all will sit there
   until you click through it.
2. **Cell 7's fields must already say what you want evaluated.** Its defaults
   reproduce `binder_eval_demo.sh` on the five bundled PD-L1 binder examples and
   ignore whatever is in Drive. To evaluate your own designs, untick
   `demo_settings` and fill in the rest.

**Run cell 10 before closing the tab.** Colab deletes `/content` when the runtime
ends, and the results live there until you copy them to Drive.

## Evaluating your own designs (cells 2 and 7)

Cell 2 mounts your Drive and creates `My Drive/PXDesign/inputs`,
`My Drive/PXDesign/outputs` and `My Drive/PXDesign/tool_weights`. An evaluation runs
on a *folder*, not a single file: put the designs you want scored into `inputs`, or a
subfolder of it, and set `inputs_from` to `Google Drive folder`.

The folder has to be homogeneous. All `.pdb` or all `.cif` (`is_mmcif` says which),
all against the same target, with the same chain naming — upstream infers the target
chains once and asserts that every file agrees. For binder designs, name the binder
chain in `binder_chains`; everything else is treated as target. One binder chain only,
which is upstream's limit, not the notebook's.

`orig_seqs_json` is worth understanding before a long run. It is optional metadata
about the target: the full uncropped sequence, and a precomputed MSA. Leave it empty
and Protenix asks a public MSA server for the target's alignment on every fresh run —
a few minutes, rate-limited, and occasionally down. Point it at a file in upstream's
format (copy `examples/orig_seqs_test.json`) and that step disappears. Since the
target is the same across a design campaign, computing it once pays for itself
immediately.

## The weights, and what they cost (cell 6)

| | Size | Needed for |
|---|---|---|
| AlphaFold2 params | ~1.1 GB kept (5.3 GB downloaded) | binder |
| ESMFold | ~2.7 GB | monomer |
| ProteinMPNN | ~50 MB | monomer (binder uses ColabDesign's copy) |
| Protenix checkpoint + CCD | ~1 GB, automatic on first run | binder |

The AlphaFold2 archive is a single 5.3 GB tar and there is no partial download of it.
The notebook pipes it straight into `tar` and keeps only the model-1 files, which is
all any default setting uses — upstream's `af2.model_ids` is `[0]`. Take all five only
if you raise `af2_model_ids`.

**By default nothing is written to your Drive.** The weights go to Colab's own disk
and disappear with the runtime, which means step 6 downloads them again next session.
Tick `cache_in_drive` to keep them in `My Drive/PXDesign/tool_weights`: it then looks
there first and uses whatever is already present, and only downloads — and saves back
— what is missing. Worth ticking once you know you will run this more than a couple of
times; the cell prints which of the two it is doing.

## Reading the output (cell 8)

Two files, both upstream's:

- `sample_level_output.csv` — one row per designed sequence.
- `summary_output.json` — averages over the run, plus `*_success.count`: how many
  designs passed each named filter.

The filters (`af2_easy`, `af2_opt`, `ptx_mini`, …) and their thresholds live in
`pxdbench/pxd_configs/eval.py` and are not touched here. The notebook's closing
section explains the columns that matter; briefly, `i_pTM` up, `i_pAE` down,
`af2_binder_pred_design_rmsd` down, `bound_unbound_RMSD` down.

Designs failing these filters is normal and expected — they are the paper's
thresholds, and real campaigns pass a small fraction. `af2_easy` is the lenient set.

## What this notebook is not

It does not **generate** designs. PXDesign's generative half is not in
PXDesignBench, so there is nothing here that invents a binder. Feed it backbones from
RFdiffusion, BindCraft, or wherever else — see
[RFdiffusion-GoogleColab](https://github.com/a-LittleMonk/RFdiffusion-GoogleColab) and
[BindCraft-GoogleColab](https://github.com/a-LittleMonk/BindCraft-GoogleColab) — and
this scores them.

It also does not do the Foldseek-based novelty and diversity post-processing upstream
ships (`pxdbench/scripts/postprocess_*.py`). Foldseek is a separate install and those
scripts are a separate, CPU-bound step; the pairwise TM-score diversity metric
(`eval_diversity`) is available in cell 7 and does not need it.

## Credit

The evaluation is [PXDesignBench](https://github.com/bytedance/PXDesignBench) by
ByteDance, installed from source and run unmodified. It stands on
[Protenix](https://github.com/bytedance/Protenix),
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN),
[AlphaFold2](https://github.com/google-deepmind/alphafold) via
[ColabDesign](https://github.com/sokrypton/ColabDesign), and
[ESMFold](https://huggingface.co/facebook/esmfold_v1). Cite those, not this notebook;
upstream's README lists the citations.

The AlphaFold2 parameters are released under CC BY-NC 4.0 — non-commercial use only.
That restriction travels with the weights regardless of how the surrounding code is
licensed. PXDesignBench itself is Apache-2.0, with the exception of the file headers
that say otherwise.
