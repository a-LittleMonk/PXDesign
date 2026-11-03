# PXDesign on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-LittleMonk/PXDesign/blob/main/PXDesign_colab.ipynb)

A Colab notebook that runs the whole PXDesign pipeline — design binders against a
target with hotspots, then score them — on current Colab. Clean template: no outputs,
no run history, parameters at their defaults.

Built on [PXDesign](https://github.com/bytedance/PXDesign) and
[PXDesignBench](https://github.com/bytedance/PXDesignBench) by ByteDance — this repo
is a personal wrapper around their work, not a replacement for it, and not a
reimplementation of any part of it. See [Credit](#credit).

**This repo is the only current version.** Check
[the releases page](https://github.com/a-LittleMonk/PXDesign/releases) for the newest
number; the notebook prints its own version in the first cell, so any copy can be
checked against that page at a glance.

Colab opens a GitHub notebook **read-only**. Use *File > Save a copy in Drive* to get
a runnable one. That copy is a snapshot — it keeps the form values you type into it
(chains, sequence counts, filters), and it never picks up later fixes. Come back here
for those, which is also why the repo stays a clean template.

## What it does

PXDesign (Ren *et al.*, bioRxiv 2025) comes in two repos, and the notebook runs both.

**Design** — [PXDesign](https://github.com/bytedance/PXDesign) is the generator:
PXDesign-d, a diffusion model. Point it at a target structure, say which residues the
binder should aim at (`hotspots`) and how long it should be, and it invents backbones.
Upstream reports 17–82% nanomolar hit rates across six of seven targets.

**Evaluate** — [PXDesignBench](https://github.com/bytedance/PXDesignBench) judges
backbones, whether they came from the step above or from RFdiffusion or BindCraft. It

1. puts sequences on each backbone with **ProteinMPNN**,
2. folds those sequences back with **AlphaFold2** and **Protenix** (binder designs) or
   **ESMFold** (monomers), knowing only the sequence,
3. measures how far the refold drifts from the design, and how confident the folding
   model is about the interface,
4. writes one CSV row per sequence plus a summary, and counts how many designs pass
   each filter from the paper.

That second number is the one you want before ordering anything: whether a design's
own sequence folds back into the thing you designed.

| Task | Sequences from | Refolded by |
|---|---|---|
| `binder` | ProteinMPNN | AlphaFold2 (complex + binder alone), Protenix |
| `monomer` | ProteinMPNN | ESMFold |

Either half works alone — skip the design steps to score designs you already have, or
stop after them if you only want backbones.

The notebook changes none of this. It installs the stack, fetches weights, and calls
the `pxdesign` CLI and `pxdbench/run.py` with the same arguments upstream's own README
and demo scripts use. Every setting in every run cell is an upstream config key under
its own name, and the defaults are upstream's defaults.

### Hotspots

The generator takes them; the evaluator does not, and does not need to — by the time a
backbone is being scored, the binding site is fixed by its coordinates. Hotspot
residues are numbered by mmCIF `label_seq_id` (what Mol\* calls *Sequence ID*, not
*Auth ID*), and the design cell renders the parsed target and prints the residue each
number actually landed on, because getting this wrong is the easiest way to waste a
long run.

## Why this exists

Both repos install through `install.sh` (conda, CUDA >= 12.1) or Docker with
`--gpus all`. Neither applies on Colab, and the stack is pinned hard enough —
PyTorch 2.3.1, JAX 0.4.29, NumPy 1.26.3 — that installing it over Colab's own Python
breaks both it and Colab. So the notebook installs Python 3.11 into
`/content/pxdenv`, builds the pinned stack there, and runs both tools as subprocesses,
leaving Colab's Python alone.

One environment serves both: PXDesign's `install.sh` installs exactly what
PXDesignBench's does and then adds itself, so there is nothing to reconcile.

Three arguments differ from upstream's recommended command, and all three are
upstream's own documented fallback for a card without the hardware: `--dtype fp32`
instead of `bf16` (a T4 has no bfloat16 unit), `--use_deepspeed_evo_attention False`
(the kernel needs a CUTLASS build this environment does not have), and
`--use_fast_ln` following whether step 5 managed to compile the fused kernel. None of
the three changes what is computed — fused LayerNorm and DeepSpeed attention are
kernel optimisations, and fp32 is the more precise of the two dtypes, not the less.

Three details that are easy to get wrong and are handled for you:

- Upstream launches ProteinMPNN, AlphaFold2 and Protenix as subprocesses that call
  plain `python3`, so `python3` has to resolve to the isolated environment. It does.
- Colab exports `PYTHONPATH=/env/python`. Left alone, that leaks Colab's NumPy 2 and
  its own torch into the environment. It is replaced for every subprocess.
- Upstream's argument parser re-splits the command line on whitespace, so no path may
  contain a space. `My Drive` does. Everything therefore runs out of `/content`, and
  the notebook copies to and from Drive itself.

## Running it

Set the runtime to a GPU, then run steps 1–6 once and 7 onward as often as you like.

| Step | | Time |
|---|---|---|
| 1 | Check the GPU | — |
| 2 | Mount your Google Drive | — |
| 3 | Install Python 3.11 | ~1 min |
| 4 | Build the environment | ~12 min, ~12 GB |
| 5 | Compile the fused LayerNorm kernel | ~3 min |
| 6 | Download model weights | 1.5–9 GB |
| 7 | Describe the design target (crop, hotspots) | — |
| 8 | Generate binders with PXDesign-d | depends |
| 9 | Run the evaluation | depends |
| 10 | Results | — |
| 11 | View a prediction | — |
| 12 | Save results to Drive | — |

Steps 7 and 8 are the generator; skip them to evaluate designs you already have.
Step 9 reads step 8's output directly — set `inputs_from` to
`PXDesign designs (step 8)` with `demo_settings` unticked.

Step 4 checks itself and raises if NumPy got pulled up to 2.x, if torch is not 2.3.1,
if torch cannot see the GPU, if ColabDesign or PXDesign will not import, or if JAX
landed on CPU — that last one is the quiet killer, because AlphaFold2 on CPU looks
like a hang rather than an error. Step 6 raises if a weights file did not arrive. Both
failures are terminal: see *If something fails* in the notebook, and do not re-run the
cell in place.

Steps 8 and 9 print a filtered view of upstream's output by default: the metrics, the
progress bars, and anything that looks like a failure, without the per-call
bookkeeping its dependencies log. The unfiltered log is always written next to the
results, and unticking `quiet_output` prints everything as-is.

Step 5 is allowed to fail. It builds Protenix's fused LayerNorm kernel, which is a
speed optimisation over `torch.nn.LayerNorm` and nothing more; if it does not compile,
steps 8 and 9 fall back and run a little slower.

### A T4 is small for this

Generation is the expensive half. Upstream's own presets draw 100 (`preview`) or 500
(`extended`) backbones per run and their advice for a real campaign is 10000+, aiming
for 10–100 that pass both the Protenix and AF2-IG filters. The notebook defaults to 10
so a first run finishes. `bf16` needs an Ampere card or newer, so on a T4 `fp32` is
the only choice — step 1 says which you have. For production work upstream recommends
[their web server](https://protenix-server.com/), and they are right.

## Run all

Two things need you; everything else, including saving back to Drive, is unattended.

1. **Step 2 waits for a Google Drive authorisation popup.** Run all will sit there
   until you click through it.
2. **Steps 7–9 must already say what you want done.** Their defaults design against
   the bundled PD-L1 example and then evaluate the five bundled binder examples,
   ignoring whatever is in Drive. For your own work, fill in step 7's target and
   untick step 9's `demo_settings`.

**Run step 12 before closing the tab.** Colab deletes `/content` when the runtime
ends, and both the designs and the results live there until you copy them to Drive.

## Designing against your own target (steps 2 and 7)

Step 2 mounts your Drive and creates `My Drive/PXDesign/inputs`,
`My Drive/PXDesign/outputs` and `My Drive/PXDesign/tool_weights`. Put the target
structure in `inputs` — `.cif` is strongly preferred, since a `.pdb` gets converted
and may have its chain and residue ids reassigned on the way — then name it in step 7
along with the chain, the region to keep, the hotspots and the binder length.

Step 7 writes the YAML, runs upstream's `check-input` and `parse-target` on it, and
shows you the result. For a target with two chains that each need their own crop or
hotspots, edit the YAML it wrote (`/content/pxd_design/<task>/<task>.yaml`) and re-run
step 8.

An MSA of the **full-length** target sequence makes the Protenix confidence scores
meaningful, and cropping does not change that. Leave `msa_dir` empty and one is
searched online during step 8, which is slower and rate-limited; a folder holding
`pairing.a3m` and `non_pairing.a3m` skips it.

## Evaluating your own designs (steps 2 and 9)

An evaluation runs on a *folder*, not a single file: put the designs you want scored
into `inputs`, or a subfolder of it, and set `inputs_from` to `Google Drive folder`.

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

## The weights, and what they cost (step 6)

| | Size | Needed for |
|---|---|---|
| AlphaFold2 params | ~1.1 GB kept (5.3 GB downloaded) | binder |
| ESMFold | ~2.7 GB | monomer |
| ProteinMPNN | ~50 MB | monomer (binder uses ColabDesign's copy) |
| Chemical component dictionary | ~570 MB | everything |
| Protenix checkpoints | ~1 GB and up, automatic on first run | binder, generation |
| PXDesign-d checkpoint | automatic on first run | generation |

The chemical component dictionary is required by anything that reads a structure, so
it is always fetched — upstream's `download_tool_weights.sh` does the same. Step 7
reads it and cannot download it for itself, which is why it is not left to the runs.

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

## Reading the output (steps 8 and 10)

The generator's `pipeline` modes write their own ranked table, `summary.csv`, with
upstream's four filters as `AF2-IG-success`, `AF2-IG-easy-success`,
`Protenix-success` and `Protenix-basic-success` columns, next to the structures:

- `orig_designed/` — the backbone as PXDesign-d drew it. This is the ranked
  *selection*, not every sample. Upstream returns up to `max_success_return`
  successes and pads to `min_total_return` with failures, and the CLI sets both to
  `n_sample` — so failed designs are in this folder too. Read `summary.csv`, not the
  folder listing.
- `passing-AF2-IG-easy/`, `passing-Protenix-basic/` — for rows that cleared those
  filters, the *docked pose* from AF2 or Protenix rather than the original backbone.
  `chosen_struct_path` in the table says which structure upstream prefers per row.

`infer` mode writes raw samples to `<task>/seed_<n>/predictions/` and no table.
Thresholds are in [upstream's README](https://github.com/bytedance/PXDesign#32-understanding-metrics--filters-summarycsv)
and in the notebook's closing section.

The evaluation writes two files, both upstream's:

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

It is not a substitute for a real GPU. Everything runs, and the defaults are sized so
a first pass finishes on a T4, but the sample counts upstream recommends for work you
would act on are far beyond one Colab session.

It does not run PXDesign-h, the hallucination-based generator described in the paper;
only PXDesign-d, the diffusion model, is in upstream's repo.

It also accepts backbones from anywhere, so the evaluation half works just as well on
output from [RFdiffusion-GoogleColab](https://github.com/a-LittleMonk/RFdiffusion-GoogleColab)
or [BindCraft-GoogleColab](https://github.com/a-LittleMonk/BindCraft-GoogleColab).

It also does not do the Foldseek-based novelty and diversity post-processing upstream
ships (`pxdbench/scripts/postprocess_*.py`). Foldseek is a separate install and those
scripts are a separate, CPU-bound step; the pairwise TM-score diversity metric
(`eval_diversity`) is available in step 9 and does not need it.

## Credit

The generator is [PXDesign](https://github.com/bytedance/PXDesign) and the evaluation
is [PXDesignBench](https://github.com/bytedance/PXDesignBench), both by ByteDance,
installed from source and run unmodified. They stand on
[Protenix](https://github.com/bytedance/Protenix),
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN),
[AlphaFold2](https://github.com/google-deepmind/alphafold) via
[ColabDesign](https://github.com/sokrypton/ColabDesign), and
[ESMFold](https://huggingface.co/facebook/esmfold_v1). Cite those, not this notebook;
upstream's README lists the citations.

The AlphaFold2 parameters are released under CC BY-NC 4.0 — non-commercial use only.
That restriction travels with the weights regardless of how the surrounding code is
licensed. PXDesign and PXDesignBench are both Apache-2.0 — free for academic and
commercial use — with the exception of file headers that say otherwise.
