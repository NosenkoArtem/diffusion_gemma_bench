# DiffusionGemma vs Gemma 4 Benchmark

Reproducible Colab Pro+ benchmark scaffold for comparing:

- `DG-Native`: DiffusionGemma 26B A4B IT with the native diffusion sampler.
- `G26-MTP`: Gemma 4 26B A4B IT with the official MTP assistant.
- `G26-AR`: Gemma 4 26B A4B IT without MTP, used only for attribution.

The primary comparison is always `DG-Native` vs `G26-MTP`. `G26-AR` explains the
effect of MTP; it is not the only practical baseline.

## Current Scope

This repository starts with the local harness that can be tested without model
weights:

- configuration files for quantization profiles, model artifacts, generation,
  MTP, and benchmark phases;
- deterministic MiniToolAgent tools and task parsing;
- full MiniToolAgent v1 seed with 60 deterministic tasks, 48 English and
  12 Russian;
- strict JSON/tool-call validation with no repair and no retry;
- preflight resource checks and profile selection;
- MTP tuning decision logic;
- speed and quality metric helpers;
- a report generator stub that turns collected JSON/JSONL results into Markdown
  and HTML;
- a Colab runner notebook that can execute repository smoke tests and phases.

Long-running model phases are intentionally gated. They should run only in Colab
after preflight confirms hardware, backend, model access, and MTP capability.

## Commands Required By The Test Plan

```bash
python run.py --profile auto --phase preflight
python run.py --profile auto --phase backend-check
python run.py --profile q6_core_native --phase smoke
python run.py --profile q6_core_native --phase pilot
python run.py --profile q6_core_native --phase core --confirm-go
python run.py --profile q8_calibration --phase quant-calibration
python run.py --profile q6_core_native --phase bfcl-lite --confirm-go
python run.py --profile q6_core_native --phase repeat-speed
python run.py --profile q6_core_native --phase report
```

## Local Developer Checks

The tests use only Python standard library modules, so they can run in a regular
terminal or in a notebook cell:

```bash
python -m unittest discover -s tests
```

For notebook use:

```python
import unittest
suite = unittest.defaultTestLoader.discover("tests")
unittest.TextTestRunner(verbosity=2).run(suite)
```

## VS Code + Colab Kernel Smoke Test

This is the preferred first integration test before downloading model weights.
The full checklist lives in `docs/first_colab_experiment.md`.

Before backend/model checks, create these secrets outside git:

- `HF_TOKEN`: Hugging Face read token with access to the required model repos.
- `GITHUB_TOKEN`: GitHub token only if Colab must push packaged result runs.

Variable names and non-secret defaults are documented in
`configs/experiment.env.example`. If you need a local file, copy it to
`configs/experiment.env`; that file is ignored by git.

1. Push this repository to GitHub.
2. In VS Code, open `notebooks/01_colab_runner.ipynb`.
3. Select a Google Colab kernel from the kernel picker.
4. Set `REPO_URL` in the first notebook cell.
5. Run the notebook cells through result packaging.
6. The expected first smoke status is `PENDING_COLAB_BACKEND_GATE`. That means
   the Colab kernel can clone and run the repo, but real vLLM/model smoke tests
   have not been enabled yet.

## Data Policy

- No secrets are written to logs, reports, JSONL, or notebook outputs.
- Raw results are appended after every request or task.
- Each phase creates or reuses a `run_id`; existing result files are not silently
  overwritten.
- Primary results must not mix backends, quant profiles, GPUs, or cold/warm
  caching tracks.

## Next Implementation Milestones

1. Add Colab package installation and Drive copy helpers.
2. Implement vLLM capability gate on the actual Colab runtime.
3. Add real smoke-test execution for `DG-Native`, `G26-AR`, and `G26-MTP`.
4. Run pilot/MTP tuning and inspect the go/no-go status.
5. Generate final reports from real raw results.
