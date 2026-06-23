# First Colab Experiment

This first experiment is intentionally small. It proves that VS Code, the Colab
kernel, GitHub checkout, local tests, result writing, result packaging, and
optional result push all work before model weights or vLLM servers are involved.

## Goal

Expected outcome:

- repository clones from GitHub into `/content/diffusion_gemma_bench`;
- the notebook prints the exact `CODE_COMMIT_SHA`;
- unit tests pass;
- `preflight` writes hardware/runtime metadata;
- placeholder `smoke` writes `PENDING_COLAB_BACKEND_GATE`;
- report generation writes `reports/final_report.md`;
- small artifacts are packaged under `results/runs/<RUN_ID>/`;
- optional push can commit that run directory to `bench-results`.

This is not yet a model benchmark. It is the integration smoke test before the
real vLLM capability gate.

## One-Time GitHub Setup

1. Create a GitHub repository for the code.
2. Push the local repository to `main`.
3. Keep large outputs out of `main`.
4. Use `bench-results` for small, reviewable result artifacts.

Recommended branch policy:

- `main`: source code, configs, tests, notebook, docs.
- `bench-results`: packaged result runs under `results/runs/<RUN_ID>/`.

## VS Code + Colab Kernel Run

1. Open `notebooks/01_colab_runner.ipynb` in VS Code.
2. Select a Google Colab kernel.
3. In the first code cell, set:

   ```python
   REPO_URL = "https://github.com/<owner>/diffusion_gemma_bench.git"
   CODE_BRANCH = "main"
   RESULTS_BRANCH = "bench-results"
   PROFILE = "q6_core_native"
   PHASE = "smoke"
   ```

4. Run all cells through "Package Result Run".
5. Confirm:

   - `unittest` passes;
   - `preflight.json` includes a GPU summary;
   - `run_manifest.json` includes `git.commit_sha`;
   - `smoke_status.json` is `PENDING_COLAB_BACKEND_GATE`;
   - `validation.ok` is `True`.

## Optional Result Push

Only after inspecting the packaged result directory:

1. Authenticate GitHub inside the Colab runtime.
2. Set:

   ```python
   PUSH_RESULTS = True
   ```

3. Run the final notebook cell.

The push script validates files before commit. It rejects model weights, zip
files, large files, logs, and secret-looking tokens.

For private repositories, avoid printing tokens. Prefer Colab secrets or an
interactive GitHub authentication method. Do not commit notebook outputs that
contain authentication details.

## Expected First Status

The first smoke status should be:

```text
PENDING_COLAB_BACKEND_GATE
```

That means the code path works, but model smoke tests are still blocked by the
next implementation step: vLLM capability gate and model artifact checks.

## Next Engineering Step

After this integration smoke passes, implement:

1. vLLM install/version check.
2. Hugging Face access check without logging `HF_TOKEN`.
3. backend health endpoint check on `127.0.0.1`.
4. streaming response check.
5. strict JSON/tool prompt check.
6. MTP startup verification for `G26-MTP`.
