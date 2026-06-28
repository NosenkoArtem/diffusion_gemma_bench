# Experiment 6: Minimal Model Load Smoke

- Phase: `model-load-smoke`
- Profile: `q6_core_native`
- Timestamp: `2026-06-28T20:35:59+00:00`

## Objective

Verify that the confirmed GGUF artifact can be downloaded/cached and loaded by vLLM before running generation benchmarks.

## Tasks

- Resolve the configured repo id and GGUF filename for the selected target.
- Download the artifact through Hugging Face cache, or reuse it if already cached.
- Instantiate a minimal vLLM engine with conservative context length.
- Record load time, artifact size, hardware snapshot, and any failure traceback.

## Metrics

| metric | value | status | note |
| --- | --- | --- | --- |
| status | MODEL_LOAD_SMOKE_NEEDS_REVIEW | MODEL_LOAD_SMOKE_NEEDS_REVIEW | hf_token_missing, huggingface_hub_not_importable, download_disabled, load_disabled, model_blocked |
| download_enabled | no | info |  |
| load_enabled | no | info |  |
| cuda_runtime_preloaded | no | info |  |
| gpu |  | info | `{"available": null, "error": "No module named 'torch'", "error_type": "ModuleNotFoundError"}` |
| disk_free_gib | 834.54 | info |  |
| G26-AR_status | BLOCKED | review |  |
| G26-AR_download_s |  | info |  |
| G26-AR_load_s |  | info |  |
| G26-AR_traceback_tail |  | info |  |
| G26-AR_artifact_bytes |  | info | gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf |

## Success Criteria

| criterion | passed | note |
| --- | --- | --- |
| HF token is loaded | no |  |
| Target artifact downloads or is already cached | no |  |
| At least one target model loads in vLLM | no |  |
| No model-load OOM or backend exception | yes |  |

## Conclusion

This was a dry run. Enable download/load in Colab when ready to consume disk/VRAM.

## Artifacts

- `results/model_load_smoke.json`
- `reports/experiment_summary_model-load-smoke.md`
- `reports/experiment_summary_model-load-smoke.json`
