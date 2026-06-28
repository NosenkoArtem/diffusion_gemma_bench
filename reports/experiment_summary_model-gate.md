# Experiment 3: Model Access and Backend Feasibility Gate

- Phase: `model-gate`
- Profile: `q6_core_native`
- Timestamp: `2026-06-28T13:55:26+00:00`

## Objective

Check whether the current Colab Pro+ runtime can move from harness smoke tests to real model smoke tests without downloading 26B weights.

## Tasks

- Capture runtime GPU, disk, package, and git metadata.
- Check Hugging Face token presence and model repository metadata access.
- Check expected profile artifact visibility for DG-Native, G26-AR, and G26-MTP.
- Check vLLM importability as the primary backend gate.
- Record blockers and next-step decision for the model-smoke phase.

## Metrics

| metric | value | status | note |
| --- | --- | --- | --- |
| gate_status | MODEL_GATE_NEEDS_SETUP | MODEL_GATE_NEEDS_SETUP | no_gpu_detected, vllm_not_importable, huggingface_hub_not_importable, hf_token_missing |
| gpu_available | no | blocked |  |
| gpu_total_vram_gib |  | info | model smoke floor is 24 GiB |
| disk_free_gib | 836.11 | ok | minimum 55 GiB |
| vllm_importable | no | blocked |  |
| hf_token_present | no | blocked |  |
| DG-Native_repo_access |  | blocked | unsloth/diffusiongemma-26B-A4B-it-GGUF |
| DG-Native_expected_file |  | unknown_or_blocked | diffusiongemma-26B-A4B-it-Q6_K.gguf |
| G26-AR_repo_access |  | blocked | unsloth/gemma-4-26B-A4B-it-GGUF |
| G26-AR_expected_file |  | unknown_or_blocked | gemma-4-26B-A4B-it-UD-Q6_K.gguf |
| G26-MTP_repo_access |  | blocked | unsloth/gemma-4-26B-A4B-it-GGUF |
| G26-MTP_expected_file |  | unknown_or_blocked | gemma-4-26B-A4B-it-UD-Q6_K.gguf |

## Success Criteria

| criterion | passed | note |
| --- | --- | --- |
| GPU is visible and has at least 24 GiB VRAM | no | `{"available": false}` |
| At least 55 GiB disk is free | yes | `{"free_gib": 836.11, "min_required_gib": 55}` |
| vLLM is importable | no |  |
| HF token and model repo metadata are accessible | no |  |
| Expected profile artifacts are visible | yes |  |

## Conclusion

Install or repair vLLM in the Colab runtime, then rerun model-gate.

## Artifacts

- `results/model_gate.json`
- `reports/experiment_summary.md`
- `reports/experiment_summary.json`
