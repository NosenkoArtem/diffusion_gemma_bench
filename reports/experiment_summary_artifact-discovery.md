# Experiment 5: Model Artifact Discovery

- Phase: `artifact-discovery`
- Profile: `q6_core_native`
- Timestamp: `2026-06-28T13:55:24+00:00`

## Objective

Find correct Hugging Face repositories and visible weight files for the configured models without downloading large artifacts.

## Tasks

- Search Hugging Face metadata for DiffusionGemma/Gemma candidate repositories.
- Inspect visible files for candidate repos without downloading weights.
- Select best repo/file candidates for DG-Native, G26-AR, and G26-MTP.
- Produce a reviewable recommendation before editing model config.

## Metrics

| metric | value | status | note |
| --- | --- | --- | --- |
| discovery_status | ARTIFACT_DISCOVERY_NEEDS_REVIEW | ARTIFACT_DISCOVERY_NEEDS_REVIEW | hf_token_missing, huggingface_hub_not_importable, model_search_failed, candidate_repo_missing |
| hf_token_present | no | blocked |  |
| huggingface_hub_version |  | blocked |  |
| DG-Native_best_repo |  | missing | expected_file_visible=None |
| DG-Native_candidate_count | 0 | info | hf_token_missing |
| G26-AR_best_repo |  | missing | expected_file_visible=None |
| G26-AR_candidate_count | 0 | info | hf_token_missing |
| G26-MTP_best_repo |  | missing | expected_file_visible=None |
| G26-MTP_candidate_count | 0 | info | hf_token_missing |

## Success Criteria

| criterion | passed | note |
| --- | --- | --- |
| HF token is loaded | no |  |
| Hugging Face search can run | no |  |
| Each configured model has at least one accessible candidate | no |  |
| Expected filenames are confirmed or require explicit config update | yes |  |

## Conclusion

Review Hugging Face search queries and model naming; no accessible candidate repo was found for at least one model.

## Artifacts

- `results/artifact_discovery.json`
- `reports/experiment_summary_artifact-discovery.md`
- `reports/experiment_summary_artifact-discovery.json`
