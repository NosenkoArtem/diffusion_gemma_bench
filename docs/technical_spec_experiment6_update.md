# Корректировка ТЗ: объединенный Experiment 6 на llama.cpp

## Причина корректировки

Experiment 6 через vLLM выполнил полезную диагностическую роль: Hugging Face token, доступ к артефакту и скачивание Gemma 4 QAT GGUF были проверены, но загрузка в vLLM завершилась ошибкой совместимости model config:

```text
Field 'num_key_value_heads' expected int, got list
```

Этот результат трактуется не как нехватка VRAM и не как ошибка notebook, а как backend incompatibility текущего vLLM с выбранным GGUF-артефактом Gemma 4 A4B QAT.

## Новое решение

Для GGUF-трека основной backend переносится с vLLM на `llama.cpp`. vLLM остается вторичным диагностическим путем для будущих native/safetensors-проверок, но не является блокирующим backend для ближайшего сравнения GGUF-моделей.

Прежние шаги 6b и 6c объединяются в один Experiment 6:

```text
phase: llama-load-smoke
targets: G26-AR,DG-Native
backend: llama.cpp / llama-cli
profile: q6_core_native
context: 512
generated_tokens: 8
temperature: 1.0
top_p: 0.95
top_k: 64
```

## Обновленные критерии успеха Experiment 6

- `llama-cli` доступен в Colab runtime или задан через `LLAMA_CLI_PATH`;
- `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` загружен из `experiment.env`;
- `G26-AR` GGUF скачивается или находится в HF cache;
- `DG-Native` GGUF скачивается или находится в HF cache;
- обе модели проходят короткий `llama-cli` load/generation smoke одной командной формой;
- создаются `results/llama_load_smoke.json`, `reports/experiment_summary_llama-load-smoke.md` и `reports/experiment_summary_llama-load-smoke.json`.

## Обновленный порядок дальнейших экспериментов

1. `llama-load-smoke`: объединенная проверка загрузки `G26-AR` и `DG-Native`.
2. `llama-server` smoke: проверка OpenAI-compatible endpoint, streaming/non-streaming, TTFC и E2E latency.
3. Мини-бенчмарк: короткий набор speed/tool/JSON задач для обеих моделей.
4. Core benchmark: расширенный прогон после успешного мини-бенчмарка.
5. MTP/speculative ветка для Gemma 4 рассматривается после стабильной baseline-линии на `llama.cpp`.

## Обновление scope

Первичное сравнение для ближайших экспериментов:

```text
DG-Native vs G26-AR
```

MTP остается исследовательской веткой и не удаляется из финального ТЗ, но временно не является обязательным условием для начала практического сравнения DiffusionGemma и Gemma 4 на Colab Pro+.
