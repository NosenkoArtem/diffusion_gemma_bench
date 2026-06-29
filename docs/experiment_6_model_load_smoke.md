# Experiment 6: combined llama.cpp model-load smoke

## Цель

Проверить, что обе подтвержденные GGUF-модели реально скачиваются или берутся из Hugging Face cache и загружаются через один и тот же `llama.cpp` backend:

- `G26-AR`: Gemma 4 26B A4B QAT GGUF;
- `DG-Native`: DiffusionGemma 26B A4B GGUF.

Это объединяет прежние шаги 6b и 6c. Мы больше не проверяем модели раздельными ручными повторами: один запуск должен дать сопоставимый результат по обоим артефактам, одним runtime, одной командной форме и одинаковыми sampling-настройками.

## Почему меняем путь после vLLM smoke

Предыдущий `model-load-smoke` был полезной диагностикой, но показал, что текущий vLLM не является рабочим основным backend для выбранного Gemma 4 QAT GGUF:

```text
Field 'num_key_value_heads' expected int, got list
```

Это не OOM, не проблема токена и не проблема скачивания. Ошибка указывает на несовместимость парсинга model config в текущем vLLM для данного GGUF. Поэтому vLLM оставляем как вторичный диагностический путь для native/safetensors-сценариев, а основной GGUF-трек переводим на `llama.cpp`.

## Что делает новая фаза

Фаза `llama-load-smoke`:

- читает `configs/models.yaml`;
- берет repo id и filename для `G26-AR` и `DG-Native` под выбранный профиль;
- проверяет наличие `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`;
- проверяет наличие `llama-cli` или использует путь из `--llama-cli-path`;
- скачивает веса через `huggingface_hub.hf_hub_download` или использует cache;
- запускает короткий `llama-cli` smoke на каждом файле;
- фиксирует download/load time, return code, stdout/stderr tail, размер артефакта, GPU/RAM/disk snapshot;
- пишет JSON и Markdown summary для отчета.

## Модели

Первый объединенный прогон:

```text
targets: G26-AR,DG-Native
backend: llama.cpp / llama-cli
profile: q6_core_native
context: 512
generated tokens: 8
temperature: 1.0
top_p: 0.95
top_k: 64
```

`G26-MTP` не удаляется из плана, но переносится после базовой llama.cpp-линии. Сначала нужно доказать, что обе основные GGUF-модели загружаются и генерируют минимальный ответ в одном backend.

## Запуск в Colab

В notebook `notebooks/01_colab_runner.ipynb` запусти ячейку Experiment 6.

Настройки по умолчанию:

```python
RUN_EXPERIMENT_6 = True
EXPERIMENT_6_PHASE = "llama-load-smoke"
LLAMA_LOAD_TARGETS = "G26-AR,DG-Native"
LLAMA_LOAD_MAX_CONTEXT = 512
LLAMA_LOAD_TIMEOUT_S = 300
LLAMA_CLI_PATH = None
```

Эквивалентная CLI-команда:

```bash
python run.py \
  --profile q6_core_native \
  --phase llama-load-smoke \
  --targets G26-AR,DG-Native \
  --max-model-len 512 \
  --llama-timeout-s 300 \
  --download \
  --load \
  --confirm-go
```

Флаг `--confirm-go` обязателен, потому что фаза может скачивать большие файлы и занимать GPU memory.

Если `llama-cli` не найден, результат будет `LLAMA_LOAD_SMOKE_NEEDS_REVIEW` с причиной `llama_cli_missing`. Тогда нужно установить или собрать `llama.cpp` с CUDA в Colab и повторить ту же фазу, не меняя модельные настройки.

## Критерии успеха

Минимальный успех:

- `results/llama_load_smoke.json` создан;
- `G26-AR_download.ok = true`;
- `DG-Native_download.ok = true`;
- `G26-AR_load.ok = true`;
- `DG-Native_load.ok = true`;
- статус `LLAMA_LOAD_SMOKE_PASSED`.

Если загрузка одной модели падает, это все равно полезный результат, но переход к generation benchmark блокируется. Нужно смотреть:

- `models[*].load.error_category`;
- `models[*].load.stderr_tail`;
- `models[*].load.stdout_tail`;
- `hardware.gpu`;
- `hardware.disk.free_gib`;
- `memory_before` / `memory_after`.

## Артефакты

После запуска должны появиться:

- `results/llama_load_smoke.json`;
- `reports/experiment_summary_llama-load-smoke.md`;
- `reports/experiment_summary_llama-load-smoke.json`.

Финальная packaging/Drive-ячейка notebook заберет эти файлы в run-папку.

## Следующий шаг

Если `llama-load-smoke` проходит для обеих моделей, переходим к Experiment 7: минимальная генерация через `llama-server` OpenAI-compatible endpoint с измерением TTFC, E2E latency и проверкой streaming/non-streaming поведения.
