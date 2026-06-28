# Experiment 6: minimal model-load smoke

## Цель

Проверить, что подтвержденный в Experiment 5 GGUF-артефакт реально скачивается или берется из Hugging Face cache и загружается через vLLM в активном Colab runtime.

Это еще не benchmark и не сравнение качества. Эксперимент отвечает на вопрос: можно ли безопасно перейти от проверки metadata к реальной загрузке модели.

## Первая модель

Первой загружаем только `G26-AR`:

```text
model_id: G26-AR
repo_id: unsloth/gemma-4-26B-A4B-it-qat-GGUF
filename: gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
mode: без MTP assistant
```

Причина выбора: это baseline Gemma без DiffusionGemma-специфики и без MTP. Если базовый путь загрузки не работает, нет смысла усложнять эксперимент.

## Что делает код

Фаза `model-load-smoke`:

- читает `configs/models.yaml`;
- берет repo id и filename для выбранного target;
- проверяет наличие `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`;
- скачивает файл через `huggingface_hub.hf_hub_download` или использует cache;
- пробует создать `vllm.LLM` с `load_format="gguf"`;
- фиксирует download time, load time, размер артефакта, GPU/RAM/disk snapshot;
- пишет JSON и Markdown summary.

## Запуск в Colab

В ноутбуке `notebooks/01_colab_runner.ipynb` запусти ячейку Experiment 6.

Настройки по умолчанию:

```python
RUN_EXPERIMENT_6 = True
MODEL_LOAD_TARGETS = "G26-AR"
MODEL_LOAD_MAX_MODEL_LEN = 512
MODEL_LOAD_GPU_MEMORY_UTILIZATION = 0.82
MODEL_LOAD_ENABLE_DOWNLOAD = True
MODEL_LOAD_ENABLE_LOAD = True
INSTALL_CUDA13_RUNTIME_FOR_VLLM = False
CUDA13_RUNTIME_PIP_SPEC = "nvidia-cuda-runtime-cu13"
```

Эквивалентная CLI-команда:

```bash
python run.py \
  --profile q6_core_native \
  --phase model-load-smoke \
  --targets G26-AR \
  --max-model-len 512 \
  --gpu-memory-utilization 0.82 \
  --download \
  --load \
  --confirm-go
```

Флаг `--confirm-go` обязателен, потому что фаза может скачать большой файл и занять GPU memory.

## Если vLLM падает на `libcudart.so.13`

Ошибка вида:

```text
ImportError: libcudart.so.13: cannot open shared object file
```

означает, что vLLM установлен в варианте, ожидающем CUDA 13 runtime library, но динамический загрузчик Python-процесса ее не видит. Это не OOM и не проблема размера модели.

Не включай автоматическую установку `nvidia-cuda-runtime-cu13` как основной путь: в Colab/pip этот пакет может подтянуться как source-package `0.0.1` и упасть на сборке wheel. Поэтому по умолчанию:

```python
INSTALL_CUDA13_RUNTIME_FOR_VLLM = False
```

Фаза `model-load-smoke` перед импортом vLLM сама:

- ищет `nvidia/*/lib` внутри active site-packages;
- добавляет эти директории в `LD_LIBRARY_PATH`;
- пытается preload `libcudart.so.13` через `ctypes.CDLL`;
- записывает диагностику в `results/model_load_smoke.json` в поле `cuda_runtime`.

Если `cuda_runtime.libcudart_so_13_candidates` пустой, текущий vLLM wheel несовместим с CUDA runtime в active Colab environment. Следующий инженерный шаг - переустановить vLLM/torch в совместимой CUDA-сборке или выбрать backend, который поддерживает текущую CUDA runtime, а не менять `max_model_len`.

После такой правки повторный запуск обычно не скачивает модель заново: HF cache уже содержит GGUF-файл.

## Критерии успеха

Минимальный успех:

- `results/model_load_smoke.json` создан;
- `G26-AR_download.ok = true`;
- `G26-AR_load.ok = true`;
- статус `MODEL_LOAD_SMOKE_PASSED`.

Если загрузка vLLM упадет, это тоже полезный результат, но статус будет `MODEL_LOAD_SMOKE_NEEDS_REVIEW`. Тогда смотри:

- `models[0].download.error_type`;
- `models[0].load.error_type`;
- `models[0].load.traceback_tail`;
- `hardware.gpu`;
- `hardware.disk.free_gib`;
- `memory_before` / `memory_after`.

## Артефакты

После запуска должны появиться:

- `results/model_load_smoke.json`;
- `reports/experiment_summary_model-load-smoke.md`;
- `reports/experiment_summary_model-load-smoke.json`.

Финальная packaging/Drive-ячейка ноутбука заберет эти файлы в run-папку.

## Следующий шаг

Если `G26-AR` успешно загрузится, следующим повтором Experiment 6 можно проверить `DG-Native`:

```python
MODEL_LOAD_TARGETS = "DG-Native"
```

Если и `DG-Native` загрузится, переходим к Experiment 7: минимальная генерация на коротком prompt с измерением TTFC/e2e latency.
