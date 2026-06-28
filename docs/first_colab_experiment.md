# Первый базовый эксперимент в Colab

Этот эксперимент специально маленький. Он проверяет связку VS Code + Colab
kernel + GitHub checkout + локальные тесты + запись результатов + упаковка
результатов + сохранение результата в Google Drive. Веса моделей и vLLM server
пока не запускаются.

## Цель

Ожидаемый результат:

- репозиторий клонируется из GitHub в `/content/diffusion_gemma_bench`;
- notebook печатает точный `CODE_COMMIT_SHA`;
- unit-тесты проходят;
- `preflight` пишет hardware/runtime metadata;
- `backend-check` пишет `results/backend_capability.json`;
- `backend-smoke` пишет `results/backend_server_smoke.json`;
- `model-gate` пишет `results/model_gate.json`;
- `model-gate` создаёт `reports/experiment_summary.md` и
  `reports/experiment_summary.json`;
- `vllm-setup` пишет `results/vllm_setup.json`;
- `vllm-setup` создаёт отдельные summary-файлы
  `reports/experiment_summary_vllm-setup.md/json`;
- `artifact-discovery` пишет `results/artifact_discovery.json`;
- `artifact-discovery` создаёт отдельные summary-файлы
  `reports/experiment_summary_artifact-discovery.md/json`;
- placeholder `smoke` пишет `PENDING_COLAB_BACKEND_GATE`;
- `report` создаёт `reports/final_report.md`;
- маленькие артефакты упакованы в `results/runs/<RUN_ID>/`;
- run-директория и zip-копия сохранены в Google Drive:
  `MyDrive/diffusion_gemma_bench_results/<RUN_ID>`.

Это ещё не benchmark моделей. Это интеграционный smoke перед настоящим vLLM
capability gate.

## Разовая настройка GitHub

1. Держи код в ветке `main`.
2. Большие результаты, веса и логи не пушь в `main`.
3. GitHub используется для кода и воспроизводимого `CODE_COMMIT_SHA`.
4. Результаты экспериментов сохраняются отдельно в Google Drive.

## Переменные и секреты

Единый список имён хранится в `configs/experiment.env.example`.

Создай:

- `HF_TOKEN`: Hugging Face read token. Он нужен для проверки доступа к model
  repos и следующих модельных smoke-тестов. Токен должен иметь доступ к gated
  репозиториям, если Hugging Face требует acceptance.
Не коммить реальные значения токенов. Для локального запуска можно скопировать:

```bash
cp configs/experiment.env.example configs/experiment.env
```

и заполнить `configs/experiment.env`; этот файл игнорируется git. Перед commit запусти:

```bash
python scripts/check_no_secrets.py
```

В обычном Colab можно использовать Secrets с тем же именем: `HF_TOKEN`.
В VS Code Colab extension этот путь может быть недоступен.

Если ты запускаешь Colab через расширение VS Code и Colab Secrets недоступны,
используй `.env`-файл. Важно: локальный `configs/experiment.env` на Windows
не виден удалённому Colab runtime автоматически. Есть два рабочих варианта:

1. Вставить содержимое env-файла в опциональную ячейку `0a`, чтобы она создала
   `/content/experiment.env`, затем сразу очистить `ENV_TEXT` перед сохранением
   notebook или commit.
2. Любым доступным способом загрузить файл в runtime как `/content/experiment.env`.

Значения попадут только в `os.environ` текущего runtime и не печатаются.

Рекомендуемая схема хранения:

- GitHub `main`: код, конфиги, тесты, notebook, документация.
- Google Drive: packaged run-и под
  `MyDrive/diffusion_gemma_bench_results/<RUN_ID>/` и zip-копии.

## Запуск из VS Code + Colab Kernel

1. Открой `notebooks/01_colab_runner.ipynb` в VS Code.
2. Выбери Google Colab kernel.
3. В первой code cell задай:

   ```python
   EXPERIMENT = {
       "REPO_URL": "https://github.com/<owner>/diffusion_gemma_bench.git",
       "CODE_BRANCH": "main",
       "RESULTS_BRANCH": "bench-results",
       "PROFILE": "q6_core_native",
       "PHASE": "backend-check",
       "PROJECT_DIR": "/content/diffusion_gemma_bench",
       "VLLM_HOST": "127.0.0.1",
       "VLLM_PORT": "8000",
   }
   ```

4. Запусти ячейки до упаковки результата.
5. Проверь:

   - `unittest` прошёл;
   - `preflight.json` содержит GPU summary;
   - `backend_capability.json` создан;
   - `backend_server_smoke.json` содержит `BACKEND_SMOKE_PASSED`;
   - `model_gate.json` содержит `MODEL_GATE_PASSED` или явный список blockers;
   - `experiment_summary.md` содержит цель, задачи, таблицу метрик и критерии;
   - `run_manifest.json` содержит `git.commit_sha`;
   - `smoke_status.json` равен `PENDING_COLAB_BACKEND_GATE`;
   - `validation.ok` равен `True`.

## Сохранение результатов

После просмотра packaged run-директории запусти финальную ячейку notebook.

Она:

- монтирует Google Drive через `google.colab.drive`;
- копирует `results/runs/<RUN_ID>/` в
  `MyDrive/diffusion_gemma_bench_results/<RUN_ID>/`;
- создаёт zip рядом с этой директорией.

Если Google Drive недоступен, notebook сохранит копию локально в
`/content/diffusion_gemma_bench_results/`, чтобы run не потерялся в текущей
сессии.

Перед commit не сохраняй notebook outputs с авторизационными данными.

## Эксперимент 3: model-gate

Цель эксперимента: проверить, можно ли переходить от harness-smoke к реальным
модельным smoke-тестам без скачивания 26B весов.

Эксперимент проверяет:

- видимость GPU и объём VRAM;
- свободное место на диске;
- версии `torch`, `vllm`, `huggingface_hub`;
- наличие `HF_TOKEN`;
- доступ к Hugging Face model repositories;
- видимость ожидаемых файлов для выбранного профиля;
- доступ к MTP assistant repo для `G26-MTP`.

Критерии успеха:

- GPU доступна и имеет минимум 24 GiB VRAM для модельного smoke;
- свободно минимум 55 GiB диска;
- `vllm` импортируется;
- `HF_TOKEN` загружен;
- repo metadata доступна для всех целевых моделей;
- ожидаемые файлы выбранного профиля видны в Hugging Face metadata.

Артефакты:

- `results/model_gate.json`: машинно-читаемый gate-result;
- `reports/experiment_summary.md`: человекочитаемое резюме эксперимента;
- `reports/experiment_summary.json`: структурированная версия резюме.
- `reports/experiment_summary_model-gate.md/json`: стабильная копия резюме
  именно для model-gate, которая не перезаписывается следующими экспериментами.

## Эксперимент 4: vLLM setup gate

Цель эксперимента: установить или проверить `vllm` в текущем Colab runtime и
понять, можно ли после этого повторить `model-gate` без backend blocker-а.

Notebook-блок содержит флаги:

```python
RUN_EXPERIMENT_4 = True
INSTALL_VLLM = True
VLLM_PIP_SPEC = "vllm"
```

Если установка меняет базовые пакеты, Colab может попросить restart runtime.
В этом случае после restart запусти notebook сверху до Experiment 4 и повтори
блок с `INSTALL_VLLM = False`, чтобы только проверить уже установленную среду.

Критерии успеха:

- GPU доступна после установки;
- `torch` импортируется;
- `vllm` импортируется;
- `huggingface_hub` импортируется;
- повторный `model-gate` больше не содержит blocker `vllm_not_importable`.

Артефакты:

- `results/vllm_setup.json`;
- `reports/experiment_summary_vllm-setup.md`;
- `reports/experiment_summary_vllm-setup.json`;
- повторно обновлённый `results/model_gate.json`.

## Передача результатов из Google Drive в локальный workspace

После финальной Drive-ячейки notebook печатает `saved_dir` и `saved_zip`.
Для локального анализа скачай или синхронизируй run-директорию в:

```text
external_results/<RUN_ID>/
```

Содержимое `external_results/` игнорируется Git, кроме `README.md`. Это
безопасная рабочая зона для анализа артефактов без коммита больших результатов.

## Резюме прогонов 1-4

Фактические результаты по сохранённым run-директориям:

- `backend-smoke` на L4 прошёл: harness, тесты, локальный OpenAI-compatible
  server smoke, strict JSON и Drive-save работают.
- `model-gate` на A100 подтвердил, что GPU/диск подходят, но `vllm` ещё не был
  установлен, а текущие Hugging Face repo ids/filenames не подтверждаются.
- `vllm-setup` на RTX PRO 6000 Blackwell прошёл: `vllm` импортируется, GPU
  доступна, VRAM достаточно. Оставшийся blocker — не backend, а model artifact
  metadata: repo ids или expected filenames требуют проверки.

## Эксперимент 5: artifact discovery

Цель эксперимента: найти корректные Hugging Face repositories и видимые файлы
весов для `DG-Native`, `G26-AR`, `G26-MTP` без скачивания больших артефактов.

Почему это следующий шаг: после Experiment 4 backend готов (`vllm` импортируется),
но `model-gate` всё ещё падает на:

- `model_repo_access_failed`;
- `expected_model_file_missing`;
- `assistant_repo_access_failed`.

Эксперимент делает:

- поиск candidate repos через Hugging Face metadata;
- чтение списка файлов candidate repos без download;
- выбор best candidate per model;
- создание reviewable recommendation перед правкой `configs/models.yaml`.

Критерии успеха:

- `HF_TOKEN` загружен;
- `huggingface_hub` импортируется;
- для каждой модели найден хотя бы один доступный candidate repo;
- для каждого candidate repo видны файлы весов или конфигурации;
- expected filenames либо подтверждены, либо явно требуют правки конфига.

Артефакты:

- `results/artifact_discovery.json`;
- `reports/experiment_summary_artifact-discovery.md`;
- `reports/experiment_summary_artifact-discovery.json`.

## Ожидаемые статусы

`smoke_status.json` пока должен быть:

```text
PENDING_COLAB_BACKEND_GATE
```

Это значит, что путь кода работает, но модельный smoke ещё заблокирован следующим
инженерным этапом.

`backend_capability.json` может вернуть `BACKEND_CHECK_NEEDS_SETUP`, если:

- `vllm` ещё не установлен;
- `HF_TOKEN` ещё не задан;
- доступ к gated Hugging Face repos не подтверждён.

## Следующий инженерный этап

После этого интеграционного smoke нужно реализовать:

1. установку/проверку vLLM;
2. проверку Hugging Face access без логирования `HF_TOKEN`;
3. запуск лёгкого vLLM/OpenAI-compatible server на `127.0.0.1`;
4. health endpoint;
5. streaming TTFC check;
6. strict JSON/tool prompt check;
7. затем уже модельный smoke для `DG-Native`, `G26-AR`, `G26-MTP`.
