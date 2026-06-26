# Первый базовый эксперимент в Colab

Этот эксперимент специально маленький. Он проверяет связку VS Code + Colab
kernel + GitHub checkout + локальные тесты + запись результатов + упаковка
результатов + опциональный push. Веса моделей и vLLM server пока не запускаются.

## Цель

Ожидаемый результат:

- репозиторий клонируется из GitHub в `/content/diffusion_gemma_bench`;
- notebook печатает точный `CODE_COMMIT_SHA`;
- unit-тесты проходят;
- `preflight` пишет hardware/runtime metadata;
- `backend-check` пишет `results/backend_capability.json`;
- `backend-smoke` пишет `results/backend_server_smoke.json`;
- placeholder `smoke` пишет `PENDING_COLAB_BACKEND_GATE`;
- `report` создаёт `reports/final_report.md`;
- маленькие артефакты упакованы в `results/runs/<RUN_ID>/`;
- опциональный push коммитит run-директорию в `bench-results`.

Это ещё не benchmark моделей. Это интеграционный smoke перед настоящим vLLM
capability gate.

## Разовая настройка GitHub

1. Держи код в ветке `main`.
2. Большие результаты, веса и логи не пушь в `main`.
3. Для маленьких reviewable-результатов используй ветку `bench-results`.
4. Если токен GitHub был напечатан в notebook output или чате, отзови его и
   создай новый.

## Переменные и секреты

Единый список имён хранится в `configs/experiment.env.example`.

Создай:

- `HF_TOKEN`: Hugging Face read token. Он нужен для проверки доступа к model
  repos и следующих модельных smoke-тестов. Токен должен иметь доступ к gated
  репозиториям, если Hugging Face требует acceptance.
- `GITHUB_TOKEN`: GitHub token для push packaged results в `bench-results`.
  Нужен только если push выполняется из Colab.

Не коммить реальные значения токенов. Для локального запуска можно скопировать:

```bash
cp configs/experiment.env.example configs/experiment.env
```

и заполнить `configs/experiment.env`; этот файл игнорируется git. Перед commit запусти:

```bash
python scripts/check_no_secrets.py
```

В обычном Colab можно использовать Secrets с теми же именами: `HF_TOKEN`,
`GITHUB_TOKEN`. В VS Code Colab extension этот путь может быть недоступен.

Если ты запускаешь Colab через расширение VS Code и Colab Secrets недоступны,
используй `.env`-файл. Важно: локальный `configs/experiment.env` на Windows
не виден удалённому Colab runtime автоматически. Есть два рабочих варианта:

1. Вставить содержимое env-файла в опциональную ячейку `0a`, чтобы она создала
   `/content/experiment.env`, затем сразу очистить `ENV_TEXT` перед сохранением
   notebook или commit.
2. Любым доступным способом загрузить файл в runtime как `/content/experiment.env`.

Значения попадут только в `os.environ` текущего runtime и не печатаются.

Рекомендуемая схема веток:

- `main`: код, конфиги, тесты, notebook, документация.
- `bench-results`: packaged run-и под `results/runs/<RUN_ID>/`.

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
   - `run_manifest.json` содержит `git.commit_sha`;
   - `smoke_status.json` равен `PENDING_COLAB_BACKEND_GATE`;
   - `validation.ok` равен `True`.

## Push результатов

Включай push только после просмотра packaged run-директории.

1. Убедись, что `GITHUB_TOKEN` загружен в runtime: ячейка настроек должна показать
   `GITHUB_TOKEN_present: True`. Для VS Code Colab extension обычно это делается
   через `/content/experiment.env`.
2. Не вставляй token-backed URL в notebook cell.
3. Установи:

   ```python
   PUSH_RESULTS = True
   ```

4. Запусти финальную ячейку. Она сначала выполнит `--auth-check`, и только после
   успешной проверки создаст commit и push в `bench-results`.

Push-скрипт валидирует файлы перед commit. Он отклоняет веса моделей, zip-файлы,
большие файлы, логи и строки, похожие на секреты.

Для приватных репозиториев не печатай токены. Не коммить notebook outputs с
авторизационными данными.

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
