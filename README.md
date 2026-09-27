# Avito candidate retrieval

Исследовательское решение задачи кандидатогенерации: для каждого поискового
запроса вернуть до 50 `item_id` из корпуса и максимизировать macro Recall@50.
Все модели работают локально, без внешних inference API.

**Рамка задачи.** Это только retrieval-стадия каскада: мы добываем 50
кандидатов для последующего ранжирования и сами их не переранжируем. Порядок
внутри 50 на метрику не влияет. Любая модель в этом пайплайне — retrieval-канал
или лёгкий fusion каналов, и весь путь должен укладываться в latency guardrail
(`config/latency_guardrails.json`, отчёт — [`reports/LATENCY.md`](reports/LATENCY.md)).

Краткое описание финального подхода, ошибок и полного воспроизведения находится
в [`SOLUTION.md`](SOLUTION.md), проверка воспроизводимости — в
[`reports/REPRODUCIBILITY.md`](reports/REPRODUCIBILITY.md).

## Быстрый запуск воспроизводимой попытки 3

Нужен [Git LFS](https://git-lfs.com): индекс BM25, rankings каналов и модель
fusion (~394 MB) хранятся в репозитории через LFS. GPU, Kaggle и сеть не нужны.

```bash
git lfs install
git clone https://github.com/M1r0-dev/Avito_test.git && cd Avito_test
# положить benchmark_queries.parquet и benchmark_items.parquet из архива задачи в dataset/
python -m venv .venv && source .venv/bin/activate && pip install -e .
python scripts/generate_answer.py --check   # ~1.5 мин на CPU
```

Скрипт строит пул кандидатов из сохранённых каналов (BM25, zero-shot и LoRA
USER-bge-m3), считает признаки learned fusion, применяет сохранённую модель
`models/learned_fusion_attempt3.cbm`, пишет `answer.csv`, проверяет формат
submission и sha256. `--check` падает, если файл отличается от отправленной
попытки 3 (`c24bf119…`, public `0.824331`). Упаковка нового лучшего решения
LoRA v2 выполняется после фиксации качества; его исследовательский путь —
notebooks 19 и 20. Если репозиторий склонирован без
LFS, файлы в `artifacts/` будут текстовыми указателями — выполните
`git lfs pull`.

## Публичные попытки

| # | Метод | Notebook | Commit | Offline Recall@50 | Public Recall@50 |
|---|---|---|---|---|---|
| 1 | CatBoost LTR: BM25 + fine-tuned dense + SPLADE + click history | 13 | `44e0dfa` | test `0.85954` | **0.698370** |
| 2 | RRF BM25 + zero-shot dense + fine-tuned dense, `1 / 0.75 / 1.25` | 14 | `a7ce7dc` | test-tail `0.86243`, test `0.83238` | **0.821129** |
| 3 | те же каналы, learned fusion (CatBoost по рангам и token overlap) вместо RRF | 17 | `7993fa6` | test-tail `0.90476`, test `0.84448` | **0.824331** |
| 4 | BM25 + zero-shot + LoRA v1 + LoRA v2, candidate selector | 19–20 | `51282ff` | test-tail `0.91005`, test `0.86120` | **0.837229** |
| 5 | BM25 + LoRA v2, candidate selector (абляция) | 19 | `1f58c03` | test-tail `0.88360`, test `0.85651` | **0.829627** |
| 6 | BM25 + zero-shot + LoRA v2, PU selector; без LoRA v1 | 24 | `55ac93d` | test-tail `0.91005`, test `0.86120` | **0.836972** |

Отказ от LTR и click history поднял публичный Recall@50 на `+0.12276`, а
разрыв offline→public сократился с `−0.161` до `−0.041`. Это согласуется с
гипотезой covariate shift из notebook 14, но holdout всё ещё оптимистичен.

Текущий `answer.csv` — попытка 6 (`27705d5…`), public `0.836972`. Она выбрана
как финальная инженерная конфигурация, хотя попытка 4 выше на `0.000257`:
попытка 6 сохраняет тот же offline Recall@50 (`0.91005` test-tail, `0.86120`
test), но полностью удаляет LoRA v1 и сокращает число тяжёлых dense query
encoder/search стадий с трёх до двух. Финальный retrieval использует BM25,
zero-shot BGE-M3 и LoRA v2; top-200 сжимается PU-bagged selector до top-50.
Все предыдущие файлы сохранены в `submissions/attempt_*.csv`.
Журнал с sha256 файлов:
[`reports/public_submissions.json`](reports/public_submissions.json).

## Порядок исследования

Notebooks — основной источник экспериментальных решений. В каждом описаны
гипотеза, метод, польза для задачи, ограничения и критерий принятия результата.

1. `notebooks/01_eda.ipynb` — качество данных, длины текстов, повторы и фильтры.
2. `notebooks/02_validation_design.ipynb` — единый стратифицированный holdout.
3. `notebooks/03_bm25_experiments.ipynb` — BM25, metadata channels и language ablation.
4. `notebooks/05_dense_rrf_experiments.ipynb` — русский embedder, fixed
   chunking, BM25+dense RRF и парные статистические тесты.
5. Если после RRF остаётся измеримый запас Recall@50 **и позволяют доступные
   вычислительные ресурсы**, отдельными notebooks проверяются SPLADE и ColBERT.
   Они не включаются в основной pipeline без статистически подтверждённого
   прироста на зафиксированном holdout.
6. `notebooks/07_supervised_ltr_experiments.ipynb` — supervised selection из
   широкого union; исходный принятый результат `0.84058` на test.
7. `notebooks/08_finetuned_dense_experiments.ipynb` — оценка leakage-safe LoRA
   domain adaptation русского USER-bge-m3 после Kaggle-прогона.
8. `notebooks/09_click_history_retrieval.ipynb` — char n-gram retrieval по
   leakage-safe click history; канал улучшает RRF, но отклонён внутри LTR.
9. `notebooks/10_history_ltr_experiments.ipynb`–`12_ltr_objective_experiments.ipynb`
   — отрицательные абляции расширения pool, history-features и ranking losses.
10. `notebooks/13_finetuned_dense_ltr_experiments.ipynb` — замена zero-shot
    dense на адаптированный канал; offline test Recall@50 `0.85954`, но
    публичная попытка показала сильный validation shift.
11. `notebooks/14_distribution_shift_robust_rrf.ipynb` — аудит shift и текущий
    submission без LTR/history: BM25 + zero-shot dense + fine-tuned dense RRF,
    выбранный на редких запросах; публичный Recall@50 `0.821129`.
12. `notebooks/17_light_candidate_selector.ipynb` — лёгкий selector попытки 3.
13. `notebooks/18_finetune_audit.ipynb` — аудит покрытия LoRA v1 и обоснование v2.
14. `notebooks/19_lora_v2_evaluation.ipynb` — channel/system evaluation LoRA v2.
15. `notebooks/20_lora_v2_rrf_vs_selector.ipynb` — финальное сравнение RRF и
    selector; selector принят и получил public Recall@50 `0.837229`.

SPLADE-stage использует русский checkpoint `naver/neuclir22-splade-ru` и
контролируемые абляции pruning, chunking и global/local retrieval. Лицензия
checkpoint — CC BY-NC-SA 4.0; это допустимо для данного исследования, но должно
учитываться при возможном коммерческом использовании.

До выбора лучшего эксперимента production-сервис не строится. Код в `src/` —
не сервис, а небольшие тестируемые исследовательские примитивы, общие для
notebooks и batch inference.

## Latency guardrail

Исследовательский warm guardrail на текущем ноутбуке: end-to-end `p95 ≤ 500 ms`
при `batch=1`. Целевой production stretch на GPU остаётся `p95 ≤ 100 ms`.
Время включает preprocessing, query encoding, все принятые retrieval-каналы,
фильтры, LTR features, CatBoost и fusion; startup, загрузка индексов и сеть
измеряются отдельно. Конфигурация лежит в `config/latency_guardrails.json`.

Первый формальный замер оптимизированного one-pass BM25: `p95=128.59 ms` на
Intel Core Ultra 7 155H, 500 запросов. Полный end-to-end guardrail пока имеет
статус `partial`: dense GPU и LTR будут добавлены после стабилизации fine-tuned
канала. Запуск: `PYTHONPATH=src python scripts/benchmark_latency.py`.

## Ключевые решения EDA

- `item_infm_params_text` и `item_description_raw` длинные, поэтому dense-поиск
  работает по fixed passages и агрегирует passage ranks обратно в уникальные items.
- Ненулевая категория безопасна как hard-filter; значение `0` означает отсутствие
  ограничения.
- Локация не является глобальным hard-filter: примерно 16.9% train positives
  находятся в другой локации. Используются global и local retrieval channels.
- Russian/Cyrillic-only фильтр отклонён на holdout: корпус уже на 99.87%
  кириллический, а hard-filter теряет объявления с латинскими брендами.

Подробности и числа: `reports/EDA.md`.

## Данные

Положите локально, не добавляя в Git:

```text
dataset/
├── train.parquet
├── benchmark_queries.parquet
└── benchmark_items.parquet
```

Raw Parquet задачи в Git не хранятся. Артефакты финального решения
(`artifacts/bm25/`, `artifacts/validation/`, rankings трёх каналов и
`models/learned_fusion_attempt3.cbm`) лежат в Git LFS; остальные индексы,
embeddings, Kaggle staging и промежуточные submissions игнорируются.
Отправленные на платформу файлы архивированы в `submissions/attempt_*.csv`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Полное воспроизведение из исходных данных (CPU, ~15 минут)

Без LFS-артефактов все CPU-этапы пересобираются из трёх parquet задачи, а
выходы GPU-этапов скачиваются с Kaggle. GPU-этапы выполнены публичными
Kaggle kernels:
[zero-shot dense](https://www.kaggle.com/code/m1r0tvorxc/avito-russian-dense-candidate-retrieval),
[SPLADE](https://www.kaggle.com/code/m1r0tvorxc/avito-russian-splade-candidate-retrieval),
[LoRA training](https://www.kaggle.com/code/m1r0tvorxc/avito-user-bge-m3-domain-adaptation),
[fine-tuned retrieval](https://www.kaggle.com/code/m1r0tvorxc/avito-finetuned-dense-retrieval).
Их выходы скачиваются без Kaggle-аккаунта и сверяются по sha256. Повторное
LoRA-обучение на GPU не бит-в-бит детерминировано, поэтому точное воспроизведение
отправленного файла опирается на сохранённые rankings.

```bash
PYTHONPATH=src python scripts/build_bm25.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_validation_design.ipynb
python scripts/fetch_public_kaggle_outputs.py            # --splade для notebooks 06–13
jupyter nbconvert --to notebook --execute --inplace notebooks/05_dense_rrf_experiments.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/14_distribution_shift_robust_rrf.ipynb
sha256sum answer.csv  # попытка 2: 7b4d257955029a8eb7d090a86add9fcd352546fcd91080aa4b4790fb1a95efcc
jupyter nbconvert --to notebook --execute --inplace notebooks/17_light_candidate_selector.ipynb
sha256sum answer.csv  # попытка 3: c24bf119dc311a5f333570f6fde19e55dfe40f25d578b0f347c2699388a44642
```

Notebook 05 материализует `artifacts/rankings/bm25_rankings.parquet` и
перезаписывает `answer.csv` двухканальным baseline; notebook 14 записывает
попытку 2, notebook 17 обучает learned fusion на dev, сохраняет модель и
записывает попытку 3. Обучение CatBoost детерминировано на эталонной машине
(повторный прогон дал тот же файл); на другом CPU надёжнее путь через
сохранённую модель (`scripts/generate_answer.py`).

## Воспроизведение текущего baseline

```bash
PYTHONPATH=src python scripts/build_bm25.py
jupyter nbconvert --to notebook --execute notebooks/02_validation_design.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_bm25_experiments.ipynb --inplace
```

На стратифицированном holdout filter-aware BM25 даёт `Recall@50 = 0.7504`,
USER-bge-m3 с четырьмя fixed chunks — `0.7681`, а настроенный только на dev
BM25+dense RRF — `0.8203`. На независимой половине holdout прирост RRF над
BM25 равен `+0.0613`, 95% paired bootstrap CI `[0.0441; 0.0791]`,
one-sided randomization `p < 0.00005`.

Oracle recall объединения BM25 и dense кандидатов на test равен `0.9409`, то
есть после RRF остаётся `0.1317` измеримого запаса. Поэтому следующий
эксперимент — SPLADE, затем, если позволяют вычислительные ресурсы, ColBERT.

SPLADE ablation выбрала `q32/d192`, четыре chunks и local-weight `2.0` только
на dev. Все эти инженерные решения подтвердились на test после Holm correction,
но добавление SPLADE к BM25+dense не подтвердилось: test Recall@50 вырос с
`0.80920` до `0.81199`, paired delta `+0.00279`, 97.5% bootstrap CI
`[-0.00673; 0.01223]`, one-sided randomization `p=0.26584`. Поэтому SPLADE не
заменяет текущий `answer.csv`. Перед дорогим ColBERT проверяются supervised
selection, дообученный bi-encoder и cross-encoder.

Следующий supervised LTR этап обучает CatBoostRanker выбирать 50 items из
top-200 union BM25+dense+SPLADE. Leakage-safe click history исключает все exact
validation query signatures. На test Recall@50 вырос с `0.80920` до `0.84058`:
paired delta `+0.03138`, 98.75% CI `[0.01054; 0.05279]`, `p=0.00025` при
скорректированном пороге `0.0125`. Результат принят и записан в `answer.csv`,
но до исследовательской цели `0.9` остаётся `0.05942`.

Leakage-safe LoRA дообучение USER-bge-m3 на 17 033 уникальных training items
подняло dense-only Recall@50 с `0.76281` до `0.79976`. Замена zero-shot dense
на fine-tuned dense внутри той же трёхканальной LTR-схемы улучшила текущий
лучший результат с `0.84058` до `0.85954`: paired delta `+0.01896`, 99.5% CI
`[0.00116; 0.03732]`, randomization `p=0.00175` при пороге `0.005`.
Эта offline-гипотеза была отправлена первой и получила публичный `0.698370`.
Notebook 14 показал, что 72.27% benchmark-текстов имеют train frequency `<=1`
против 14.89% в holdout, а 62.64% вообще не встречаются в train. Поэтому
текущий `answer.csv` заменён на робастный RRF без CatBoost/history. На
независимом tail-test он получил `0.86243` против `0.84656` у BM25+fine;
дельта `+0.01587`, но строгий 99.375% CI `[-0.02646; 0.06349]` пересекает ноль.
Это была внешняя проверка обоснованной shift-гипотезы. Публичный результат —
`0.821129`, то есть `+0.12276` к первой попытке. Прирост относится ко всей
замене LTR/history на RRF; отдельный вклад выбранных весов RRF на benchmark не
измерялся.

## Лицензии внешних моделей

- `deepvk/USER-bge-m3` — Apache-2.0, русский sentence encoder для semantic search.
- `intfloat/multilingual-e5-small` — MIT, лёгкий multilingual control baseline.
- `naver/neuclir22-splade-ru` — CC BY-NC-SA 4.0, русский learned-sparse
  retriever; исследован, но не принят в финальный ensemble.

Финальный список реально использованных моделей фиксируется в dense notebook и
run metadata.
