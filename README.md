# Avito candidate retrieval

Решение задачи кандидатогенерации: для каждого поискового запроса вернуть до
50 `item_id` из корпуса и максимизировать macro Recall@50. Все модели работают
локально, без внешних inference API.

**Рамка задачи.** Это только retrieval-стадия каскада: мы добываем 50
кандидатов для последующего ранжирования и сами их не переранжируем. Порядок
внутри 50 на метрику не влияет. Любая модель в этом пайплайне — retrieval-канал
или лёгкий selector поверх каналов, и весь путь должен укладываться в latency
guardrail (`config/latency_guardrails.json`, отчёт —
[`reports/LATENCY.md`](reports/LATENCY.md)).

**Описание решения для проверяющего** — [`SOLUTION.md`](SOLUTION.md): данные и
признаки, модели, проверка качества, найденные ошибки, latency. Происхождение
каждого артефакта и проверка воспроизводимости —
[`reports/REPRODUCIBILITY.md`](reports/REPRODUCIBILITY.md).

## Быстрый запуск финального решения (попытка 7)

Нужен [Git LFS](https://git-lfs.com): индекс BM25, rankings каналов и модели
selector хранятся в репозитории через LFS. GPU, Kaggle и сеть не нужны.

```bash
git lfs install
git clone https://github.com/M1r0-dev/Avito_test.git && cd Avito_test
# положить benchmark_queries.parquet и benchmark_items.parquet из архива задачи в dataset/
python -m venv .venv && source .venv/bin/activate && pip install -e .
python scripts/generate_answer.py --check   # ~2 мин на CPU
```

Скрипт берёт сохранённые rankings BM25, zero-shot и LoRA v2 USER-bge-m3,
строит RRF top-200 пул, считает признаки selector
(`src/avito_retrieval/pu_selector.py`), применяет три PU-bagged модели,
обученные Recall@50-лоссом (`models/recall50_seed{41,42,43}_round{1,2}.cbm`,
`src/avito_retrieval/recall_lambda.py`), пишет `answer.csv`, проверяет формат
submission и sha256. `--check` падает, если файл отличается от отправленной
попытки 7 (`994ef6f3…`, public `0.840831`). Модели переобучаются командой
`python scripts/train_pu_selector.py` и дают тот же sha. Предыдущие финалы
воспроизводятся отдельно: попытка 6 —
`python scripts/generate_answer.py --objective yetirank --check --output /tmp/a6.csv`,
попытка 3 — `python scripts/generate_attempt3_answer.py --check`. Если
репозиторий склонирован без LFS, файлы в `artifacts/` будут текстовыми
указателями — выполните `git lfs pull`.

## Публичные попытки

| # | Метод | Notebook | Commit | Offline Recall@50 | Public Recall@50 |
|---|---|---|---|---|---|
| 1 | CatBoost LTR: BM25 + fine-tuned dense + SPLADE + click history | 13 | `44e0dfa` | test `0.85954` | **0.698370** |
| 2 | RRF BM25 + zero-shot dense + fine-tuned dense, `1 / 0.75 / 1.25` | 14 | `a7ce7dc` | test-tail `0.86243`, test `0.83238` | **0.821129** |
| 3 | те же каналы, learned fusion (CatBoost по рангам и token overlap) вместо RRF | 17 | `7993fa6` | test-tail `0.90476`, test `0.84448` | **0.824331** |
| 4 | BM25 + zero-shot + LoRA v1 + LoRA v2, candidate selector | 19–20 | `51282ff` | test-tail `0.91005`, test `0.86120` | **0.837229** |
| 5 | BM25 + LoRA v2, candidate selector (абляция) | 19 | `1f58c03` | test-tail `0.88360`, test `0.85651` | **0.829627** |
| 6 | BM25 + zero-shot + LoRA v2, PU selector; без LoRA v1 | 24 | `55ac93d` | test-tail `0.91005`, test `0.86120` | **0.836972** |
| 7 | попытка 6 + Recall@50-лосс внутри PU bags | 25 | `8f88b21` | test-tail `0.91534`, test `0.86650` | **0.840831** |

Отказ от LTR и click history (попытка 1 → 2) поднял публичный Recall@50 на
`+0.12276`, а разрыв offline→public сократился с `−0.161` до `−0.041`: это
подтвердило гипотезу covariate shift из notebook 14. Holdout при этом остаётся
оптимистичным — у поздних попыток public ниже offline на ~0.026.

Текущий `answer.csv` — попытка 7 (`994ef6f3…`), public **`0.840831`**, лучшая
из семи. Это система попытки 6 (BM25 + zero-shot USER-bge-m3 + LoRA v2, RRF
top-200, три PU-bagged CatBoost), в которой selector обучен собственным
Recall@50-лоссом (LambdaMART с весами `|ΔRecall@50|`, notebook 25). Offline
прирост к попытке 6 `+0.0053` на test и test-tail был незначим (p `0.15` /
`0.50`); публично попытка 7 стала лучшей из семи (`+0.003859` к попытке 6),
но и эта разница в пределах шума, поэтому доказанным улучшением её считать
нельзя — только совпадением знака с offline. Попытка 6 в своё
время заменила попытку 4 при том же offline Recall@50, убрав LoRA v1 (на один
568M-encoder меньше в online-пути). Все отправленные файлы сохранены в
`submissions/attempt_*.csv`, журнал с sha256 —
[`reports/public_submissions.json`](reports/public_submissions.json).

## Порядок исследования

Notebooks — основной источник экспериментальных решений. В каждом описаны
гипотеза, протокол, статистический тест и решение; отрицательные результаты
сохранены так же, как положительные.

1. `01_eda` — данные, длины текстов, повторы, фильтры (`reports/EDA.md`).
2. `02_validation_design` — стратифицированный holdout 2 452 запросов, dev/test.
3. `03_bm25_experiments` — BM25; отдельный канал локации поднял Recall@50 с
   `0.316` до `0.750`; фильтр «только кириллица» отклонён.
4. `05_dense_rrf_experiments` — USER-bge-m3: 4 passages вместо одного вектора
   (`+0.037`), BM25 + dense RRF (`+0.061`); `multilingual-e5-small` — контроль.
5. `06_splade_experiments` — SPLADE: `+0.003`, не доказано — отклонён.
6. `07_supervised_ltr_experiments` — CatBoost LTR над union каналов:
   `0.809 → 0.841` на test.
7. `08_finetuned_dense_experiments` — LoRA v1 (Kaggle 08a/08b): `+0.021`.
8. `09`–`12` — канал click history, расширение пула, history-признаки,
   альтернативные LTR-лоссы: отрицательные абляции.
9. `13_finetuned_dense_ltr_experiments` — LTR с LoRA v1: offline `0.860` →
   попытка 1, public `0.698` — переобучение на частые запросы.
10. `14_distribution_shift_robust_rrf` — аудит shift (72% benchmark — редкие
    запросы), RRF без LTR/history, выбор на редких запросах → попытка 2.
11. `15_recall_headroom_diagnostics` — разбор промахов: 76% — ошибки отбора
    (relevant был в пуле каналов) → нужен selector, а не новые каналы.
12. `17_light_candidate_selector` — лёгкий CatBoost selector без history → попытка 3.
13. `18_finetune_audit` — аудит покрытия LoRA v1, обоснование LoRA v2.
14. `19_lora_v2_evaluation` — LoRA v2: канал `+0.027`, selector на 4 каналах →
    попытка 4; абляция BM25 + LoRA v2 → попытка 5.
15. `20_lora_v2_rrf_vs_selector` — selector против RRF на тех же каналах: selector принят.
16. `21_multiview_retrieval_experiments` — multi-view dense: tail не улучшен, отклонено.
17. `22_bm25_lora_v2_pool_compression` — в пуле top-200 уже 92% relevant:
    узкое место — сжатие 200 → 50.
18. `23_pu_recall_selector` — PU-bagging против ложных негативов в разметке.
19. `24_pu_selector_best_pool` — PU selector на BM25 + zero-shot + LoRA v2 без
    LoRA v1: тот же recall, на один encoder меньше → попытка 6.
20. `25a`–`25c` — собственный Recall@50-лосс (встроенного Recall@k в CatBoost
    нет: `StochasticFilter` игнорирует `metric=RecallAt`): test `+0.0053`,
    offline незначимо; отправлен последней попыткой — public `0.840831`,
    лучший из семи (`+0.0039` к попытке 6, в пределах шума). **Это финальное
    решение (попытка 7).**
21. `26a`–`26d` — попытка улучшить лосс запасом на train (cutoff 10 вместо 50):
    OOF dev `+0.007`, test-tail `−0.0053` — отклонено, в финал не вошло.
22. `27`–`29` — online-путь под latency guardrail (точный CPU-поиск, ONNX
    Runtime encoder, финальный selector): Recall@50 не хуже offline
    (non-inferiority, граница `−0.005`); latency — `reports/LATENCY.md`.

Код в `src/` — не сервис, а тестируемые примитивы, общие для notebooks,
воспроизведения ответа и замера latency.

## Latency guardrail

Warm guardrail на ноутбуке: end-to-end `p95 ≤ 500 ms` при `batch=1`. Целевой
production stretch на GPU — `p95 ≤ 100 ms` (не измерялся). Время включает
preprocessing, query encoding, все retrieval-каналы, фильтры, признаки,
CatBoost и fusion; startup и загрузка индексов измеряются отдельно.
Конфигурация — `config/latency_guardrails.json`.

Финальное решение (попытка 7) проходит guardrail на Intel Core Ultra 7 155H
строго по контракту (warm-up 25, 500 запросов, CPU): **p50 176 ms, p95 357 ms,
p99 682 ms**. Для этого query encoder переведён на ONNX Runtime fp32 без
квантизации, три retrieval-ветки выполняются параллельно, потоки распределены
явно, а объекты, загруженные при старте, заморожены для сборщика мусора
(`gc.freeze`). Recall@50 online-пути совпадает с offline (notebooks 27–29).
Это замер времени, а не рабочий сервис: векторы passages zero-shot не
сохранены, поэтому в замере zero-shot ищет по случайной матрице той же формы
(время от значений не зависит), а сданный `answer.csv` собран из сохранённых
rankings — подробности в `SOLUTION.md`, раздел 8.
Все прогоны, включая неудачные, — в [`reports/LATENCY.md`](reports/LATENCY.md).

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
├── train.parquet              # нужен только для пересборки с нуля
├── benchmark_queries.parquet
└── benchmark_items.parquet
```

Raw Parquet задачи в Git не хранятся. Всё, что нужно для воспроизведения
финального ответа, лежит в Git LFS: `artifacts/bm25/`, `artifacts/validation/`,
rankings каналов (`artifacts/rankings/`, `artifacts/dense_kaggle/`,
`artifacts/finetuned_dense_kaggle/`, `artifacts/finetuned_v2_kaggle/`) и модели
(`models/recall50_*.cbm` — попытка 7, `models/pu_selector_*.cbm` — попытка 6,
`models/learned_fusion_attempt3.cbm` — попытка 3). Векторы эмбеддингов, ONNX-
экспорты и промежуточные файлы игнорируются. Отправленные на платформу файлы
архивированы в `submissions/attempt_*.csv`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'           # + '.[latency]' для замера latency online-пути
pytest
```

## Полное воспроизведение из исходных данных (CPU, ~15 минут)

Без LFS-артефактов все CPU-этапы пересобираются из трёх parquet задачи, а
выходы GPU-этапов скачиваются с публичных Kaggle kernels без аккаунта, со
сверкой sha256:
[zero-shot dense](https://www.kaggle.com/code/m1r0tvorxc/avito-russian-dense-candidate-retrieval),
[LoRA v2 training](https://www.kaggle.com/code/m1r0tvorxc/user-bge-m3-lora-v2),
[LoRA v2 retrieval](https://www.kaggle.com/code/m1r0tvorxc/lora-v2-dense-retrieval);
для ранних попыток также
[LoRA v1 training](https://www.kaggle.com/code/m1r0tvorxc/avito-user-bge-m3-domain-adaptation),
[LoRA v1 retrieval](https://www.kaggle.com/code/m1r0tvorxc/avito-finetuned-dense-retrieval),
[SPLADE](https://www.kaggle.com/code/m1r0tvorxc/russian-splade-candidate-retrieval).
Обучение и энкодинг на GPU не бит-в-бит детерминированы, поэтому точное
воспроизведение опирается на сохранённые rankings этих kernels.

```bash
PYTHONPATH=src python scripts/build_bm25.py                                  # BM25-индекс
jupyter nbconvert --to notebook --execute --inplace notebooks/02_validation_design.ipynb   # holdout
python scripts/fetch_public_kaggle_outputs.py                                # dense rankings
jupyter nbconvert --to notebook --execute --inplace notebooks/05_dense_rrf_experiments.ipynb  # BM25 rankings
python scripts/train_pu_selector.py                                          # модели попытки 7
python scripts/generate_answer.py --check                                    # answer.csv = попытка 7
```

Notebook 05 попутно перезаписывает `answer.csv` двухканальным baseline;
последняя команда восстанавливает попытку 7 и проверяет sha256. Промежуточные
попытки 2 и 3 пересобираются notebooks 14 и 17 (sha256 `7b4d2579…` и
`c24bf119…`, см. `reports/REPRODUCIBILITY.md`).

## История ранних этапов (попытки 1–2)

На стратифицированном holdout filter-aware BM25 дал `Recall@50 = 0.7504`,
USER-bge-m3 с четырьмя fixed chunks — `0.7681`, настроенный только на dev
BM25+dense RRF — `0.8203`. На независимой половине holdout прирост RRF над
BM25 — `+0.0613`, 95% paired bootstrap CI `[0.0441; 0.0791]`, one-sided
randomization `p < 0.00005`. Oracle recall объединения BM25 и dense кандидатов
на test был `0.9409`: после RRF оставалось `0.1317` измеримого запаса.

SPLADE ablation выбрала `q32/d192`, четыре chunks и local-weight `2.0` только
на dev; эти инженерные решения подтвердились на test после Holm correction, но
добавление SPLADE к BM25+dense — нет: `0.80920 → 0.81199`, paired delta
`+0.00279`, 97.5% CI `[-0.00673; 0.01223]`, `p=0.26584`. SPLADE в решение не
вошёл.

Supervised LTR (CatBoostRanker выбирает 50 items из top-200 union
BM25+dense+SPLADE, leakage-safe click history) поднял test Recall@50 с
`0.80920` до `0.84058`: paired delta `+0.03138`, 98.75% CI
`[0.01054; 0.05279]`, `p=0.00025` при скорректированном пороге `0.0125`.
Leakage-safe LoRA v1 на 17 033 training items подняла dense-only Recall@50 с
`0.76281` до `0.79976`, а в той же LTR-схеме — с `0.84058` до `0.85954`
(delta `+0.01896`, 99.5% CI `[0.00116; 0.03732]`, `p=0.00175` при пороге `0.005`).

Эта конфигурация стала попыткой 1 и получила public `0.698370`. Notebook 14
показал причину: 72.27% benchmark-текстов имеют train frequency `<=1` против
14.89% в holdout, а 62.64% вообще не встречаются в train (после нормализации
регистра, «ё» и пробелов; при точном совпадении строк — 63.0%) — модель с
популярностью объявлений переобучилась на частые запросы. Попытка 2 заменила
её робастным RRF без CatBoost и history, выбранным на редких запросах: на
test-tail `0.86243` против `0.84656` у BM25+fine (delta `+0.01587`, 99.375% CI
`[-0.02646; 0.06349]` — внутренне не доказано), public `0.821129`, то есть
`+0.12276` к попытке 1. Дальнейший путь — лёгкий selector без history
(попытки 3–7), см. «Порядок исследования».

## Лицензии внешних моделей

- `deepvk/USER-bge-m3` — Apache-2.0: оба dense-канала финального решения
  (zero-shot и LoRA v2).
- `intfloat/multilingual-e5-small` — MIT: только контрольный baseline в notebook 05.
- `naver/neuclir22-splade-ru` — CC BY-NC-SA 4.0: исследован в notebook 06, в
  финальное решение не вошёл.
