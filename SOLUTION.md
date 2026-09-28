# Avito: кандидатогенерация под Recall@50 — финальное решение

Документ для проверяющего: что сдано, как это проверить за несколько минут и
почему выбран каждый шаг. Все числа ссылаются на notebook/отчёт, где их можно
перепроверить; все сравнения — парные статистические тесты на одних и тех же
запросах.

## 1. Коротко

- **Сдаётся** `answer.csv` = публичная попытка 6, sha256 `27705d5…`,
  public Recall@50 **0.836972**.
- **Рамка**: только первая стадия каскада — 50 кандидатов с максимальным
  Recall@50 для следующего ранжирующего этапа; re-ranking вне задачи.
- **Offline** (test-половина holdout, 1 226 запросов): Recall@50 **0.86120**;
  на редких запросах (test-tail, 189 запросов, ≈ профиль benchmark) **0.91005**.
- **Latency guardrail** (ноутбук, warm p95 ≤ 500 ms, batch=1): пройден —
  p95 **398 ms** (раздел 6).

```
query ─┬─ BM25 (filtered + plain)                  ─┐
       ├─ USER-bge-m3 zero-shot → exact global/local ─┼─ union top-100 → RRF top-200
       └─ USER-bge-m3 LoRA v2   → exact global/local ─┘        │
                                                               ▼
           33 признака (ранги каналов, token overlap, локация, приоры объявления)
                                                               ▼
           3 PU-bagged CatBoost (YetiRankPairwise) → средний 1/(20+rank) → top-50
```

## 2. Как проверить

Нужны три parquet задачи в `dataset/` и `git lfs pull` (ranking-артефакты и
модели лежат в LFS). GPU и сеть не нужны.

| Что проверяется | Команда | Ожидаемо |
|---|---|---|
| финальный `answer.csv` воспроизводится из артефактов | `python scripts/generate_answer.py --check` | sha256 `27705d5bc6a6…`, ~2 мин CPU |
| модели selector обучаются заново бит-в-бит | `python scripts/train_pu_selector.py --verify-export artifacts/recall50_kaggle_input/c2_validation.parquet`¹ | признаки = notebook 19, те же `.cbm` |
| unit-тесты | `pytest` | зелёные |
| latency online-пути | `python scripts/export_onnx_encoder.py` и `python scripts/benchmark_final_latency.py --encoder onnx` ² | раздел 6 |

¹ Экспорт создаёт `notebooks/25a_recall50_export_pool.ipynb`; без флага
скрипт просто обучает модели из LFS-артефактов.
² Нужны FP16 passage vectors LoRA v2 (`kaggle kernels output
m1r0tvorxc/avito-lora-v2-dense-retrieval -p artifacts/finetuned_v2_kaggle`) и
merged LoRA v2 (`kaggle kernels output m1r0tvorxc/avito-user-bge-m3-lora-v2`),
экспортированный командой `python scripts/export_onnx_encoder.py --model
<путь к merged-модели> --revision "" --output artifacts/onnx/lora_v2`.

Ключевые файлы: `src/avito_retrieval/pu_selector.py` (признаки и ансамбль),
`src/avito_retrieval/dense_search.py` (точный CPU-поиск),
`src/avito_retrieval/onnx_encoder.py`, `scripts/generate_answer.py`,
`reports/public_submissions.json` (журнал попыток с sha256).

## 3. Валидация: почему числам можно доверять и где их предел

- **Holdout.** Из train выделены 2 452 query signatures со стратификацией
  (notebook 02), разделены пополам на dev и test. Все validation-сигнатуры
  исключены из обучения dense-моделей и click-history до обучения.
- **Выбор только на dev.** Гиперпараметры и конфигурации выбираются по
  2-fold out-of-fold оценке на dev; test смотрится один раз на гипотезу.
- **Тесты.** Paired bootstrap CI + односторонний sign-randomization test по
  per-query Recall@50 (`avito_retrieval.statistics.paired_recall_test`),
  Bonferroni по endpoints и повторным просмотрам test. Отрицательные
  результаты тоже закоммичены отдельными notebooks.
- **Shift.** 62.6% текстов benchmark не встречаются в train, 72% имеют
  частоту ≤ 1; в holdout таких 15%. Поэтому primary endpoint — test-tail.
- **Предел.** Holdout оптимистичен (public ниже offline на 0.02–0.07), а при
  ~1.1 relevant на запрос разрешение test около ±0.01: эффекты меньше этого
  статистически не отличимы. Это главный ограничитель дальнейшего тюнинга.

| # | Метод | Offline test-tail / test | Public |
|---|---|---|---|
| 1 | LTR + click history | — / 0.85954 | 0.698370 |
| 2 | RRF BM25 + zero-shot + LoRA v1 | 0.86243 / 0.83238 | 0.821129 |
| 3 | learned fusion (CatBoost по рангам) | 0.90476 / 0.84448 | 0.824331 |
| 4 | + LoRA v2, selector (4 канала) | 0.91005 / 0.86120 | 0.837229 |
| 5 | абляция BM25 + LoRA v2 | 0.88360 / 0.85651 | 0.829627 |
| **6** | **BM25 + zero-shot + LoRA v2, PU selector** | **0.91005 / 0.86120** | **0.836972** |

## 4. Шаги pipeline и их обоснование

1. **Текст запроса и hard filters** (notebooks 03, 05). Запрос = query +
   текст фильтров; категория (кроме `0`) и минимальный рейтинг — жёсткие
   фильтры. Локация **не** жёсткая: 16.9% positive train-пар в другой
   локации, поэтому каждый канал ищет global и location-local и сливает их RRF.
2. **BM25** по title, параметрам и описанию — сильный lexical baseline для
   брендов/моделей; `bm25_plain` (без фильтров) — дополнительный признак.
3. **Zero-shot `deepvk/USER-bge-m3`** (notebook 05). Объявление режется на ≤ 4
   passages по 140 слов (длинные описания размывают один вектор), passages
   сворачиваются в items. Лицензия Apache-2.0.
4. **LoRA v2 того же encoder** (Kaggle 10A/10B, notebook 19): 457 439
   leakage-safe пар, маска ложных негативов. Канал против LoRA v1: test
   Recall@50 0.8268 против 0.7998, `+0.027`, p = 0.0004.
5. **Пул** — union top-100 каналов, обрезанный до RRF top-200 (notebooks 22,
   24): pool Recall@200 0.921 на test и 0.974 на test-tail. Узкое место —
   сжатие 200 → 50, а не поиск.
6. **Selector** (notebooks 17, 19, 24) — лёгкий CatBoost по рангам каналов,
   token overlap, совпадению локации и приорам объявления; click history
   исключена после попытки 1 (переобучение на head-запросы). Selector против
   RRF тех же каналов: test 0.86120 против 0.84652.
7. **PU-bagging и отказ от LoRA v1** (notebooks 23, 24). Непрокликанные
   объявления — unlabeled, а не негативы, поэтому каждый из трёх bags берёт
   все positives и 48 unlabeled (2/3 hard). Сам по себе PU **не доказан**
   (notebook 23: +0.0106 на tail, p = 0.31). Без LoRA v1 обычный selector
   (C2, notebook 19) отставал от попытки 4 на test-tail на один запрос
   (0.90476 против 0.91005); с PU-bagging и пулом top-200 он совпал с ней
   (tail 0 побед / 0 поражений, test 0.86120), public −0.000257. Разница в
   пределах шума, поэтому решающим аргументом за попытку 6 была latency: на
   один 568M-encoder меньше при том же offline Recall@50.

## 5. Что проверено и отклонено

| Гипотеза | Результат | Где |
|---|---|---|
| LTR + click history | offline +, public 0.698: переобучение на head/popularity | 13, 14 |
| SPLADE (`neuclir22-splade-ru`) | прямой прирост не прошёл gate | 06, 07 |
| Multi-view dense (query-only, title+params) | tail не улучшен, отклонено | 21 |
| Cross-encoder re-ranker | вне рамки (это ранжирование) и ~мин/запрос на CPU | — |
| Встроенный Recall@k objective CatBoost | `LambdaMart`/`StochasticRank`/`YetiRank` отвергают `RecallAt`; `StochasticFilter:metric=RecallAt;top=50` молча игнорирует параметры (FilteredDCG), OOF dev 0.847 | 25C |
| LambdaMART c весами \|ΔRecall@50\| (свой objective) | test +0.0053 и на tail, и на test, p = 0.50 / 0.15 — не принято | 25A–C |
| Он же с запасом на train (cutoff 10) | OOF dev +0.007, но test-tail −0.0053, test +0.0014 (α = .0125) — не перенеслось | 26A–D |

Про собственный objective: диагностика показала, что при in-sample весах все
обучающие positives пула попадают в top-50 уже через 100 деревьев, после чего
лосс перестаёт учить обобщению; запас на train это исправил на dev, но не на
test. Вывод: на этом holdout разница между objectives меньше его разрешения.

## 6. Latency

Guardrail ноутбука (Intel Core Ultra 7 155H, CPU, batch=1, 500 запросов):
**p50 282 ms, p95 398 ms, p99 471 ms** — пройден (`reports/LATENCY.md`).

| Конфигурация online-пути | p50 / p95 | Почему Recall@50 не меняется |
|---|---|---|
| исходная: PyTorch encoders, последовательно | 584 / 872 ms | — |
| PyTorch, три retrieval-ветки параллельно | 461 / 649 ms | top-50 идентичен на всех 500 запросах |
| **финальная**: ONNX Runtime fp32 encoders + параллельные ветки + бюджет потоков | **282 / 398 ms** | ORT: косинус к PyTorch ≥ 0.9999992, online v2 non-inferior (test +0.0019, CI [0, 0.0046], tail 0/0, notebook 28); CPU-поиск вместо GPU fp16 non-inferior (notebook 27); потоки меняют только расписание |

Главный выигрыш — ONNX Runtime (encode p50 ~210 → ~80–115 ms). Бюджет
потоков (ORT/MKL/CatBoost по 4, `KMP_BLOCKTIME=0`) — страховка от
переподписки ядер между ветками; его отдельный эффект не выделен: без него
parallel в одном из окон дал p95 278 ms, но прогоны шли в разных условиях.

Хвост на этом ноутбуке зависит от памяти: при заполненном swap первые
~1 500 запросов после загрузки индексов дают p95 > 1 s, после этого оба
режима проходят; в отчёте приведены все 12 прогонов, включая неудачные.
Попутно найдена ошибка интеграции (префикс `"query: "` у dense-запроса,
−0.006 Recall@50), исправлена `dense_query_text`.

## 7. Ограничения

- Holdout не моделирует benchmark полностью (category `0`, новые тексты),
  поэтому offline-приросты систематически больше публичных.
- Zero-shot passage vectors не сохранялись: CPU-поиск проверен на LoRA v2
  (тот же код), zero-shot в latency-замере — случайная матрица той же формы.
- Latency измерена на ноутбуке с фоновыми процессами (IDE, браузер); хвост
  p99 чувствителен к ним.
