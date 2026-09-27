# Avito candidate retrieval

Исследовательское решение задачи кандидатогенерации: для каждого поискового
запроса вернуть до 50 `item_id` из корпуса и максимизировать macro Recall@50.
Все модели работают локально, без внешних inference API.

Краткое описание финального подхода, ошибок и полного воспроизведения находится
в [`SOLUTION.md`](SOLUTION.md). Фактический Recall@50 отправленного
`answer.csv` на платформе: **0.698370**; offline holdout Recall@50: `0.85954`.

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
10. `notebooks/13_finetuned_dense_ltr_experiments.ipynb` — принятая замена
    zero-shot dense на адаптированный канал; текущий test Recall@50 `0.85954`.

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

Raw Parquet, индексы, embeddings, Kaggle staging и промежуточные submissions
игнорируются Git.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

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
Результат принят и записан в `answer.csv`; до цели `0.9` осталось `0.04046`.

## Лицензии внешних моделей

- `deepvk/USER-bge-m3` — Apache-2.0, русский sentence encoder для semantic search.
- `intfloat/multilingual-e5-small` — MIT, лёгкий multilingual control baseline.
- `naver/neuclir22-splade-ru` — CC BY-NC-SA 4.0, русский learned-sparse
  retriever; исследован, но не принят в финальный ensemble.

Финальный список реально использованных моделей фиксируется в dense notebook и
run metadata.
