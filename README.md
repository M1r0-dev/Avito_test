# Avito candidate retrieval

Исследовательское решение задачи кандидатогенерации: для каждого поискового
запроса вернуть до 50 `item_id` из корпуса и максимизировать macro Recall@50.
Все модели работают локально, без внешних inference API.

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

SPLADE-stage использует русский checkpoint `naver/neuclir22-splade-ru` и
контролируемые абляции pruning, chunking и global/local retrieval. Лицензия
checkpoint — CC BY-NC-SA 4.0; это допустимо для данного исследования, но должно
учитываться при возможном коммерческом использовании.

До выбора лучшего эксперимента production-сервис не строится. Код в `src/` —
не сервис, а небольшие тестируемые исследовательские примитивы, общие для
notebooks и batch inference.

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

## Лицензии внешних моделей

- `deepvk/USER-bge-m3` — Apache-2.0, русский sentence encoder для semantic search.
- `intfloat/multilingual-e5-small` — MIT, лёгкий multilingual control baseline.

Финальный список реально использованных моделей фиксируется в dense notebook и
run metadata.
