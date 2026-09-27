# Проверка воспроизводимости

Дата проверки: 2026-09-27. Цель — убедиться, что отправленный `answer.csv`
(попытка 2, public Recall@50 `0.821129`) и промежуточные артефакты получаются
из публичного репозитория 1 в 1.

## Методика

1. Анонимный `git clone https://github.com/M1r0-dev/Avito_test.git` (commit
   `a7ce7dc`) во временную директорию, без локальных `artifacts/`.
2. В `dataset/` подкладываются только три исходных parquet из архива задачи.
3. Все шаги запускаются командами из README; каждый артефакт сравнивается с
   исходным по sha256, parquet при расхождении — ещё и по содержимому.
4. Выходы Kaggle kernels скачиваются заново, а исполненный код kernels
   (`kaggle kernels pull`) сравнивается с закоммиченными notebooks.

Окружение: Python 3.13.5, pandas 2.3.3, NumPy 2.1.3, scikit-learn 1.6.1,
SciPy 1.15.3, PyArrow 24.0.0, CatBoost 1.2.10, Linux x86_64, 22 CPU.

## Результаты

| Шаг | Артефакт | Результат |
|---|---|---|
| `pytest` | 6 тестов | passed |
| `scripts/build_bm25.py` | `bm25.joblib`, `bm25_matrix.npz`, `items.parquet` | побайтно совпадают |
| notebook 02 | `validation/manifest.parquet`, `validation/labels.parquet` | побайтно совпадают |
| Kaggle 04 output | `dense_rankings.parquet` `b725e38e…` | побайтно совпадает |
| Kaggle 06 output | `splade_rankings.parquet` `535576a2…` | побайтно совпадает |
| Kaggle 08b output | `finetuned_dense_rankings.parquet` `d08f8fed…` | побайтно совпадает |
| notebook 05 | `rankings/bm25_rankings.parquet` | побайтно совпадает |
| notebook 05 | `dense_rrf_metrics.json` | совпадает, кроме 17-го знака одного bootstrap CI (`…396` → `…397`, float round-off) |
| notebook 08 | `finetuned_dense_metrics.json` | совпадает; подтверждает local weight `1.5` и BM25+fine `rrf_k=20`, weight `2.0` |
| notebook 03 (после исправления 1 ниже) | `bm25_ablations.json`, `bm25_stat_tests.json` | совпадают |
| notebook 12 | `features/ltr_validation_features.parquet`, `answer_objective_ltr.csv`, `ltr_objective_metrics.json` | побайтно совпадают |
| **notebook 13** | **`answer_finetuned_dense_ltr.csv`** `d017aa45…` | **побайтно совпадает с отправленной попыткой 1** |
| notebook 13 | `finetuned_dense_ltr_metrics.json` | побайтно совпадает |
| **notebook 14** | **`answer.csv`** `7b4d2579…` | **побайтно совпадает с отправленной попыткой 2** |
| notebook 14 | `distribution_shift_rrf_metrics.json` | совпадает |

Время на 22 CPU: build_bm25 ~2 мин, notebook 05 ~3 мин, 12 ~12 мин,
13 ~16 мин, 14 ~20 с. Notebooks 01, 06, 07, 09–11 в этой проверке не
перезапускались: они не лежат на пути ни к одной из отправленных попыток.

`labels.parquet` из notebook 02 совпадает с `validation_labels.parquet` в
выходах kernels 06 и 08b, то есть GPU-этапы работали на том же split.

## Код Kaggle kernels

Исполненные версии kernels 04 и 08b совпадают с `kaggle/*.ipynb` в репозитории.
Для 08a единственное отличие — закреплённая в репозитории после запуска ревизия
модели (`MODEL_REVISION = "0cc6cfe…"`). Исполненная версия вызывала
`snapshot_download(MODEL)` без ревизии, но head `deepvk/USER-bge-m3` равен
`0cc6cfe48e260fb0474c753087a69369e88709ae` с 2024-07-18, поэтому использована та
же модель.

## Что воспроизводится бит-в-бит, а что нет

- **CPU-часть** (BM25, split, RRF, CatBoost с `random_seed=42`) детерминирована.
- **GPU-часть** (LoRA-обучение на 2×T4 через DDP, dense encoding) не фиксирует
  `torch.use_deterministic_algorithms`, поэтому повторный запуск даст близкие,
  но не идентичные rankings. Для точного воспроизведения используются
  сохранённые выходы публичных kernels:
  `python scripts/fetch_public_kaggle_outputs.py [--splade]` скачивает их без
  Kaggle-аккаунта и проверяет sha256.
- Zero-shot kernel 04 загружает `deepvk/USER-bge-m3` без явной ревизии. При
  будущем обновлении модели на Hugging Face повторный GPU-прогон может
  отличаться.

## Найденные и исправленные дефекты

1. `notebooks/03_bm25_experiments.ipynb` безусловно читал
   `artifacts/bm25/item_language.parquet`, который не создавал ни один файл в
   истории репозитория, и падал на чистом clone. Определение восстановлено:
   `dominant_script(title + " " + description[:1000])`. Оно даёт 100% совпадение
   меток и побайтно тот же файл; генерация добавлена в `scripts/build_bm25.py`.
2. `notebooks/14_distribution_shift_robust_rrf.ipynb` архивировал попытку 1
   копированием текущего `answer.csv`. На чистом checkout после notebook 05 в
   `submissions/attempt_1_finetuned_ltr.csv` попадал двухканальный baseline
   (`44d722a5…`) вместо LTR-ответа (`d017aa45…`). Теперь источник — явный выход
   notebook 13 `answer_finetuned_dense_ltr.csv`. На `answer.csv` дефект не
   влиял.

## Публичная доступность

- GitHub-репозиторий публичный; история не содержит raw parquet, весов моделей и
  токенов.
- Все четыре Kaggle kernels публичны, их выходы доступны анонимно.
- Kaggle dataset `m1r0tvorxc/avito-candidate-data` намеренно приватный: это
  исходные данные организаторов. Проверяющий использует свой экземпляр из
  архива задачи.
