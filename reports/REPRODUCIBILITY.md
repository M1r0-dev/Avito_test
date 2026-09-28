# Проверка воспроизводимости

## Финальное решение (попытка 7): откуда каждый файл

`answer.csv` попытки 7 (sha256 `994ef6f3…`, public `0.840831`) собирается
одной командой **без GPU, Kaggle и сети** — все входы лежат в Git LFS:

```bash
git lfs pull                                   # артефакты ниже
python scripts/generate_answer.py --check      # ~2 мин CPU → sha256 994ef6f3…
python scripts/train_pu_selector.py --models-dir /tmp/m   # опционально: модели заново
python scripts/generate_answer.py --models-dir /tmp/m --check  # тот же sha
```

### Входы `generate_answer.py` и их происхождение

| Файл (в LFS) | Что внутри | Кто создал | Как получить заново | Бит-в-бит |
|---|---|---|---|---|
| `dataset/benchmark_*.parquet` | запросы и объявления задачи | организаторы | из архива задачи (в репо не кладём) | — |
| `artifacts/bm25/*` | BM25-индекс: матрица документ×термин, словарь, items | `scripts/build_bm25.py` | `PYTHONPATH=src python scripts/build_bm25.py` | да |
| `artifacts/validation/{manifest,labels}.parquet` | holdout 2 452 запросов и их relevant | notebook 02 из train | notebook 02 | да |
| `artifacts/rankings/bm25_rankings.parquet` | для 4 904 запросов (holdout + benchmark): `bm25` — top-250 с фильтрами, `bm25_plain` — top-250 без фильтров | notebook 05 | notebook 05 | да |
| `artifacts/dense_kaggle/dense_rankings.parquet` | zero-shot USER-bge-m3: `dense_global`, `dense_local` — top-250 items | Kaggle `avito-russian-dense-candidate-retrieval` (`kaggle/dense_gpu.py`, 2×T4) | `python scripts/fetch_public_kaggle_outputs.py` (без аккаунта, sha256) | скачивание — да; перезапуск GPU — нет |
| `artifacts/finetuned_v2_kaggle/finetuned_v2_dense_rankings.parquet` | LoRA v2: `finetuned_v2_global`, `finetuned_v2_local` — top-250 items | Kaggle `lora-v2-dense-retrieval` (`kaggle/finetune_v2_retrieval_gpu.py`) на модели из `user-bge-m3-lora-v2` (`kaggle/finetune_v2_train_gpu.py`) | то же | то же |
| `models/recall50_seed{41,42,43}_round{1,2}.cbm` | selector попытки 7: 3 PU bags × 2 раунда по 50 деревьев | `scripts/train_pu_selector.py` (порт stage 25B) на dev-половине holdout | `python scripts/train_pu_selector.py` | да, проверено |

### Почему от dense-каналов у нас только ranking-листы

Dense-каналы считались на Kaggle (2×T4): энкодинг 554 920 passages на CPU
занял бы часы. Каждое ядро выдаёт **ranking-лист**: для каждого из 4 904
запросов — top-250 `item_id` глобально и top-250 в локации запроса. Именно
эти листы, а не векторы, являются входом selector и лежат в LFS. Как они
получены:

1. объявление → ≤ 4 passages по 140 слов (overlap 30), title в каждом;
2. passages и запрос (`query + фильтры`, **без** префикса `"query: "`)
   кодируются `deepvk/USER-bge-m3` (CLS, L2-norm, max 256 токенов);
3. точный поиск fp16 на двух GPU-шардах: top-1 500 passages с каждого, плюс
   top-750 среди passages локации запроса;
4. passages сворачиваются в items с hard filters (категория, мин. рейтинг),
   первые 250 уникальных items — ranking.

Для LoRA v2 предварительно обучается LoRA (r=32) на 457 439 уникальных парах
запрос–объявление из train, все validation-запросы исключены (ядро
`user-bge-m3-lora-v2`), и тот же поиск выполняется merged-моделью.

GPU-этапы не бит-в-бит воспроизводимы при перезапуске (DDP, недетерминированные
ядра CUDA), поэтому воспроизводимый артефакт — их **сохранённый выход**. Его
можно взять из LFS или скачать заново с публичного ядра той же версии;
`fetch_public_kaggle_outputs.py` сверяет sha256 с файлами, на которых
построены сабмиты.

**Векторы.** Ядро zero-shot векторы passages не сохраняло — только rankings.
Ядро LoRA v2 сохранило FP16-векторы passages и запросов; они не нужны для
`answer.csv` и не лежат в LFS (1.1 ГБ), но нужны для online-пути и замера
latency: `python scripts/fetch_public_kaggle_outputs.py --online` скачивает их
и merged LoRA v2 без аккаунта.

## Историческая проверка: попытка 2 с чистого clone (2026-09-27)

Цель — убедиться, что отправленный тогда `answer.csv` (попытка 2, public
Recall@50 `0.821129`) и промежуточные артефакты получаются из публичного
репозитория 1 в 1.

### Методика

1. Анонимный `git clone https://github.com/M1r0-dev/Avito_test.git` (commit
   `a7ce7dc`) во временную директорию, без локальных `artifacts/`.
2. В `dataset/` подкладываются только три исходных parquet из архива задачи.
3. Все шаги запускаются командами из README; каждый артефакт сравнивается с
   исходным по sha256, parquet при расхождении — ещё и по содержимому.
4. Выходы Kaggle kernels скачиваются заново, а исполненный код kernels
   (`kaggle kernels pull`) сравнивается с закоммиченными notebooks.

Окружение: Python 3.13.5, pandas 2.3.3, NumPy 2.1.3, scikit-learn 1.6.1,
SciPy 1.15.3, PyArrow 24.0.0, CatBoost 1.2.10, Linux x86_64, 22 CPU.

### Результаты

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

### Быстрый путь через Git LFS (попытка 3)

Отдельная проверка пути проверяющего: анонимный `git clone` с установленным
Git LFS (commit `6e8a25d`, ~4 минуты на загрузку 395 MB), в `dataset/` —
только `benchmark_queries.parquet` и `benchmark_items.parquet`, затем
`python scripts/generate_answer.py --check`. За 86 s на CPU скрипт записал
`answer.csv` с sha256 `c24bf119…`, побайтно равный отправленной попытке 3
(public `0.824331`) и архиву `submissions/attempt_3_learned_fusion.csv`;
рабочее дерево осталось чистым. Путь не требует GPU, Kaggle, сети и
переобучения: признаки строятся из LFS-rankings, а скоры — сохранённой
моделью `models/learned_fusion_attempt3.cbm`.

### Код Kaggle kernels

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
- Все Kaggle kernels, чьи выходы входят в сабмиты, публичны и доступны
  анонимно: zero-shot (`avito-russian-dense-candidate-retrieval`), LoRA v1
  (`avito-user-bge-m3-domain-adaptation`, `avito-finetuned-dense-retrieval`),
  LoRA v2 (`user-bge-m3-lora-v2`, `lora-v2-dense-retrieval`), SPLADE
  (`russian-splade-candidate-retrieval`).
- Kaggle dataset `m1r0tvorxc/avito-candidate-data` намеренно приватный: это
  исходные данные организаторов. Проверяющий использует свой экземпляр из
  архива задачи.
