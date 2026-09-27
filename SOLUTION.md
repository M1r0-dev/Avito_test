# Решение Avito candidate retrieval

Финальный файл — `answer.csv`. Репозиторий содержит полный исследовательский
pipeline: EDA, фиксированный validation split, retrieval-эксперименты,
статистические тесты, GPU-notebooks и learned candidate selector.

Решается только первая стадия каскада — кандидатогенерация: 50 кандидатов с
максимальным Recall@50 передаются дальше, на ранжирование. Ранжирование вне
рамки задачи. Ограничения — Recall@50 и latency guardrail `p95 ≤ 500 ms` на
ноутбуке / `≤ 100 ms` на GPU при batch=1.

## Данные и признаки

Используются все доступные признаки запроса и объявления:

- `search_query` и `search_infm_params_text` формируют текст запроса;
- title, параметры и description объявления формируют документы;
- `search_category` применяется как hard filter, кроме значения `0`;
- location используется двумя каналами — global и local, потому что 16.9%
  положительных train-пар находятся в разных локациях;
- требование минимального рейтинга извлекается из текста фильтра и применяется
  как hard constraint;
- цена, рейтинг, отзывы, координаты, доступность телефона/сообщений, совпадения
  токенов и leakage-safe click counts используются LTR-моделью.

EDA показал, что description и параметры часто длинные. Поэтому объявление
представляется максимум четырьмя фиксированными passages по 140 слов с overlap
30, после чего passage ranks сворачиваются в уникальные `item_id`. Отдельный
language hard filter отклонён: корпус уже на 99.87% кириллический, а удаление
латиницы теряет бренды и смешанные названия.

## Модели и текущий submission

1. BM25 ищет lexical candidates по title, parameters и description.
2. Русский `deepvk/USER-bge-m3` представлен zero-shot каналом и двумя LoRA:
   v1 на 17 033 парах и v2 на 457 439 leakage-safe уникальных query-item парах.
   Все validation query signatures исключены до обучения.
3. Каждый bi-encoder строит global и location-local rankings по 554 920
   passages на двух Tesla T4; v2 сохраняет также FP16 passage/query vectors.
4. Лёгкий CatBoost candidate selector выбирает 50 items из union top-100 BM25,
   zero-shot, LoRA v1 и LoRA v2. Используются только ranks, token overlap,
   location match и item priors — click history исключена.
5. RRF-only с теми же каналами сохранён как интерпретируемый контроль:
   test-tail/test `0.89947/0.84652` против `0.91005/0.86120` у selector.
   SPLADE не входит в submission, потому что его прямой прирост не прошёл gate.

Open-source зависимости: pandas, NumPy, scikit-learn, SentenceTransformers,
PEFT, PyTorch, CatBoost и PyArrow. Внешние inference API не используются.
`USER-bge-m3` имеет Apache-2.0; исследованный SPLADE checkpoint
`naver/neuclir22-splade-ru` — CC BY-NC-SA 4.0.

## Проверка качества

Из train создан фиксированный стратифицированный holdout на 2 452 query
signatures. Он делится на dev/test; параметры выбираются только на dev. Все
сравнения выполняются по одинаковым запросам с paired bootstrap confidence
interval и односторонним sign-randomization test. Порог значимости уменьшается
с учётом последовательных model-family экспериментов.

Offline-оценка первой LTR-попытки:

- прежний zero-shot dense LTR: Recall@50 `0.84058`;
- fine-tuned dense LTR: Recall@50 `0.85954`;
- paired delta `+0.01896`;
- 99.5% CI `[0.00116; 0.03732]`;
- randomization `p=0.00175` при threshold `0.005`.

Фактический Recall@50 первой загруженной попытки — **`0.698370`**. Разрыв
с offline-оценкой явно указан: train-derived holdout переоценивает перенос на
benchmark. Вероятные причины — shift запросов/объявлений, переоценка head-query
и popularity/history сигналов, а также многократная последовательная адаптация
к одному holdout. Это ограничение текущей validation-схемы, а не форматная
ошибка submission. Аудит после этой попытки показал: 62.64% benchmark query
texts не встречаются в train, 72.27% имеют train frequency `<=1`, тогда как в
holdout tail занимает только 14.89%. Кроме того, category `0` составляет 9.05%
benchmark против 0.08% holdout.

Текущий shift-aware RRF выбран на dev-tail. На независимом test-tail его
Recall@50 равен `0.86243` против `0.84656` у BM25+fine. Paired delta `+0.01587`,
99.375% CI `[-0.02646; 0.06349]`, `p=0.253`: внутреннее улучшение не доказано,
поэтому это явно обозначено как контролируемая публичная проверка переноса.

Фактический Recall@50 второй попытки — **`0.821129`**:

| Попытка | Offline Recall@50 | Public Recall@50 | Разрыв |
|---|---|---|---|
| 1. Fine-tuned dense LTR | test `0.85954` | `0.698370` | `−0.161` |
| 2. Shift-aware RRF | test-tail `0.86243`, test `0.83238` | `0.821129` | `−0.041` / `−0.011` |

Публичный прирост `+0.12276` подтверждает направление shift-гипотезы: supervised
LTR с popularity/history переобучался на head-запросы holdout. Прирост
относится ко всей замене LTR/history на RRF. Отдельный вклад весов
`1 / 0.75 / 1.25` не измерялся, потому что на benchmark нет разметки и каждая
попытка ограничена. Оставшийся разрыв означает, что holdout по-прежнему не
моделирует benchmark: category `0` (9.05% benchmark) и тексты, не встречавшиеся
в train (62.64%), в нём почти отсутствуют. Журнал попыток с sha256 —
`reports/public_submissions.json`.

## Найденные ошибки и принятые решения

- Location hard filter терял межрегиональные positives — заменён global/local
  retrieval каналами.
- Russian-only filter не дал прироста и терял mixed-script объявления —
  отклонён.
- Длинные документы ухудшали единичное dense-представление — введено
  фиксированное chunking с item-level collapse.
- Первый DDP fine-tuning завис на параллельной загрузке checkpoint и достиг
  12-часового лимита Kaggle — модель скачивается один раз, ranks загружают её
  локально и последовательно, обучение и retrieval разделены.
- Kaggle `torchao 0.10` конфликтовал с Transformers — неиспользуемый optional
  пакет удаляется перед PEFT training.
- History-expanded pool, history-only LTR features и альтернативные ranking
  objectives ухудшили test Recall и были отклонены.
- LTR улучшал исходный holdout, но публичный результат выявил covariate shift;
  текущий submission исключает CatBoost и history, уменьшая зависимость от
  head-query train distribution.
- Итоговый CSV отдельно проверен на точное покрытие query, 50 уникальных corpus
  IDs, lowercase hex-формат и отсутствие индексной колонки.
- Аудит воспроизводимости на чистом clone нашёл два дефекта. Notebook 03 читал
  `item_language.parquet`, который не создавался кодом репозитория: генерация
  восстановлена в `scripts/build_bm25.py` и даёт побайтно тот же файл. Notebook
  14 архивировал первую попытку копированием текущего `answer.csv`, поэтому на
  чистом checkout сохранял туда ответ notebook 05: теперь источник — явный
  выход notebook 13 `answer_finetuned_dense_ltr.csv`.
- Был начат zero-shot cross-encoder этап (`BAAI/bge-reranker-v2-m3`, 568M) над
  пулом ~200 кандидатов. Это ошибка рамки: cross-encoder — ранжирование, то
  есть следующая стадия каскада, а по латентности он не проходит guardrail на
  два порядка (на CPU ~3.4 пары/с, то есть порядка минуты на запрос). Kernel
  отменён до получения результатов, код удалён; решения по нему не принимались.
- End-to-end замер показал, что сама попытка 2 не проходит CPU guardrail:
  p95 `811.6 ms` против `500 ms`, из них ~590 ms — два прохода 568M
  query encoder (`reports/LATENCY.md`).
- Закоммиченный `kaggle/finetune_train_gpu.py` фиксирует ревизию
  `deepvk/USER-bge-m3` `0cc6cfe…`, но исполненная Kaggle-версия 08a скачивала
  модель без явной ревизии. Это та же ревизия: head репозитория модели не
  менялся с 2024-07-18.

## Воспроизведение

1. Положить три исходных parquet в `dataset/`, как описано в
   `dataset/README.md`.
2. Установить окружение:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e '.[dev]'
   pytest
   ```

3. Построить BM25 artifacts (`PYTHONPATH=src python scripts/build_bm25.py`),
   затем выполнить локальные notebooks `01`–`03`.
4. Запустить `kaggle/04_dense_gpu_experiment.ipynb`, скачать output через
   `scripts/fetch_kaggle_output.sh`, затем выполнить notebook `05`.
   Все четыре Kaggle kernels публичны. Без собственного GPU-прогона их
   сохранённые выходы скачиваются без Kaggle-аккаунта командой
   `python scripts/fetch_public_kaggle_outputs.py --splade` с проверкой sha256.
   Этот путь нужен для бит-в-бит воспроизведения: повторное LoRA-обучение на
   2×T4 не детерминировано.
5. Запустить SPLADE GPU notebook `kaggle/splade/06_splade_gpu_experiment.ipynb`,
   скачать output и выполнить локальные notebooks `06` и `07`.
6. Запустить `kaggle/finetune/08a_finetune_train_gpu.ipynb`, затем зависимый
   `kaggle/finetune_retrieval/08b_finetuned_dense_retrieval.ipynb`. Скачать
   rankings командой `scripts/fetch_finetuned_dense_output.sh`.
7. Выполнить notebooks `08`, `12` и `13`. Notebook `12` материализует
   зафиксированную zero-shot control pair table; notebook `13` обучает
   fine-tuned dense LTR первой попытки.
8. Выполнить `notebooks/14_distribution_shift_robust_rrf.ipynb`: он измеряет
   shift, выбирает RRF на dev-tail, проводит paired-тест на test-tail, сохраняет
   предыдущую попытку и воспроизводит текущий `answer.csv`.

Ключевой финальный notebook:
`notebooks/14_distribution_shift_robust_rrf.ipynb`. Все random seeds,
chunking parameters, split и CatBoost iterations сохранены в коде и reports.
Минимальный путь до текущего `answer.csv` (build_bm25 → 02 → fetch → 05 → 14)
и результаты проверки на чистом clone с хешами описаны в
`reports/REPRODUCIBILITY.md`.
