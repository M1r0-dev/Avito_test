# Решение Avito candidate retrieval

Финальный файл — `answer.csv`. Репозиторий содержит полный исследовательский
pipeline: EDA, фиксированный validation split, retrieval-эксперименты,
статистические тесты, GPU-notebooks и shift-aware RRF-сборку.

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
2. Русский `deepvk/USER-bge-m3` дообучается leakage-safe LoRA на 17 033
   уникальных положительных item. Validation query signatures исключены из
   обучения до sampling.
3. Fine-tuned bi-encoder строит global и location-local dense rankings по
   554 920 passages на двух Tesla T4.
4. Zero-shot и fine-tuned dense rankings объединяются с BM25 через RRF с
   `k=20` и весами `1 / 0.75 / 1.25`. Веса выбраны только на редком dev-tail.
5. SPLADE и CatBoostRanker подробно исследованы, но не входят в текущий
   submission. LTR дал лучший offline score, однако плохо перенёсся на
   benchmark; SPLADE не прошёл прямой statistical gate.

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

3. Выполнить локальные notebooks `01`–`03` и построить BM25 artifacts.
4. Запустить `kaggle/04_dense_gpu_experiment.ipynb`, скачать output через
   `scripts/fetch_kaggle_output.sh`, затем выполнить notebook `05`.
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
