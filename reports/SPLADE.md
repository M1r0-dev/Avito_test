# SPLADE: итог эксперимента

## Проверяемая гипотеза

Русский learned-sparse retriever может находить лексические расширения, которые
пропускает BM25, и при этом дополнять dense retrieval. Проверялся checkpoint
`naver/neuclir22-splade-ru` (CC BY-NC-SA 4.0), поскольку EDA показал почти
полностью кириллический корпус, а tokenizer checkpoint дал `[UNK] = 0/4828` на
детерминированной выборке запросов.

## Дизайн

- Один и тот же стратифицированный dev/test split, что в BM25+dense этапе.
- 554 920 fixed passages: 140 слов, overlap 30, максимум четыре на item.
- Полный pruning-factorial `q32/q64 × d64/d192`.
- Global и local retrieval; категория и минимальный рейтинг применяются как
  совместимые hard-фильтры, локация — только в отдельном local-channel.
- Variant, local-weight и параметры RRF выбирались только на dev.
- Из-за второго просмотра test primary alpha по Bonferroni снижен до `0.025`,
  использован 97.5% paired bootstrap CI. Пять secondary-тестов скорректированы
  методом Холма.

## Подтверждённые инженерные решения

| Сравнение на test | Paired Δ Recall@50 | 95% CI | Holm p |
|---|---:|---:|---:|
| q32 против q64 при d192 | +0.02416 | [0.01169; 0.03691] | 0.00025 |
| d192 против d64 при q32 | +0.05998 | [0.04070; 0.07932] | 0.00025 |
| четыре chunks против первого | +0.10909 | [0.08333; 0.13506] | 0.00025 |
| global+local против global | +0.48097 | [0.45256; 0.50897] | 0.00025 |

Итоговый SPLADE-only вариант: `q32/d192`, четыре chunks, local-weight `2.0`.
Его полный validation Recall@50 равен `0.63730`; это существенно ниже BM25.

## Primary outcome

| Метод | Dev | Test | Full validation |
|---|---:|---:|---:|
| BM25+dense | 0.83143 | 0.80920 | 0.82032 |
| BM25+dense+SPLADE | 0.84271 | 0.81199 | 0.82735 |

На test paired delta равна `+0.00279`, 97.5% CI
`[-0.00673; 0.01223]`, one-sided randomization `p=0.26584`. Эффект мал и не
подтверждён; SPLADE не принимается в основной pipeline и `answer.csv` не
заменяется. Широкий union трёх retriever-ов имеет Recall `0.94440`, поэтому
запас кандидатов остаётся, но простой RRF его не реализует в top-50.

Машиночитаемые результаты находятся в `reports/splade_metrics.json`, а полный
исполненный анализ — в `notebooks/06_splade_experiments.ipynb`.
