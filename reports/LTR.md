# Supervised LTR: итог эксперимента

> Исторический отчёт notebook 07 (ранний этап). LTR с click history стал
> публичной попыткой 1 (public `0.698370`) и из-за переобучения на частые
> запросы в финальное решение не вошёл; финал — попытка 7, см. `SOLUTION.md`.
> Упоминания `answer.csv` ниже относятся к состоянию репозитория на тот момент.

## Гипотеза

Test-oracle широкого пула существенно выше RRF@50, поэтому CatBoostRanker может
лучше сжать union retrieval-каналов до 50 items. Использовались top-200 каждого
канала; pool Recall равен `0.93760` на test.

## Защита от утечки

- Exact validation query signatures удалены из click-history по всем пяти
  query-полям.
- Dev разделён на fit/tune со стратификацией.
- Feature set и iteration budget выбраны только на tune.
- Для четвёртого model-family look принят Bonferroni threshold `0.0125` и
  98.75% paired bootstrap CI.

## Tune ablations

| Features | Tune Recall@50 | Iterations |
|---|---:|---:|
| ranks | 0.74376 | 58 |
| ranks + content/metadata | 0.79533 | 92 |
| ranks + content/metadata/history | 0.82410 | 195 |

Content против rank-only: paired delta `+0.05157`, 95% CI
`[0.02932; 0.07655]`, `p=0.00005`. History против content: delta `+0.02877`,
95% CI `[0.00109; 0.05809]`, `p=0.03060`; эффект положительный, но менее
устойчивый. Победитель выбран по заранее определённому tune Recall@50.

## Primary test

| Метод | Test Recall@50 |
|---|---:|
| BM25+dense RRF | 0.80920 |
| Supervised LTR | 0.84058 |

Paired delta `+0.03138`, 98.75% CI `[0.01054; 0.05279]`, one-sided paired
randomization `p=0.00025`. Критерий принятия выполнен, поэтому LTR submission
записан в `answer.csv`.

До цели `0.9` остаётся `0.05942`. Наиболее заметные регрессии наблюдаются в
очень малом location bucket `1–99` и на query-frequency `1`; их необходимо
контролировать при следующих моделях. Полные числа находятся в
`reports/ltr_metrics.json`, исполненный эксперимент — в
`notebooks/07_supervised_ltr_experiments.ipynb`.
