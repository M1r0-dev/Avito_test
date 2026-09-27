# Latency guardrail

## Контракт

- Исследовательский ноутбук: warm `p95 ≤ 500 ms`, batch size 1.
- Production GPU stretch: warm `p95 ≤ 100 ms`, batch size 1.
- Минимум 500 измеряемых запросов после 25 warm-up запросов.
- Включаются preprocessing, query encoding, все принятые retriever-ы, hard
  filters, LTR feature extraction, CatBoost и финальный top-50.
- Process/model/index startup и сеть измеряются отдельно.
- GPU timings требуют `torch.cuda.synchronize()` до и после участка.

Машиночитаемый контракт: `config/latency_guardrails.json`.

## Текущий результат

На Intel Core Ultra 7 155H исходный BM25 online path дважды вычислял один и тот
же corpus score vector для global/local channels:

- mean: `200.56 ms`;
- p95: `249.40 ms`.

После переиспользования одного score vector:

- 500 benchmark queries, deterministic sample seed 42;
- mean: `111.95 ms`;
- p50: `110.21 ms`;
- p95: `128.59 ms`;
- p99: `135.52 ms`;
- max: `151.53 ms`.

BM25 проходит исследовательский guardrail 500 ms и почти укладывается в общий
stretch 100 ms, но end-to-end статус остаётся **partial**. Нельзя складывать
batch throughput из экспериментальных notebooks и называть его online latency.
После завершения fine-tuned dense этапа необходим отдельный прогретый GPU-тест,
а принятый LTR должен сохранять model artifact для реального `predict`-замера.

## BM25: инвертированный индекс в памяти

`SparseBM25.score` больше не умножает всю матрицу документ×термин (33.6M
ненулевых) на вектор запроса. При загрузке строится CSC-копия — posting-списки
терминов (0.5 s, в памяти), и запрос складывает только списки своих терминов в
порядке возрастания id термина с float32-накоплением: ровно тот порядок и та
точность, которые использует SciPy в `matrix @ query.T`. Проверка на всех
4 904 validation и benchmark запросах: score-векторы бит-в-бит равны прежним,
rankings совпадают, поэтому все rankings, отчёты и submissions остаются
валидными.

| BM25 | до | после |
|---|---|---|
| scoring, p50 / p95 | 97.6 / 120.3 ms | 1.5 / 6.2 ms |
| вся стадия (scoring + фильтры + top-k + global/local RRF), p50 / p95 | 110.2 / 128.6 ms | 12.5 / 18.4 ms |

Оставшееся время стадии — в основном построение фильтров и top-k на pandas.
В end-to-end таблице попытки 2 ниже BM25 ещё прежний (148.7 ms p95);
итоговый путь будет перемерен целиком после ускорения dense-части.

Фактические hardware metadata и числа: `reports/latency_bm25.json`.

## End-to-end: попытка 2 на CPU

`scripts/benchmark_pipeline_latency.py` измеряет полный online-путь текущего
submission: BM25 global/local, два прохода USER-bge-m3 по запросу (zero-shot
и fine-tuned), точный global + location-local поиск по 554 920 passages для
каждого, свёртку passages в items с category filter и RRF top-50. Протокол тот
же: 25 warm-up, 500 benchmark-запросов, seed 42, batch=1.

Эмбеддинги passages синтетические: kernels сохранили rankings, а не матрицы,
и точный поиск по float32-матрице той же формы стоит одинаково при любых
значениях. Карта passage→item и размеры локаций настоящие. Fine-tuned encoder
замерен с базовыми весами: архитектура идентична.

| Стадия | p95, ms |
|---|---|
| BM25 global/local | 148.7 |
| encode zero-shot | 288.9 |
| search zero-shot | 59.1 |
| encode fine-tuned | 297.4 |
| search fine-tuned | 56.5 |
| RRF top-50 | 1.5 |
| **end-to-end** | **811.6** (p50 642.8, p99 862.9) |

**Попытка 2 не проходит исследовательский guardrail 500 ms.** Около 600 ms
занимают два прохода 568M-encoder по короткому запросу; число потоков это не
исправляет (16 потоков — лучший вариант, p95 273 ms на один проход). Любое
улучшение recall должно сначала вернуть пайплайн в бюджет либо оплатить себя
внутри него. Числа: `reports/latency_attempt2_cpu.json`.
