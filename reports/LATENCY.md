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

Фактические hardware metadata и числа: `reports/latency_bm25.json`.
