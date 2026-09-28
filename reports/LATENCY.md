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

## Финальное решение (попытка 7): guardrail пройден по контракту

**Итог:** online-путь попытки 7 на Intel Core Ultra 7 155H (CPU, batch=1,
500 benchmark-запросов, seed 42, warm-up 25 — ровно по контракту, parallel
замерен первым сразу после загрузки): **p50 176 ms, p95 357 ms, p99 682 ms**
(`reports/latency_final_cpu_attempt7_gcfreeze.json`). Sequential-режим того же
процесса: p50 291, p95 419, p99 475 ms. Recall@50 online-пути с финальным
selector совпадает с offline на test и test-tail (notebook 29).

Команда (нужны `fetch_public_kaggle_outputs.py --online` и два ONNX-экспорта):

```bash
pip install -e '.[latency]'
python scripts/fetch_public_kaggle_outputs.py --online
python scripts/export_onnx_encoder.py
python scripts/export_onnx_encoder.py --model artifacts/lora_v2_model/user_bge_m3_avito_v2 \
    --revision "" --output artifacts/onnx/lora_v2
python scripts/benchmark_final_latency.py --encoder onnx --threads 4 --blas-threads 4 \
    --catboost-threads 4 --gc-freeze --modes parallel
```

Что измеряется (`scripts/benchmark_final_latency.py`): BM25 (filtered + plain),
два query encoder USER-bge-m3 (zero-shot и LoRA v2) с точным global/local
поиском по 554 920 passages, RRF top-200, признаки selector, три CatBoost и
top-50. LoRA v2 ищет по настоящим сохранённым векторам; zero-shot — по
случайной матрице той же формы (векторы не сохранялись, стоимость точного
поиска от значений не зависит). Поэтому это замер времени, а не рабочий
сервис: ответы zero-shot-канала в замере бессмысленны, а сданный `answer.csv`
собран из сохранённых GPU-rankings. Эквивалентность по Recall@50 доказана для
CPU-поиска и для LoRA v2 целиком (notebooks 27–29), для zero-shot на его
настоящих векторах — нет.

### Что изменено и почему recall не меняется

| Изменение | Эффект | Почему recall тот же |
|---|---|---|
| Три ветки (BM25, zero-shot, LoRA v2) в параллельных потоках | end-to-end ≈ max ветки, а не сумма | выход идентичен: top-50 совпадает на всех 500 запросах (`identical_top50_across_modes`) |
| Encoder через ONNX Runtime fp32 с fused attention/GELU/LayerNorm (`scripts/export_onnx_encoder.py`) | encode p50 ~210 → ~75–115 ms | те же веса, без квантизации: косинус к PyTorch ≥ 0.9999992; весь online v2 (ORT + CPU-поиск) non-inferior к offline: test +0.0019, 97.5% CI [0, 0.0046], tail 0/0 (notebook 28) |
| Точный CPU-поиск: одно произведение на global и local | local без второго matmul | CPU fp32 против GPU fp16: non-inferior, test +0.0016, tail 0/0 (notebook 27) |
| Бюджет потоков: ORT 4, MKL 4, CatBoost 4, `KMP_BLOCKTIME=0`, без spinning | ~9 активных потоков вместо переподписки (2×22 MKL + ORT + CatBoost); отдельный эффект не выделен — прогоны A и F шли в разных условиях окружения | влияет только на расписание потоков |
| `gc.collect(); gc.freeze()` после загрузки (`--gc-freeze`) | сборщик мусора перестаёт обходить миллионы загруженных объектов (кеш признаков объявлений, BM25): p95 первого окна после загрузки 699 → 357 ms в одинаковых условиях | не касается вычислений |

Попутно найдена ошибка интеграции: dense-каналы Kaggle кодировали запрос без
префикса `"query: "`, а `query_text` его добавляет (он нужен BM25). С
префиксом векторы v2 расходились (косинус ~0.93) и test Recall@50 падал на
0.006; теперь dense-каналы используют `dense_query_text`.

### Прогоны с финальным selector (попытка 7)

Docker-контейнеры остановлены, warm-up 25 по контракту, parallel замерен
первым после загрузки.

| Конфигурация | Режим | Порядок | p50 | p95 | p99 | ≤ 500 |
|---|---|---|---|---|---|---|
| ORT + бюджет потоков | parallel | 1 | 186 | 699 | 845 | ✗ |
| ORT + бюджет потоков | sequential | 2 | 302 | 419 | 463 | ✓ |
| **ORT + бюджет потоков + `gc.freeze`** | **parallel** | 1 | **176** | **357** | 682 | ✓ |
| ORT + бюджет потоков + `gc.freeze` | sequential | 2 | 291 | 419 | 475 | ✓ |

Отчёты: `latency_final_cpu_attempt7.json`, `latency_final_cpu_attempt7_gcfreeze.json`.
Разница с `gc.freeze` и без — по одному прогону каждого, но в одинаковых
условиях и того же размера, что и всплески: это согласуется с тем, что
«плохое первое окно» давали паузы полного прохода GC по объектам, созданным
при загрузке, а не нехватка памяти.

### Прогоны с selector попытки 6 (история оптимизации)

Каждая строка — 500 измеряемых запросов; «Порядок» — какой по счёту режим в
процессе после загрузки ~13–16 ГБ индексов и моделей.

| Конфигурация | Режим | Порядок | Warm-up | p50 | p95 | p99 | ≤ 500 |
|---|---|---|---|---|---|---|---|
| PyTorch, 16 потоков | sequential | 1 | 25 | 584 | 872 | 1713 | ✗ |
| PyTorch, 16 потоков | parallel | 2 | 25 | 461 | 649 | 808 | ✗ |
| ORT, 4 потока (A) | sequential | 1 | 25 | 377 | 1285 | 1625 | ✗ |
| ORT, 4 потока (A) | parallel | 2 | 25 | 198 | 278 | 329 | ✓ |
| ORT, 4 потока (B) | parallel | 1 | 25 | 725 | 1138 | 1249 | ✗ |
| ORT, 4 потока (B) | sequential | 2 | 25 | 336 | 446 | 520 | ✓ |
| ORT, 4 потока (C) | parallel | 1 | 500 | 186 | 901 | 1146 | ✗ |
| ORT, 4 потока (D) | parallel | 1 | 500 | 175 | 724 | 865 | ✗ |
| ORT + бюджет потоков (E) | parallel | 1 | 25 | 235 | 918 | 1053 | ✗ |
| ORT + бюджет потоков (E) | sequential | 2 | 25 | 310 | 415 | 462 | ✓ |
| ORT + бюджет потоков (F) | sequential | 1 | 1500 | 330 | 454 | 529 | ✓ |
| **ORT + бюджет потоков (F)** | **parallel** | 2 | 1500 | **282** | **398** | **471** | ✓ |

Отчёты: `reports/latency_final_cpu_torch.json`, `latency_final_cpu_onnx*.json`.

**Почему первые окна были плохими (до `gc.freeze`).** Во время этих замеров swap ноутбука был заполнен
(8/8 ГБ), ~15 ГБ RAM занимали IDE, браузер, neo4j и docker-контейнеры, часть
которых перезапускалась в цикле. Пока процесс после загрузки вытесняет чужие
страницы, всплески задевают **все** стадии одновременно, включая CatBoost по
~175 строкам, — это остановки процесса, а не медленный код. 25 и даже 500
warm-up запросов этого не снимали; после ~1 500 (прогон F) оба режима проходят,
в том числе замеренный первым. Warm-up 1 500 — отклонение от контракта (25),
оправданное тем, что контракт измеряет warm steady state. Позже выяснилось,
что главную роль играли паузы GC: с `--gc-freeze` контракт с warm-up 25
выполняется и для режима, замеренного первым (таблица выше).

**Production.** Guardrail ноутбука выполнен без его ослабления, поэтому
пересчёт production-бюджета на это железо не понадобился. GPU stretch
(`≤ 100 ms`) не измерялся: на нём оба encoder и точный поиск переносятся на
GPU, а CPU-часть (BM25, признаки, CatBoost) занимает ~43 ms p50 в
sequential-режиме.

## История: BM25 one-pass


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
