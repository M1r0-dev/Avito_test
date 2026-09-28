# Kaggle GPU stages

GPU-этапы выполнены публичными Kaggle kernels на 2×T4. Их выходы (rankings)
лежат в Git LFS и скачиваются без аккаунта со сверкой sha256
(`scripts/fetch_public_kaggle_outputs.py`); GPU-обучение и энкодинг не
бит-в-бит повторяемы, поэтому воспроизводимый артефакт — сохранённый выход.
Raw-данные задачи подключаются к kernels как private dataset
`m1r0tvorxc/avito-candidate-data`; kernels находят входы по имени файла под
`/kaggle/input`, а не по slug.

| Код | Kernel | Что делает | В финале (попытка 7) |
|---|---|---|---|
| `dense_gpu.py`, `04_dense_gpu_experiment.ipynb` | `avito-russian-dense-candidate-retrieval` | zero-shot USER-bge-m3: passages → точный поиск → rankings | **да**, канал zero-shot |
| `finetune_v2_train_gpu.py`, `finetune_v2/` | `user-bge-m3-lora-v2` | LoRA v2 на 457 439 парах train (stage 10A) | **да**, encoder LoRA v2 |
| `finetune_v2_retrieval_gpu.py`, `finetune_v2_retrieval/` | `lora-v2-dense-retrieval` | поиск merged LoRA v2, rankings + FP16 векторы (stage 10B) | **да**, канал LoRA v2 |
| `finetune_train_gpu.py`, `finetune_dense_gpu.py`, `finetune/`, `finetune_retrieval/` | `avito-user-bge-m3-domain-adaptation`, `avito-finetuned-dense-retrieval` | LoRA v1 и её rankings (stages 8A/8B) | нет, использовалась в попытках 1–4 |
| `splade_gpu.py`, `splade/` | `russian-splade-candidate-retrieval` | SPLADE (`naver/neuclir22-splade-ru`, CC BY-NC-SA 4.0) | нет, отклонён в notebook 06 |
| `multiview_v2_gpu.py`, `multiview_v2/` | `lora-v2-multi-view-retrieval` | дополнительные views запроса/объявления | нет, отклонено в notebook 21 |
| `recall50_selector_gpu.py`, `recall50_selector/` | `recall50-selector-objective` | поиск Recall@50-лосса selector (stage 25B) | лосс — да; сам прогон остановлен, досчитан на CPU (`notebooks/25b_recall50_lambda_cpu.ipynb`) |

Обучение LoRA разделено на отдельные kernels (обучение → поиск), чтобы долгий
энкодинг корпуса не терял обученный checkpoint при 12-часовом лимите Kaggle.
