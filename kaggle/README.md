# Kaggle GPU stage

`dense_gpu.py` is the reproducible accelerator stage. It expects the three raw
Parquet files as a private Kaggle dataset, requires two GPUs, encodes fixed
passages with the Russian-focused `deepvk/USER-bge-m3`, performs exact sharded cosine
search, and exports compact item-level rankings for validation and benchmark.

Raw data, model weights and embeddings are not committed. `scripts/fetch_kaggle_output.sh`
downloads only `dense_rankings.parquet`, `validation_labels.parquet`, and run metadata.

`splade/06_splade_gpu_experiment.ipynb` is the next accelerator experiment. It
uses the Russian-specific `naver/neuclir22-splade-ru` checkpoint and exports
wide rankings for controlled query/document pruning, chunking, and local-channel
ablations. Its CC BY-NC-SA 4.0 license is recorded explicitly. Use
`scripts/fetch_splade_output.sh` after the kernel completes.

Fine-tuning is split across committed Kaggle versions so a long corpus encoding
cannot discard a trained checkpoint at the 12-hour limit. Stage
`finetune/08a_finetune_train_gpu.ipynb` performs leakage-safe LoRA adaptation of
`deepvk/USER-bge-m3` on both T4 GPUs and persists the merged model. Stage 8B
is `finetune_retrieval/08b_finetuned_dense_retrieval.ipynb`; it consumes that
saved model and rebuilds filter-aware rankings. Use
`scripts/fetch_finetuned_dense_output.sh` after retrieval completes.
