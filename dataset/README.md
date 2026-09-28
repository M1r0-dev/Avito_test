# Local data directory

Place the task files here; Parquet files are intentionally ignored by Git.

- `benchmark_queries.parquet`, `benchmark_items.parquet` — enough to reproduce
  the submitted `answer.csv` with `python scripts/generate_answer.py --check`.
- `train.parquet` — needed only to rebuild everything from scratch (holdout,
  notebooks); see "Полное воспроизведение" in the root `README.md`.
