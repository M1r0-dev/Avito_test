# %% [markdown]
# # Stage 10A — LoRA v2: USER-bge-m3 on all usable train pairs (2×T4)
#
# **Зачем v2.** Аудит v1 (notebook 18) показал, что fine-tune даёт
# +0.044 Recall@50 на head-запросах, но на хвосте (train frequency `<=1`, 72%
# benchmark) прирост не доказан: +0.011, 95% CI [-0.025; +0.047]. При этом v1
# видел 17 033 пары из 457 439 доступных: 8A брал только пары с объявлениями из
# benchmark-корпуса и по одной паре на объявление. Для bi-encoder это не нужно —
# тексты объявлений есть в `train.parquet` у 100% строк.
#
# **Что меняется относительно v1 (зафиксировано до запуска):**
#
# 1. данные — все пары train, кроме тех, что вредят обучению:
#    - сигнатуры holdout-запросов (утечка в оценку) — anti-join до всего;
#    - пустой текст запроса или объявления (нечему учиться);
#    - точные дубли `(текст запроса, item_id)`;
#
#    Потолок пар на текст запроса сознательно не вводится: потолок 32 оставил
#    бы 98.3% текстов, но лишь 66.6% пар — треть данных, почти целиком из 1.7%
#    частых текстов (у «маникюр …» 3 634 разных объявления). Эти пары несут
#    разные объявления и учат сторону документов; единственный их вред —
#    одинаковый текст как ложный негатив в batch — снимает маска из пункта 2.
#    Итог: 457 439 пар, 106 817 текстов запросов;
# 2. маска ложных негативов: in-batch пара с тем же текстом запроса или тем же
#    объявлением — не негатив (v1 гарантировал это только для объявлений,
#    ценой отказа от 96% данных);
# 3. batch 32 на GPU вместо 10 — 63 in-batch negatives вместо 19;
# 4. LoRA r=32 на query/key/value и всех dense-слоях энкодера (attention output
#    и FFN) вместо r=16 только на q/k/v — ёмкость под в 27 раз больше данных;
# 5. adapter checkpoint каждые 1000 шагов — многочасовое обучение не должно
#    теряться при сбое.
#
# Не меняются: базовая модель и ревизия, CLS-pooling, температура 0.03,
# длины 96/256 токенов, выбор positive passage по lexical overlap, 1 epoch,
# lr 2e-4 cosine. Retrieval и сохранение эмбеддингов корпуса — stage 10B.

# %%
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

# Same Kaggle image workaround as stage 8A: torchao 0.10 breaks Transformers.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.12,<1"], check=True)

MODEL = "deepvk/USER-bge-m3"
MODEL_REVISION = "0cc6cfe48e260fb0474c753087a69369e88709ae"
MODEL_LICENSE = "apache-2.0"
QUERY_COLUMNS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]
ITEM_TEXT_COLUMNS = ["item_title_raw", "item_description_raw", "item_infm_params_text"]
CHUNK_WORDS = 140
CHUNK_OVERLAP = 30
MAX_CHUNKS = 4
BATCH_PER_GPU = 32
LORA_RANK = 32
LEARNING_RATE = 2e-4
TEMPERATURE = 0.03
CHECKPOINT_EVERY = 1000
SEED = 42
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def tokenize(value: object) -> list[str]:
    return TOKEN_RE.findall(clean(value))


def item_passages(row: object) -> list[str]:
    """Identical to stages 8A/8B: title prefix + 140-word windows, max 4."""
    title = clean(row.item_title_raw)
    content = tokenize(row.item_infm_params_text) + ["описание"] + tokenize(row.item_description_raw)
    content = content or [""]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [
        f"{title}. {' '.join(content[start:start + CHUNK_WORDS])}".strip()
        for start in range(0, len(content), step)
    ][:MAX_CHUNKS]


def query_text(row: object) -> str:
    return f"{clean(row.search_query)} {clean(row.search_infm_params_text)}".strip()


def choose_positive_passage(query: str, row: object) -> str:
    query_terms = set(tokenize(query))
    passages = item_passages(row)
    scores = [len(query_terms & set(tokenize(passage))) for passage in passages]
    return passages[int(np.argmax(scores))]


def stable_id(value: str) -> int:
    """63-bit id for in-batch collision masks (identical across DDP ranks)."""
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little") >> 1


def locate_data() -> Path:
    roots = list(Path("/kaggle/input").rglob("benchmark_items.parquet"))
    if not roots:
        raise FileNotFoundError("benchmark_items.parquet not found under /kaggle/input")
    return roots[0].parent


# %% [markdown]
# ## DDP training process
#
# Загрузка checkpoint сериализована по rank'ам (урок stage 8A). Маска ложных
# негативов строится из id текста запроса и id объявления, собранных со всех
# GPU вместе с эмбеддингами документов.

# %%
TRAIN_SCRIPT = r'''
from __future__ import annotations
import argparse, json, os, random, time
from datetime import timedelta
from pathlib import Path
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

class PairDataset(Dataset):
    def __init__(self, frame):
        self.queries = frame.query_text.tolist()
        self.documents = frame.document_text.tolist()
        self.query_ids = frame.query_id.tolist()
        self.item_ids = frame.item_key.tolist()
    def __len__(self): return len(self.queries)
    def __getitem__(self, index):
        return self.queries[index], self.documents[index], self.query_ids[index], self.item_ids[index]

def load_base_serially(model_path, rank):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base = None
    for owner in range(dist.get_world_size()):
        if rank == owner:
            # No pooler: CLS is read from last_hidden_state, and an unused
            # pooler.dense would receive a LoRA adapter without gradients.
            base = AutoModel.from_pretrained(
                model_path, local_files_only=True, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, add_pooling_layer=False,
            )
            print(f"rank={rank}: local model loaded", flush=True)
        dist.barrier()
    return tokenizer, base

def main():
    parser = argparse.ArgumentParser()
    for name in ("--pairs", "--output", "--model-path", "--config"):
        parser.add_argument(name, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank(); local_rank = int(os.environ["LOCAL_RANK"]); world = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg["seed"] + rank); random.seed(cfg["seed"] + rank)
    dataset = PairDataset(pd.read_parquet(args.pairs))
    # Pairs are pre-shuffled with a fixed seed; the sampler interleaves them over ranks.
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    tokenizer, base = load_base_serially(args.model_path, rank)
    base.gradient_checkpointing_enable(); base.enable_input_require_grads(); base.config.use_cache = False
    lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=2 * cfg["lora_rank"], lora_dropout=0.05, bias="none",
                      target_modules=["query", "key", "value", "dense"])
    model = get_peft_model(base, lora).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    loader = DataLoader(dataset, batch_size=cfg["batch_per_gpu"], sampler=sampler, num_workers=2,
                        pin_memory=True, drop_last=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg["learning_rate"], weight_decay=0.01)
    total_steps = len(loader); warmup = max(1, int(total_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda"); running = 0.0; masked_total = 0; started = time.perf_counter()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if rank == 0: print(f"training started: steps={total_steps}", flush=True)
    model.train()
    for step, (queries, documents, query_ids, item_ids) in enumerate(loader, 1):
        qtok = tokenizer(list(queries), padding=True, truncation=True, max_length=96, return_tensors="pt").to(local_rank)
        dtok = tokenizer(list(documents), padding=True, truncation=True, max_length=256, return_tensors="pt").to(local_rank)
        query_ids = query_ids.to(local_rank); item_ids = item_ids.to(local_rank)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            q = F.normalize(model(**qtok).last_hidden_state[:, 0].float(), p=2, dim=1)
            d = F.normalize(model(**dtok).last_hidden_state[:, 0].float(), p=2, dim=1)
        gathered = [torch.zeros_like(d) for _ in range(world)]
        dist.all_gather(gathered, d.detach()); gathered[rank] = d
        all_query_ids = [torch.zeros_like(query_ids) for _ in range(world)]
        all_item_ids = [torch.zeros_like(item_ids) for _ in range(world)]
        dist.all_gather(all_query_ids, query_ids); dist.all_gather(all_item_ids, item_ids)
        all_query_ids = torch.cat(all_query_ids); all_item_ids = torch.cat(all_item_ids)
        logits = (q @ torch.cat(gathered, dim=0).T) / cfg["temperature"]
        targets = torch.arange(len(q), device=local_rank) + rank * len(q)
        # False negatives: same query text or same item as the query's own pair.
        collision = (query_ids[:, None] == all_query_ids[None, :]) | (item_ids[:, None] == all_item_ids[None, :])
        collision[torch.arange(len(q), device=local_rank), targets] = False
        masked_total += int(collision.sum())
        logits = logits.masked_fill(collision, torch.finfo(logits.dtype).min)
        loss = F.cross_entropy(logits, targets)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step(); running += loss.item()
        if rank == 0 and (step == 1 or step % 50 == 0):
            elapsed = time.perf_counter() - started
            print(f"train {step}/{total_steps} loss={running/step:.5f} masked={masked_total} "
                  f"elapsed_s={elapsed:.1f} eta_s={elapsed / step * (total_steps - step):.0f}", flush=True)
        if rank == 0 and step % cfg["checkpoint_every"] == 0:
            model.module.save_pretrained(output / "adapter_checkpoint")
            print(f"adapter checkpoint saved at step {step}", flush=True)
    dist.barrier()
    if rank == 0:
        trainable = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        model.module.save_pretrained(output / "adapter_final")
        merged = model.module.merge_and_unload()
        merged.save_pretrained(output, safe_serialization=True); tokenizer.save_pretrained(output)
        (output / "training_metrics.json").write_text(json.dumps({
            **cfg, "pairs": len(dataset), "steps": total_steps, "mean_loss": running / total_steps,
            "masked_false_negatives_rank0": masked_total,
            "elapsed_seconds": time.perf_counter() - started,
            "trainable_parameters": trainable, "world_size": world,
        }, indent=2))
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''


# %% [markdown]
# ## Отбор пар
#
# Порядок фильтров фиксирован и печатается с числом пар после каждого шага,
# чтобы в логе было видно, сколько данных убрал каждый фильтр.

# %%
def build_pairs(data: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    manifest = pd.read_parquet(data / "validation_manifest.parquet")
    train = pd.read_parquet(data / "train.parquet", columns=QUERY_COLUMNS + ["item_id", *ITEM_TEXT_COLUMNS])
    train["item_id"] = train.item_id.astype(str)
    report = {"train_rows": len(train)}

    marked = train.merge(manifest[QUERY_COLUMNS].drop_duplicates().assign(_validation=1),
                         on=QUERY_COLUMNS, how="left")
    pairs = marked[marked._validation.isna()].drop(columns="_validation")
    report["after_holdout_signature_exclusion"] = len(pairs)

    pairs["query_text"] = [query_text(row) for row in pairs.itertuples(index=False)]
    has_item_text = (pairs.item_title_raw.fillna("").str.strip().ne("")
                     | pairs.item_description_raw.fillna("").str.strip().ne(""))
    pairs = pairs[pairs.query_text.ne("") & has_item_text]
    report["after_empty_text_removal"] = len(pairs)

    pairs = pairs.drop_duplicates(["query_text", "item_id"])
    report["after_exact_duplicate_removal"] = len(pairs)

    # Seeded shuffle fixes the training order (the DDP sampler does not reshuffle).
    pairs = pairs.sample(frac=1, random_state=SEED)
    report["unique_query_texts"] = int(pairs.query_text.nunique())
    report["unique_items"] = int(pairs.item_id.nunique())

    leakage = pairs.merge(manifest[QUERY_COLUMNS].drop_duplicates(), on=QUERY_COLUMNS, how="inner")
    assert leakage.empty, "holdout signatures leaked into training pairs"

    pairs["document_text"] = [
        choose_positive_passage(query, row)
        for query, row in zip(pairs.query_text, pairs.itertuples(index=False))
    ]
    pairs["query_id"] = pairs.query_text.map(stable_id).astype("int64")
    pairs["item_key"] = pairs.item_id.map(stable_id).astype("int64")
    return pairs.reset_index(drop=True), report


# %%
def main() -> None:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"This experiment requires two GPUs, found {torch.cuda.device_count()}")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = Path("/kaggle/working")
    pairs, report = build_pairs(locate_data())
    print(json.dumps(report, indent=2), flush=True)
    pair_path = output / "finetune_v2_pairs.parquet"
    pairs[["query_text", "document_text", "query_id", "item_key"]].to_parquet(pair_path, index=False)

    # Resolve every remote file before DDP starts. Workers receive only a local path.
    model_path = snapshot_download(MODEL, revision=MODEL_REVISION)
    config = {
        "seed": SEED, "batch_per_gpu": BATCH_PER_GPU, "lora_rank": LORA_RANK,
        "learning_rate": LEARNING_RATE, "temperature": TEMPERATURE,
        "checkpoint_every": CHECKPOINT_EVERY,
    }
    train_script = output / "train_lora_v2.py"
    train_script.write_text(TRAIN_SCRIPT, encoding="utf-8")
    tuned_model = output / "user_bge_m3_avito_v2"
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=2", str(train_script),
        "--pairs", str(pair_path), "--output", str(tuned_model),
        "--model-path", str(model_path), "--config", json.dumps(config),
    ], check=True, env=os.environ.copy())
    # Pair texts are derived from the private train data; keep only ids/counts.
    pair_path.unlink()

    metrics = json.loads((tuned_model / "training_metrics.json").read_text())
    run = {
        "stage": "train_only_v2", "base_model": MODEL, "revision": MODEL_REVISION,
        "license": MODEL_LICENSE, "pair_selection": report, "training": metrics,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    }
    (output / "finetuned_v2_train_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False, indent=2), flush=True)


# %%
main()
