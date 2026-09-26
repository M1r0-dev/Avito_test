# %% [markdown]
# # Domain adaptation of USER-bge-m3 on Avito clicks (2×T4)
#
# Этот notebook проверяет, улучшает ли supervised domain adaptation русский
# dense retriever. Из train полностью исключаются пять полей каждого validation
# query signature. Поэтому ни текст, ни фильтр, ни location/category конкретного
# holdout-запроса не могут участвовать в обучении.
#
# Выбор обучения:
#
# - LoRA rank 16 только на attention query/key/value сохраняет исходную русскую
#   модель и ограничивает число обучаемых параметров;
# - один epoch заранее фиксирован из-за 17k уникальных items и риска forgetting;
# - один positive на item исключает false negatives с одинаковым документом;
# - строки отсортированы по category, поэтому in-batch negatives труднее
#   случайных межкатегорийных, которые production hard-filter всё равно удалит;
# - positive passage из четырёх chunks выбирается максимальным lexical overlap
#   только внутри training pairs; inference остаётся честным по всем chunks.

# %%
from __future__ import annotations

import gc
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
from sentence_transformers import SentenceTransformer, models

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft>=0.12,<1"],
    check=True,
)

MODEL = "deepvk/USER-bge-m3"
MODEL_LICENSE = "apache-2.0"
QUERY_COLUMNS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]
CHUNK_WORDS = 140
CHUNK_OVERLAP = 30
MAX_CHUNKS = 4
MAX_LENGTH = 256
TOP_CHUNKS_PER_SHARD = 1500
TOP_ITEMS = 250
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")
RATING_RE = re.compile(r"рейтинг[^\d]{0,20}([1-5](?:[.,]\d+)?)", re.IGNORECASE)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("ё", "е").lower().split())


def tokenize(value: object) -> list[str]:
    return TOKEN_RE.findall(clean(value))


def item_passages(row: object) -> list[str]:
    title = clean(row.item_title_raw)
    content = tokenize(row.item_infm_params_text) + ["описание"] + tokenize(row.item_description_raw)
    content = content or [""]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [
        f"{title}. {' '.join(content[start:start + CHUNK_WORDS])}".strip()
        for start in range(0, len(content), step)
    ][:MAX_CHUNKS]


def make_passages(items: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    passages, item_rows = [], []
    for item_row, row in enumerate(items.itertuples(index=False)):
        current = item_passages(row)
        passages.extend(current)
        item_rows.extend([item_row] * len(current))
    return passages, np.asarray(item_rows, dtype=np.int32)


def query_text(row: object) -> str:
    return f"{clean(row.search_query)} {clean(row.search_infm_params_text)}".strip()


def requested_min_rating(value: object) -> float | None:
    match = RATING_RE.search(str(value))
    return float(match.group(1).replace(",", ".")) if match else None


def choose_positive_passage(query: str, row: object) -> str:
    query_terms = set(tokenize(query))
    passages = item_passages(row)
    scores = [len(query_terms & set(tokenize(passage))) for passage in passages]
    return passages[int(np.argmax(scores))]


def locate_data() -> Path:
    roots = list(Path("/kaggle/input").rglob("benchmark_items.parquet"))
    if not roots:
        raise FileNotFoundError("benchmark_items.parquet not found under /kaggle/input")
    return roots[0].parent


def encode_multi_gpu(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    if len(devices) != 2:
        raise RuntimeError(f"Expected two GPUs, found {devices}")
    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        result = model.encode(
            texts, pool=pool, batch_size=batch_size, chunk_size=4096,
            normalize_embeddings=True, show_progress_bar=True,
        )
    finally:
        model.stop_multi_process_pool(pool)
    return np.asarray(result, dtype=np.float32)


def gpu_top_chunks(query_embeddings: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
    shards = np.array_split(np.arange(len(passage_embeddings)), 2)
    corpora = [
        torch.from_numpy(passage_embeddings[rows]).to(f"cuda:{device}", dtype=torch.float16)
        for device, rows in enumerate(shards)
    ]
    all_indices = []
    for start in range(0, len(query_embeddings), 128):
        candidates = []
        scores = []
        for device, (rows, corpus) in enumerate(zip(shards, corpora)):
            query = torch.from_numpy(query_embeddings[start:start + 128]).to(
                f"cuda:{device}", dtype=torch.float16
            )
            values, local = torch.topk(
                query @ corpus.T, k=min(TOP_CHUNKS_PER_SHARD, corpus.shape[0]), dim=1
            )
            scores.append(values.float().cpu().numpy())
            candidates.append(rows[local.cpu().numpy()])
        merged_scores = np.concatenate(scores, axis=1)
        merged_indices = np.concatenate(candidates, axis=1)
        k = min(TOP_CHUNKS_PER_SHARD * 2, merged_scores.shape[1])
        keep = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
        batch_rows = np.arange(len(keep))[:, None]
        order = np.argsort(merged_scores[batch_rows, keep], axis=1)[:, ::-1]
        sorted_keep = np.take_along_axis(keep, order, axis=1)
        all_indices.append(merged_indices[batch_rows, sorted_keep].astype(np.int32))
        print(f"global search {min(start + 128, len(query_embeddings))}/{len(query_embeddings)}", flush=True)
    return np.concatenate(all_indices)


def gpu_local_chunks(
    queries: pd.DataFrame,
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    passage_item_rows: np.ndarray,
    items: pd.DataFrame,
) -> list[np.ndarray]:
    locations = items.item_location_id.to_numpy()[passage_item_rows]
    result = [np.array([], dtype=np.int32) for _ in range(len(queries))]
    for location, positions in queries.groupby("search_location_id").indices.items():
        passage_rows = np.flatnonzero(locations == int(location))
        if not len(passage_rows):
            continue
        corpus = torch.from_numpy(passage_embeddings[passage_rows]).to("cuda:0", dtype=torch.float16)
        positions = np.asarray(positions)
        for start in range(0, len(positions), 128):
            batch_positions = positions[start:start + 128]
            query = torch.from_numpy(query_embeddings[batch_positions]).to("cuda:0", dtype=torch.float16)
            local = torch.topk(query @ corpus.T, k=min(750, len(passage_rows)), dim=1).indices.cpu().numpy()
            for offset, position in enumerate(batch_positions):
                result[int(position)] = passage_rows[local[offset]].astype(np.int32)
        del corpus
    return result


def item_rankings(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    passage_item_rows: np.ndarray,
    global_chunks: np.ndarray,
    local_chunks: list[np.ndarray],
) -> pd.DataFrame:
    item_ids = items.item_id.astype(str).to_numpy()
    categories = items.item_category_id.to_numpy()
    ratings = items.item_rating.fillna(-1).to_numpy()
    records = []
    for query_no, query in enumerate(queries.itertuples(index=False)):
        category = int(query.search_category)
        min_rating = requested_min_rating(query.search_infm_params_text)

        def collapse(chunk_rows: np.ndarray) -> list[str]:
            answer, seen = [], set()
            for item_row in passage_item_rows[chunk_rows]:
                item_row = int(item_row)
                if category and categories[item_row] != category:
                    continue
                if min_rating is not None and ratings[item_row] < min_rating:
                    continue
                if item_row not in seen:
                    seen.add(item_row)
                    answer.append(item_ids[item_row])
                if len(answer) == TOP_ITEMS:
                    break
            return answer

        records.append({
            "query_key": str(query.query_key), "split": str(query.split),
            "finetuned_global": " ".join(collapse(global_chunks[query_no])),
            "finetuned_local": " ".join(collapse(local_chunks[query_no])),
        })
    return pd.DataFrame(records)


# %% [markdown]
# ## Leakage-safe training pairs
#
# Anti-join выполняется до выбора одного примера на item. Итоговый parquet
# содержит только query, положительный passage и category для формирования
# category-hard batches; никакие validation labels не передаются trainer-у.

# %%
TRAIN_SCRIPT = r'''
from __future__ import annotations
import argparse, json, math, os, random
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
    def __len__(self): return len(self.queries)
    def __getitem__(self, index): return self.queries[index], self.documents[index]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    dist.init_process_group('nccl')
    rank=dist.get_rank();local_rank=int(os.environ['LOCAL_RANK']);world=dist.get_world_size()
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42+rank);random.seed(42+rank)
    frame=pd.read_parquet(args.pairs)
    dataset=PairDataset(frame)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=False,drop_last=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    base=AutoModel.from_pretrained(args.model)
    base.gradient_checkpointing_enable();base.enable_input_require_grads();base.config.use_cache=False
    config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',target_modules=['query','key','value'])
    model=get_peft_model(base,config).to(local_rank)
    model=DDP(model,device_ids=[local_rank],find_unused_parameters=False)
    loader=DataLoader(dataset,batch_size=10,sampler=sampler,num_workers=2,pin_memory=True,drop_last=True)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4,weight_decay=0.01)
    total_steps=len(loader);warmup=max(1,int(total_steps*0.05))
    scheduler=get_cosine_schedule_with_warmup(optimizer,warmup,total_steps)
    scaler=torch.amp.GradScaler('cuda')
    running=0.0
    model.train()
    for step,(queries,documents) in enumerate(loader,1):
        qtok=tokenizer(list(queries),padding=True,truncation=True,max_length=96,return_tensors='pt').to(local_rank)
        dtok=tokenizer(list(documents),padding=True,truncation=True,max_length=256,return_tensors='pt').to(local_rank)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.float16):
            q=F.normalize(model(**qtok).last_hidden_state[:,0].float(),p=2,dim=1)
            d=F.normalize(model(**dtok).last_hidden_state[:,0].float(),p=2,dim=1)
            gathered=[torch.zeros_like(d) for _ in range(world)]
            dist.all_gather(gathered,d.detach());gathered[rank]=d
            all_d=torch.cat(gathered,dim=0)
            logits=(q@all_d.T)/0.03
            targets=torch.arange(len(q),device=local_rank)+rank*len(q)
            loss=F.cross_entropy(logits,targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer);scaler.update();scheduler.step()
        running+=loss.item()
        if rank==0 and step%100==0: print(f'train {step}/{total_steps} loss={running/step:.5f}',flush=True)
    dist.barrier()
    if rank==0:
        trainable=sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        merged=model.module.merge_and_unload()
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        merged.save_pretrained(output,safe_serialization=True);tokenizer.save_pretrained(output)
        (output/'training_metrics.json').write_text(json.dumps({
            'pairs':len(dataset),'steps':total_steps,'mean_loss':running/total_steps,
            'lora_rank':16,'learning_rate':2e-4,'temperature':0.03,
            'trainable_parameters':trainable,'world_size':world,
        },indent=2))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
'''


def main() -> None:
    if torch.cuda.device_count() != 2:
        raise RuntimeError("This experiment requires exactly two GPUs")
    data = locate_data()
    output = Path("/kaggle/working")
    item_columns = [
        "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
        "item_category_id", "item_location_id", "item_rating",
    ]
    items = pd.read_parquet(data / "benchmark_items.parquet", columns=item_columns)
    items["item_id"] = items.item_id.astype(str)
    manifest = pd.read_parquet(data / "validation_manifest.parquet")
    labels = pd.read_parquet(data / "validation_labels.parquet")
    train = pd.read_parquet(data / "train.parquet", columns=QUERY_COLUMNS + ["item_id"])
    train["item_id"] = train.item_id.astype(str)

    marked = train.merge(
        manifest[QUERY_COLUMNS].drop_duplicates().assign(_validation=1),
        on=QUERY_COLUMNS, how="left",
    )
    eligible = marked[marked._validation.isna()].drop(columns="_validation")
    eligible = eligible[eligible.item_id.isin(set(items.item_id))].drop_duplicates()
    # One document per batch epoch prevents identical documents acting as negatives.
    pairs = eligible.sample(frac=1, random_state=42).drop_duplicates("item_id")
    pairs = pairs.merge(items, on="item_id", how="inner", validate="one_to_one")
    leakage_check = pairs.merge(
        manifest[QUERY_COLUMNS].drop_duplicates(), on=QUERY_COLUMNS, how="inner"
    )
    assert pairs.item_id.is_unique and leakage_check.empty
    pairs["query_text"] = [query_text(row) for row in pairs.itertuples(index=False)]
    pairs["document_text"] = [
        choose_positive_passage(query, row)
        for query, row in zip(pairs.query_text, pairs.itertuples(index=False))
    ]
    pairs["_shuffle"] = np.random.default_rng(42).random(len(pairs))
    pairs = pairs.sort_values(["item_category_id", "_shuffle"])
    pair_path = output / "finetune_pairs.parquet"
    pairs[["query_text", "document_text", "item_category_id"]].to_parquet(pair_path, index=False)
    print(f"eligible={len(eligible):,}; unique training items={len(pairs):,}", flush=True)

    train_script = output / "train_lora.py"
    train_script.write_text(TRAIN_SCRIPT, encoding="utf-8")
    tuned_model = output / "user_bge_m3_avito"
    environment = os.environ.copy()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=2", str(train_script),
        "--pairs", str(pair_path), "--output", str(tuned_model), "--model", MODEL,
    ], check=True, env=environment)

    benchmark = pd.read_parquet(data / "benchmark_queries.parquet")
    benchmark["query_key"] = benchmark.query_id.astype(str)
    benchmark["split"] = "benchmark"
    validation = manifest.copy()
    validation["query_key"] = validation.eval_query_id.astype(str)
    validation["split"] = "validation"
    queries = pd.concat(
        [validation, benchmark[QUERY_COLUMNS + ["query_key", "split"]]], ignore_index=True
    )
    passages, passage_item_rows = make_passages(items)
    print(f"passages={len(passages):,}; avg/item={len(passages)/len(items):.2f}", flush=True)

    transformer = models.Transformer(str(tuned_model), max_seq_length=MAX_LENGTH)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
    )
    model = SentenceTransformer(modules=[transformer, pooling])
    passage_embeddings = encode_multi_gpu(model, passages, batch_size=48)
    query_embeddings = model.encode(
        [query_text(row) for row in queries.itertuples(index=False)],
        batch_size=128, device="cuda:0", normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    del model, passages
    gc.collect();torch.cuda.empty_cache()

    global_chunks = gpu_top_chunks(query_embeddings, passage_embeddings)
    local_chunks = gpu_local_chunks(
        queries, query_embeddings, passage_embeddings, passage_item_rows, items
    )
    rankings = item_rankings(queries, items, passage_item_rows, global_chunks, local_chunks)
    rankings.to_parquet(output / "finetuned_dense_rankings.parquet", index=False)
    labels[["eval_query_id", "item_id"]].drop_duplicates().to_parquet(
        output / "validation_labels.parquet", index=False
    )
    training_metrics = json.loads((tuned_model / "training_metrics.json").read_text())
    run = {
        "base_model": MODEL, "license": MODEL_LICENSE,
        "eligible_pairs_after_holdout_exclusion": len(eligible),
        "training": training_metrics,
        "items": len(items), "passages": len(passage_item_rows), "queries": len(queries),
        "chunk_words": CHUNK_WORDS, "overlap": CHUNK_OVERLAP, "max_chunks": MAX_CHUNKS,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    }
    (output / "finetuned_dense_run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Локальная оценка
#
# Kaggle экспортирует только item rankings и metadata, не модель и не embeddings.
# Локальный notebook выбирает global/local weight только на dev, сравнивает
# fine-tuned dense с исходным USER dense парным тестом и проверяет новый union.
# Если новый канал подтверждён, следующим отдельным экспериментом он добавляется
# в supervised LTR. Само дообучение не получает права заменить `answer.csv`.
