# %% [markdown]
# # Stage 8A — leakage-safe USER-bge-m3 adaptation (2×T4)
#
# Первый полный запуск не дошёл до обучения: два DDP worker одновременно
# загружали удалённый checkpoint и оставались внутри materialization весов до
# 12-часового лимита Kaggle. Поэтому эксперимент разделён на две сохранённые
# версии. Этот notebook делает только обучение и сохраняет модель; построение
# 554 920 passage embeddings выполняется отдельным notebook 8B.
#
# Инженерные решения:
#
# - validation query signatures удаляются anti-join до любого sampling;
# - один positive на item предотвращает одинаковые документы в роли in-batch
#   negatives;
# - один epoch и LoRA rank 16 заранее фиксированы как консервативная адаптация;
# - checkpoint скачивается один раз главным процессом, а DDP ranks читают его
#   локально и последовательно — это устраняет наблюдавшееся зависание;
# - пары сортируются по category, делая in-batch negatives содержательнее.

# %%
from __future__ import annotations

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
TOKEN_RE = re.compile(r"(?u)[\w]+(?:[-'][\w]+)*")


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


def query_text(row: object) -> str:
    return f"{clean(row.search_query)} {clean(row.search_infm_params_text)}".strip()


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


# %% [markdown]
# ## DDP training process
#
# Оба rank используют один уже скачанный snapshot. Загрузка большой модели
# сериализована барьерами: сначала rank 0, затем rank 1. Барьер после обеих
# загрузок отделяет startup от измеряемого времени обучения.

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
    def __len__(self): return len(self.queries)
    def __getitem__(self, index): return self.queries[index], self.documents[index]

def load_base_serially(model_path, rank):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base = None
    for owner in range(dist.get_world_size()):
        if rank == owner:
            print(f"rank={rank}: loading local model", flush=True)
            base = AutoModel.from_pretrained(
                model_path, local_files_only=True, torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            print(f"rank={rank}: local model loaded", flush=True)
        dist.barrier()
    return tokenizer, base

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--model-path',required=True)
    args=parser.parse_args()
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    rank=dist.get_rank();local_rank=int(os.environ['LOCAL_RANK']);world=dist.get_world_size()
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42+rank);random.seed(42+rank)
    frame=pd.read_parquet(args.pairs)
    dataset=PairDataset(frame)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=False,drop_last=True)
    tokenizer,base=load_base_serially(args.model_path,rank)
    base.gradient_checkpointing_enable();base.enable_input_require_grads();base.config.use_cache=False
    config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',target_modules=['query','key','value'])
    model=get_peft_model(base,config).to(local_rank)
    model=DDP(model,device_ids=[local_rank],find_unused_parameters=False)
    loader=DataLoader(dataset,batch_size=10,sampler=sampler,num_workers=2,pin_memory=True,drop_last=True)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4,weight_decay=0.01)
    total_steps=len(loader);warmup=max(1,int(total_steps*0.05))
    scheduler=get_cosine_schedule_with_warmup(optimizer,warmup,total_steps)
    scaler=torch.amp.GradScaler('cuda');running=0.0;started=time.perf_counter()
    if rank==0: print(f"training started: steps={total_steps}",flush=True)
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
            logits=(q@torch.cat(gathered,dim=0).T)/0.03
            targets=torch.arange(len(q),device=local_rank)+rank*len(q)
            loss=F.cross_entropy(logits,targets)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer);scaler.update();scheduler.step();running+=loss.item()
        if rank==0 and (step==1 or step%25==0):
            elapsed=time.perf_counter()-started
            print(f'train {step}/{total_steps} loss={running/step:.5f} elapsed_s={elapsed:.1f}',flush=True)
    dist.barrier()
    if rank==0:
        trainable=sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        merged=model.module.merge_and_unload()
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        merged.save_pretrained(output,safe_serialization=True);tokenizer.save_pretrained(output)
        (output/'training_metrics.json').write_text(json.dumps({
            'pairs':len(dataset),'steps':total_steps,'mean_loss':running/total_steps,
            'elapsed_seconds':time.perf_counter()-started,
            'lora_rank':16,'learning_rate':2e-4,'temperature':0.03,
            'trainable_parameters':trainable,'world_size':world,
        },indent=2))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__': main()
'''


# %%
def main() -> None:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"This experiment requires two GPUs, found {torch.cuda.device_count()}")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data = locate_data()
    output = Path("/kaggle/working")
    item_columns = [
        "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
        "item_category_id",
    ]
    items = pd.read_parquet(data / "benchmark_items.parquet", columns=item_columns)
    items["item_id"] = items.item_id.astype(str)
    manifest = pd.read_parquet(data / "validation_manifest.parquet")
    train = pd.read_parquet(data / "train.parquet", columns=QUERY_COLUMNS + ["item_id"])
    train["item_id"] = train.item_id.astype(str)

    marked = train.merge(
        manifest[QUERY_COLUMNS].drop_duplicates().assign(_validation=1),
        on=QUERY_COLUMNS, how="left",
    )
    eligible = marked[marked._validation.isna()].drop(columns="_validation")
    eligible = eligible[eligible.item_id.isin(set(items.item_id))].drop_duplicates()
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

    # Resolve every remote file before DDP starts. Workers receive only a local path.
    model_path = snapshot_download(MODEL)
    print(f"model snapshot ready: {model_path}", flush=True)
    train_script = output / "train_lora.py"
    train_script.write_text(TRAIN_SCRIPT, encoding="utf-8")
    tuned_model = output / "user_bge_m3_avito"
    environment = os.environ.copy()
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=2", str(train_script),
        "--pairs", str(pair_path), "--output", str(tuned_model),
        "--model-path", str(model_path),
    ], check=True, env=environment)

    metrics = json.loads((tuned_model / "training_metrics.json").read_text())
    run = {
        "stage": "train_only", "base_model": MODEL, "license": MODEL_LICENSE,
        "eligible_pairs_after_holdout_exclusion": len(eligible),
        "unique_training_items": len(pairs), "training": metrics,
        "chunk_words": CHUNK_WORDS, "overlap": CHUNK_OVERLAP,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    }
    (output / "finetuned_train_run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
