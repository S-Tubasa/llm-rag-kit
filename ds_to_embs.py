from __future__ import annotations
from dataclasses import dataclass
from typing import Generator
import numpy as np
from sentence_transformers import SentenceTransformer
from hf_hub_ctranslate2 import CT2SentenceTransformer

from tqdm import tqdm
import torch
import argparse
from pathlib import Path
from datasets import load_dataset

parser = argparse.ArgumentParser(
    description="Convert datasets to embeddings",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-d",
    "--ds_path",
    type=str,
    default="",
    help="target jsonl dataset path",
)

parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="",
    help="sentence transformer model path",
)

parser.add_argument(
    "-p",
    "--passage_prefix",
    type=str,
    required=False,
    default="passage: ",
    help="prefix string for passage, ' or 'query: ' or 'passage: ' or 'prefix-foo-bar: '",
)

parser.add_argument(
    "--passage_template",
    type=str,
    required=False,
    default="# {title}\n\n## {section}\n\n### {text}",
    help="template for passage, {title}, {section}, {text} are replaced",
)

parser.add_argument(
    "-l",
    "--max_seq_length",
    type=int,
    required=False,
    default=512,
    help="max sequence length",
)

parser.add_argument(
    "-w",
    "--working_dir",
    type=str,
    default="outputs",
    help="working_dir dir",
)

args = parser.parse_args()

@dataclass
class EmbConfig:
    model_name: str
    passage_prefix: str
    max_seq_length: int
    
TEMPLATE = args.passage_template
PASSAGE_PREFIX = args.passage_prefix
    
target_ds = load_dataset('json', data_files=args.ds_path, split="train")

def data_to_passage(data, template=TEMPLATE, prefix=PASSAGE_PREFIX):
    title = data["title"]
    section = data["section"]
    text = data["text"]
    formatted = template.format(title=title, section=section, text=text)
    return prefix + formatted

emb_config = EmbConfig(
    model_name=args.model_path,
    passage_prefix=args.passage_prefix,
    max_seq_length=args.max_seq_length,
)

MODEL = CT2SentenceTransformer(emb_config.model_name, compute_type="float16")

model_name_for_embs_dir = emb_config.model_name.split("/")[-1]

if "-e5-" in model_name_for_embs_dir:
    if emb_config.passage_prefix == "query: ":
        model_name_for_embs_dir += "-query"
    elif emb_config.passage_prefix == "passage: ":
        model_name_for_embs_dir += "-passage"
    else:
        raise ValueError("passage_prefix should be one of")

def to_embs(
    texts: list[str], group_size=1024, model=MODEL
) -> Generator[np.ndarray, None, None]:
    group = []
    for text in texts:
        group.append(text)
        if len(group) == group_size:
            embeddings = model.encode(  # type: ignore
                group,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            yield embeddings  # type: ignore
            group = []
    if len(group) > 0:
        embeddings = model.encode(  # type: ignore
            group, normalize_embeddings=True, show_progress_bar=False
        )
        yield embeddings  # type: ignore


def ds_to_embs(
    ds,
    text_fn,
    group_size: int,
):
    texts = []
    total = len(ds)
    pbar = tqdm(total=total)
    # text は group_size 件ごとに処理する
    for i in range(0, total, group_size):
        texts = []
        for data in ds.select(range(i, min(i + group_size, total))):
            data: dict = data
            text = text_fn(data)
            texts.append(text)
        embs = []
        for group_embs in to_embs(texts):
            embs.append(group_embs)
            pbar.update(len(group_embs))
        embs = np.concatenate(embs)
        yield embs, i, pbar

if torch.cuda.is_available():
    print("use cuda")
    MODEL.to("cuda")  # type: ignore
elif torch.backends.mps.is_available():  # type: ignore
    print("use mps (apple selicon)")
    MODEL.to("mps")  # type: ignore
else:
    print("!! Warning: use cpu")

ds = target_ds  # type: ignore

working_dir_embs_path = Path(
    "/".join([args.working_dir, "embs",model_name_for_embs_dir])
)

working_dir_embs_path.mkdir(parents=True, exist_ok=True)

print("output embs path:", working_dir_embs_path)

for embs, idx, pbar in ds_to_embs(ds, data_to_passage, group_size=10_000):
    filename = f"{idx}.npz"
    filepath = working_dir_embs_path / filename
    pbar.desc = f"saving...: {str(filepath)}"
    np.savez_compressed(filepath, embs=embs.astype(np.float16))
    pbar.desc = ""
