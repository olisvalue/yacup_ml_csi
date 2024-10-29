import glob
import json
import os
import re
from typing import Dict, List, Tuple

import jsonlines
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from models.data_loader import cover_dataloader
from models.data_model import Postfix

import wandb


class Zero:
    def __init__(self):
        self.value = 0.0
    def __float__(self):
        return self.value
    def item(self):
        return self.value

def reduce_func(D_chunk, start):
    top_size = 100
    nearest_items = np.argsort(D_chunk, axis=1)[:, :top_size + 1]
    return [(i, items[items!=i]) for i, items in enumerate(nearest_items, start)]

def dataloader_factory(config: Dict, data_split: str) -> List[DataLoader]:
    return cover_dataloader(
        data_path=config["data_path"],
        file_ext=config["file_extension"],
        # dataset_path=config[data_split]["dataset_path"],
        data_split=data_split,
        debug=config["debug"],
        max_len=50,
        **config[data_split]
    )

def calculate_ranking_metrics(embeddings: np.ndarray, cliques: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(embeddings)
    s_distances = np.argsort(distances, axis=1)
    cliques = np.array(cliques)
    query_cliques = cliques[s_distances[:, 0]]
    search_cliques = cliques[s_distances[:, 1:]]

    query_cliques = np.tile(query_cliques, (search_cliques.shape[-1], 1)).T
    mask = np.equal(search_cliques, query_cliques)

    ranks = 1.0 / (mask.argmax(axis=1) + 1.0)

    cumsum = np.cumsum(mask, axis=1)
    mask2 = mask * cumsum
    mask2 = mask2 / np.arange(1, mask2.shape[-1] + 1)
    average_precisions = np.sum(mask2, axis=1) / np.sum(mask, axis=1)

    return (ranks, average_precisions)

def calculate_ranking_metrics_batched(embeddings: np.ndarray, cliques: List[int], batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    num_samples = embeddings.shape[0]
    ranks = []
    average_precisions = []

    cliques = np.array(cliques)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_embeddings = embeddings[start_idx:end_idx]

        distances = pairwise_distances(batch_embeddings, embeddings)
        s_distances = np.argsort(distances, axis=1)

        batch_query_cliques = cliques[s_distances[:, 0]]
        batch_search_cliques = cliques[s_distances[:, 1:]]

        batch_query_cliques = np.tile(batch_query_cliques, (batch_search_cliques.shape[-1], 1)).T
        mask = np.equal(batch_search_cliques, batch_query_cliques)

        batch_ranks = 1.0 / (mask.argmax(axis=1) + 1.0)

        cumsum = np.cumsum(mask, axis=1)
        mask2 = mask * cumsum
        mask2 = mask2 / np.arange(1, mask2.shape[-1] + 1)
        batch_average_precisions = np.sum(mask2, axis=1) / np.sum(mask, axis=1)

        ranks.extend(batch_ranks)
        average_precisions.extend(batch_average_precisions)

    return np.array(ranks), np.array(average_precisions)



def dir_checker(output_dir: str) -> str:
    output_dir = re.sub(r"run-[0-9]+/*", "", output_dir)
    runs = glob.glob(os.path.join(output_dir, "run-*"))
    if runs != []:
        max_run = max(map(lambda x: int(x.split("-")[-1]), runs))
        run = max_run + 1
    else:
        run = 0
    outdir = os.path.join(output_dir, f"run-{run}")
    return outdir

def save_test_predictions(predictions: List, output_dir: str) -> None:
    with open(os.path.join(output_dir, 'submission.txt'), 'w') as foutput:
        for query_item, query_nearest in predictions:
            foutput.write('{}\t{}\n'.format(query_item, '\t'.join(map(str,query_nearest))))

def save_predictions(outputs: Dict[str, np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key in outputs:
        if "_ids" in key:
            with jsonlines.open(os.path.join(output_dir, f"{key}.jsonl"), "w") as f:
                if len(outputs[key][0]) == 4:
                    for clique, anchor, pos, neg in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor, "positive_id": pos, "negative_id": neg})
                else:
                    for clique, anchor in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor})
        else:
            np.save(os.path.join(output_dir, f"{key}.npy"), outputs[key])



def save_logs(outputs: dict, output_dir: str, name: str = "log", use_wandb: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"{name}.jsonl")
    with jsonlines.open(log_file, "a") as f:
        f.write(outputs)
    if use_wandb:
        wandb.log({k: float(v) for k, v in outputs.items()})



def save_best_log(outputs: Postfix, output_dir: str, use_wandb: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "best-log.json")
    with open(log_file, "w") as f:
        json.dump(outputs, f, indent=2)
    if use_wandb:
        wandb.log({k: float(v) for k, v in dict(outputs).items()})

