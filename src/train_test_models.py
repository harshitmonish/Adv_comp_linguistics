"""
Use this file as a starting point to understand how to load in the data.
"""

import click
import pickle
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

TRAIN_PATH = './data/train.jsonl'
TEST_PATH = './data/test.jsonl'
VECS_PATH = './data/buckeye.vecs'
BATCH_SIZE = 64
SHUFFLE_DATA = True


@click.command()
@click.option("--train_path", default=TRAIN_PATH)
@click.option("--test_path", default=TEST_PATH)
@click.option("--embeddings_path", default=VECS_PATH)
@click.option("--batch_size", default=BATCH_SIZE)
@click.option("--shuffle_data", default=SHUFFLE_DATA)
def main(train_path, test_path, embeddings_path, batch_size, shuffle):
    training_data = BuckeyeDataset(train_path, embeddings_path)
    test_data = BuckeyeDataset(test_path, embeddings_path)
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x
    )


def read_jsonl_file(path: str):
    data = []
    with open(path, 'r') as fid:
        for line in fid:
            data.append(json.loads(line))
    return data


def compute_log_duration(record):
    return np.log(
        np.sum(record['segment_duration_ms']))


def extract_phones(record):
    return record['observed_pron'].split(" ")


def load_vecs(path):
    with open(path, 'rb') as fid:
        return pickle.load(fid)


def get_embedding(record, vecs):
    return vecs[record['word']]


def read_record(record, vecs):
    phones = extract_phones(record)
    log_duration = compute_log_duration(record)
    embedding = get_embedding(record, vecs)
    return phones, embedding, log_duration


class BuckeyeDataset(Dataset):
    def __init__(self, jsonl_path, embeddings_path):
        self.records = read_jsonl_file(jsonl_path)
        self.vecs = load_vecs(embeddings_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        phones, embedding, log_duration = read_record(record, self.vecs)
        return phones, embedding, log_duration

if __name__=="__main__":
    main()
