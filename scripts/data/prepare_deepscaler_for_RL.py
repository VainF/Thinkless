"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from deepscaler.data.utils import load_dataset
from deepscaler.data.dataset_types import TrainDataset, TestDataset
import datasets

import random

def prepare_deepscaler():
    ds = datasets.load_dataset("agentica-org/DeepScaleR-Preview-Dataset", "default")
    data_items = []
    for idx, line in enumerate(ds['train']):
        question = line['problem']
        answer = line['answer']

        # Add the following instruction may make the training sensitive to hyperparameters.
        # Need more investigation.
        #instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        #question = f"{instruction}\n{question}"

        # Uncomment the following lines if the answer is within \\boxed{} and need to be extracted
        #answer = extract_answer(answer_raw)
        #if answer is None:
        #    continue

        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': 'train',
                'index': idx
            }
        }
        data_items.append(data)
    #random.shuffle(data_items)
    print(data_items[:10])
    return data_items

def prepare_aime24():
    ds = datasets.load_dataset("HuggingFaceH4/aime_2024", "default")
    data_items = []
    for idx, line in enumerate(ds['train']):
        question = line['problem']
        answer = line['answer']

        #instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        #question = f"{instruction}\n{question}"

        # Uncomment the following lines if the answer is within \\boxed{} and need to be extracted
        #answer = extract_answer(answer_raw)
        #if answer is None:
        #    continue

        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': 'train',
                'index': idx
            }
        }
        data_items.append(data)
    return data_items

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('./data/deepscaler'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir)

    # Process training data
    train_data = prepare_deepscaler()
    val_data = prepare_aime24()

    # Save validation dataset
    val_df = pd.DataFrame(val_data)
    val_df.to_parquet(os.path.join(local_dir, f'aime24.parquet'))
    print(f"AIME-24 val data size:", len(val_data))

    # Save training dataset
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    print("DeepScaler data size:", len(train_data))

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)