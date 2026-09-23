from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from collections import defaultdict
import random


MAX_LENGTH = 512


class BaseDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                if line.strip():
                    self.samples.append(line)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        el = self.samples[idx]
        input_ids = self.tokenizer(el)['input_ids']
        input_ids = input_ids[:self.max_length]
        length = len(input_ids)
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        return torch.tensor(input_ids, dtype=torch.int64), length


class StandardDataset(Dataset):
    """
    See task desciption
    """
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        ### YOUR CODE HERE
        pass
    
    def __len__(self):
        ### YOUR CODE HERE
        pass

    def __getitem__(self, idx: int):
        ### YOUR CODE HERE
        pass


class SequencedDataset(IterableDataset):
    """
    See task desciption
    """
    def __init__(
        self, 
        data_path: str,
        tokenizer: AutoTokenizer,
        batch_size: int, 
        max_length: int = MAX_LENGTH
    ):
        ### YOUR CODE HERE
        pass
    
    def __iter__(self):
        ### YOUR CODE HERE
        pass

    def collate_data(self, batch):
        # Return a loss_mask in addition to the block-causal attention mask.
        # It must exclude transitions between adjacent packed samples.
        ### YOUR CODE HERE
        pass


def base_collate_fn(
    batch: list[tuple[str, torch.Tensor]],
    pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_processed = torch.stack([el[0] for el in batch])
    lengthes = torch.tensor([el[1] for el in batch])
    return {
        'tokens': batch_processed, 
        'lengthes': lengthes, 
        'attention_mask': None,
        'loss_mask': None,
    }


def collate_fn(
    batch: list[tuple[str, torch.Tensor]],
    pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    See task desciption
    """
    ### YOUR CODE HERE
    pass
    


class BalancedBatchSampler(Sampler):
    """
    See task desciption
    """
    def __init__(self, dataset, k: int, batch_size: int):
        ### YOUR CODE HERE
        pass

    def __len__(self):
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        ### YOUR CODE HERE
        pass
