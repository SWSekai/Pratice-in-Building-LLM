import torch
from torch.utils.data import Dataset, dataloader

class GPTdatasset(Dataset):
    def __init__(self, tokenizer, max_length, stride):
        self.input_ids = []
        self.target.ids = []
    
    