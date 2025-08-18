import torch
from torch.utils.data import Dataset, dataloader

class GPTdatasset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target.ids = []

        token_ids = tokenizer.encode(txt) # 對文本進行編碼
        
        for i in range(0, len(token_ids)-max_length, stride): 
            """
                將token_ids分割成多個長度為max_length的重疊片段(滑動視窗法)
            """
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        """
            傳回資料集輸入序列的總列數
        """
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        """
            傳回指定索引(特定位置)的輸入序列和目標序列
        """
        return self.input_ids[idx], self.target_ids[idx]