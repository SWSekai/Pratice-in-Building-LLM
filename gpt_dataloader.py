import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken
from preprocessed_txt import Txt_preprocessor

class GPTDataset(Dataset):
    """
        批次載入自訂資料集
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

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

def create_dataloader(txt, 
                      batch_size= 4, 
                      max_length= 256,
                      stride= 128, 
                      shuffle= True,
                      drop_last= True,
                      num_worker= 0):
    """
        批次載入資料載入器
    """
    tokenizer = tiktoken.get_encoding("gpt2") # 使用GPT-2的預訓練分詞器
    dataset = GPTDataset(txt, tokenizer, max_length, stride) # 建立自訂資料集
    data_loader = DataLoader(dataset, 
                            batch_size= batch_size, 
                            shuffle= shuffle, 
                            drop_last= drop_last, # 丟棄最後一個不完整的批次
                            num_workers= 0 # 額外的子進程數量(Windows系統設為0)
                            )
    
    return data_loader
    

if __name__ == "__main__":
    preprocessed_txt = Txt_preprocessor()
    data_loader = create_dataloader(preprocessed_txt.rawText, batch_size= 1, max_length= 4, stride= 1, shuffle= False)
    data_iter = iter(data_loader) # 建立資料迭代器
    first_batch = next(data_iter) # 取得第一個批次
    print(first_batch)