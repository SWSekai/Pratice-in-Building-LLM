import torch
from gpt_dataloader import create_dataloader
from text_processing import Txt_preprocessor

def create_embedding_layer(vocab_size= 50257, embedding_dim= 256):
    """
        建立詞嵌入層
    """
    token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim) # 建立詞嵌入層
    
    return token_embedding_layer

if __name__ == "__main__":
    vocab_size = 50257  # GPT-2的詞彙表大小
    embedding_dim = 256 # 詞嵌入維度

    raw_text = Txt_preprocessor().rawText
    data_loader = create_dataloader(
        raw_text,
        batch_size= 8,
        max_length= 4,
        stride= 4,
        shuffle= False
    )
    
    create_embedding_layer(vocab_size, embedding_dim)