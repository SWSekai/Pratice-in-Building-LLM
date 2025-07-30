import re

class SimpletokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text):
        """
            text to token ID
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text) # 使用正則表達式分割文本
        preprocessed_text = [token for token in tokens if token.strip()] # 過濾掉空字符串
        ids = [self.str_to_int[s] for s in preprocessed_text]
        
        return ids
    
    def decode(self, ids):
        """
            token ID to text
        """
        text = " ".join([self.int_to_str[i] for i in ids]) # 使用空格連接
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # 去除標點符號前多餘空格
        
        return text.strip()  # 去除首尾空格