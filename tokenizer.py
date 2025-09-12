import re
from text_processing import Txt_preprocessor

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text):
        """
            text to token ID (tokenize)
        """
        preprocessing_text = re.split(r'([,.:;?_!"()\']|--|\s)', text) # 使用正則表達式分割文本
        preprocessed_text = [token.strip() for token in preprocessing_text if token.strip()] # 過濾掉空字符串
        tokens = [item if item in self.str_to_int
                                else "<|unk|>" for item in preprocessed_text] # 將未知詞替換為 <|unk|>
        
        ids = [self.str_to_int[s] for s in tokens]
        
        return ids
    
    def decode(self, ids):
        """
            token ID to text
        """
        text = " ".join([self.int_to_str[i] for i in ids]) # 使用空格連接
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # 去除標點符號前多餘空格
        
        return text.strip()  # 去除首尾空格
    
if __name__ == "__main__":
    file_path = 'the-verdict.txt'
    preprocessed_txt = Txt_preprocessor()
    
    tokenizer = SimpleTokenizer(preprocessed_txt.vocab)
    
    # ids = tokenizer.encode(file_path)
    ids = tokenizer.encode("Hello, do you like tea? <|endoftext|> In the snlit terraces of the palace.")
    print(ids)
    print(tokenizer.decode(ids))