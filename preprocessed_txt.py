import re

class Txt_preprocessor:
    def __init__(self, file_path= 'the-verdict.txt'):
        self.file_path = file_path
        self.rawText = self.read_file()
        self.token = self.tokenize(self.rawText)
        self.vocab = self.token_to_id(self.token)
        
        print(f"字元數: {len(self.rawText)}")
        print(f"預處理後的字元數: {len(self.token)}")
        print(f"詞彙表大小: {len(self.vocab)}")
        
    def read_file(self):
        """
            讀取文本文件
        """
        with open(self.file_path, "r", encoding= "utf-8") as f:
            rawText = f.read()
        
        return rawText
    
    def tokenize(self, text):
        """
            Simple tokenizer function, 斷詞
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text) # 使用正則表達式分割文本
        
        token = [token for token in tokens if token.strip()] # 過濾掉空字符串
        
        return token

    def token_to_id(self, tokens):
        """
            transform token to token id
        """
        all_tokens = sorted(list(set(tokens)))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # 添加特殊標記
        vocab = {token: idx for idx, token in enumerate(all_tokens)} # 建立詞彙表
        
        return vocab

    def get_vocab_table(self):
        
        for i, idx in enumerate(self.vocab.items()):
            print(f"詞彙表索引: {i}, 詞彙: {idx[0]}, ID: {idx[1]}")

if __name__ == "__main__":
    preprocessor = Txt_preprocessor()
    # rawText, preprocessedText, vocab = preprocessor.get_vocab_table()
    
    