import re

class Txt_preprocessor:
    def __init__(self, file_path= 'the-verdict.txt'):
        self.file_path = file_path
        
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
        
        result = [token for token in tokens if token.strip()] # 過濾掉空字符串
        
        return result

    def token_to_id(self, tokens):
        """
            transform token to token id
        """
        all_tokens = sorted(list(set(tokens)))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # 添加特殊標記
        vocab = {token: idx for idx, token in enumerate(all_tokens)} # 建立詞彙表
        
        return vocab

    def get_txt_preprocessed(self):
        rawText = self.read_file()
        preprocessedText = self.tokenize(rawText)
        vocab = self.token_to_id(preprocessedText)
        
        return rawText, preprocessedText, vocab

if __name__ == "__main__":
    preprocessor = Txt_preprocessor()
    rawText, preprocessedText, vocab = preprocessor.get_txt_preprocessed()
    print(f"字元數: {len(rawText)}")
    print(f"預處理後的字元數: {len(preprocessedText)}")
    print(f"詞彙表大小: {len(vocab)}")