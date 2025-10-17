import re

class Txt_preprocessor:
    def __init__(self, file_path= 'the-verdict.txt'):
        self.file_path = file_path
        self.rawText = self.__read_file__()
        self.token = self.__tokenize__(self.rawText)
        self.vocab = self.__token_to_id__(self.token)
        
        print(f"字元數: {len(self.rawText)}")
        print(f"預處理後的字元數: {len(self.token)}")
        print(f"詞彙表大小: {len(self.vocab)}")
        
    def __read_file__(self):
        """
            讀取文本文件
        """
        with open(self.file_path, "r", encoding= "utf-8") as f:
            rawText = f.read()
        
        return rawText
    
    def __tokenize__(self, text):
        """
            Simple tokenizer function, 斷詞
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text) # 使用正則表達式分割文本
        
        token = [token for token in tokens if token.strip()] # 過濾掉空字符串
        
        return token

    def __token_to_id__(self, tokens):
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
    print(preprocessor.token[:50])
    # rawText, preprocessedText, vocab = preprocessor.get_vocab_table()
    # print(preprocessor.get_vocab_table())