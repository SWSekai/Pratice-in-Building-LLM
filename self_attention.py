import torch

if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]]
    )
    
    x = inputs[1]
    in_dimension = inputs.shape[1] # 輸入的特徵維度
    out_dimension = 2 # 輸出的特徵維度
    
    torch.manual_seed(123)
    query = torch.nn.Parameter(torch.rand(in_dimension, out_dimension), requires_grad= False) # 查詢權重矩陣, requires_grad= False 表示不需要梯度
    key = torch.nn.Parameter(torch.rand(in_dimension, out_dimension), requires_grad= False)   # 鍵權重矩陣
    value = torch.nn.Parameter(torch.rand(in_dimension, out_dimension), requires_grad= False) # 值權重矩陣
    
    query_x = x @ query  # 查詢向量
    key_x = x @ key      # 鍵向量
    value_x = x @ value  # 值向量
    
    keys = inputs @ key      # 所有輸入的鍵向量
    values = inputs @ value  # 所有輸入的值向量
    
    keys_2 = keys[1]
    attn_score_x_2 = query_x @ keys.T  # 計算注意力分數
    
    attn_weights_x = torch.softmax(attn_score_x_2 / keys.shape[-1]**0.5, dim=-1)  # 計算注意力權重
    
    context_vec_x = attn_weights_x @ values  # 計算上下文向量
    
    print(context_vec_x)