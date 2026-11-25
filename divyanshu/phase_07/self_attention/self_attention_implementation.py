import math
import numpy as np


"""Softmax implementation using pure python"""

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    sum_exps = sum(exps)
    
    probabilities = [(e / sum_exps) for e in exps] 
    return probabilities

# scores = [2.0, 1.0, 0.1]

# print(f'raw logits: {scores}')
# probs = softmax(scores)
# print(f'softmax probabilities: {[round(p, 4) for p in probs]}')


"""softmax implementation using numpy"""
import numpy as np

def softmax_fn(x):
    shifted = x - np.max(x, axis=-1,keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
# print(f'softmax probabilities: {softmax_fn(scores)}')

"""scaled dot product attention"""
# sentence = ["black", "cat"] -> input embeddings shape (2,4)
X = np.array([
    [0.2, 0.8, 0.1, 0.5],  # "Black"
    [0.9, 0.1, 0.4, 0.2]   # "cat"
])
# learned weight matrices (initialized randomly, updated during training)-> shape(4,4)
np.random.seed(42)
w_q = np.random.randn(4,4)
w_k = np.random.randn(4,4)
w_v = np.random.randn(4,4)

# project X into Q, K, V -> each has shape of (2,4)
Q = X @ w_q
K = X @ w_k
V = X @ w_v

# print(f'Query matrix: \n{Q}')
# print(f'Key matrix:\n {K}')
# print(f'Value matrix:\n {V}')

def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    # print("scores in scaled dot product attention function: ", scores)
    weights = softmax_fn(scores)
    output = weights @ V
    return output, weights


class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk
    
    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights


sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat']
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model)) # 6 x 8

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)


print('Attention weights (each row: where that token looks):\n')
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()


def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f'\n{'':>6}', end="")
    for t in tokens:
        print(f'{t:>6}', end="")
    print()
    
    for i in range(n):
        print(f'{tokens[i]:>6}', end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '  '}", end="")
        print()

ascii_heatmap(weights, sentence)
        
    