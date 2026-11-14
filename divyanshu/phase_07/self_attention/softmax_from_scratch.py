import math

"""Softmax implementation using pure python"""

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    sum_exps = sum(exps)
    
    probabilities = [(e / sum_exps) for e in exps] 
    return probabilities

scores = [2.0, 1.0, 0.1]

print(f'raw logits: {scores}')
probs = softmax(scores)
print(f'softmax probabilities: {[round(p, 4) for p in probs]}')


"""softmax implementation using numpy"""
import numpy as np

def softmax_fn(x):
    shifted = x - np.max(x, axis=-1,keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    

print(f'softmax probabilities: {softmax_fn(scores)}')

