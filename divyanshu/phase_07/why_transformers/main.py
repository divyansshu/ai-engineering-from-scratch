# """Why Transformers - demonstrate the serial-depth gap between RNN-style
# recurrence and attention-style parallel reduction.

# Runs in pure stdlib. No numpy, no torch.
# """

import math
import time
import torch


def rnn_style(xs, decay=0.9):
    """Sequential recurrence: h_t depends on h_{t-1}. Cannot parallelize."""
    h = [0.0] * 64
    for x in xs:
        for k in range(64):
            h[k] = decay * h[k] + x
    return h


def attention_style(xs):
    """Order-independent reduction: every element is independent."""
    return sum(xs) / len(xs)


def serial_scan(xs):
    """Prefix sum computed serially. Depth O(N)."""
    out = []
    acc = 0.0
    for x in xs:
        acc += x
        out.append(acc)
    return out


def parallel_scan(xs):
    """Hillis-Steele parallel prefix sum. Depth O(log N).

    In pure Python each step is still serial, but the data-dependency
    graph has depth log2(N). On real hardware with N-wide SIMD this
    gets you a log-depth scan; on a CPU it's the same wall-clock but
    the graph shape is what matters for GPU kernels.
    """
    out = list(xs)
    step = 1
    n = len(out)
    while step < n:
        new = list(out)
        for i in range(step, n):
            new[i] = out[i] + out[i - step]
        out = new
        step *= 2
    return out


def benchmark(n, reps=3):
    xs = [0.001 * (i % 17) for i in range(n)]

    best_rnn = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = rnn_style(xs)
        best_rnn = min(best_rnn, time.perf_counter() - t0)

    best_attn = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = attention_style(xs)
        best_attn = min(best_attn, time.perf_counter() - t0)

    return best_rnn, best_attn


def depth(n):
    """Serial-depth count for RNN vs attention-style reductions."""
    rnn_depth = n
    attn_depth = max(1, math.ceil(math.log2(n)))
    return rnn_depth, attn_depth


def main():
    print("=== serial-depth comparison ===")
    print(f"{'N':>8}  {'rnn depth':>12}  {'attn depth':>12}  {'speedup (ops)':>16}")
    for n in [64, 512, 4096, 32768, 262144]:
        rd, ad = depth(n)
        print(f"{n:>8}  {rd:>12}  {ad:>12}  {rd / ad:>15.0f}x")

    print()
    print("=== wall-clock on this machine (pure Python) ===")
    print(f"{'N':>8}  {'rnn (ms)':>10}  {'attn (ms)':>10}  {'ratio':>8}")
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        rnn_t, attn_t = benchmark(n)
        ratio = rnn_t / attn_t if attn_t > 0 else float("inf")
        print(f"{n:>8}  {rnn_t * 1000:>10.2f}  {attn_t * 1000:>10.2f}  {ratio:>7.1f}x")

    print()
    print("=== prefix-sum equivalence check ===")
    xs = [float(i) for i in range(16)]
    print(xs, end='\n')
    ser = serial_scan(xs)
    par = parallel_scan(xs)
    mismatches = sum(1 for a, b in zip(ser, par) if abs(a - b) > 1e-9)
    print(f"length: {len(xs)}, mismatches between serial and parallel scan: {mismatches}")
    print(f"last value (serial):   {ser[-1]}")
    print(f"last value (parallel): {par[-1]}")

    print()
    print("takeaway: attention wins on every dimension but memory.")
    print("memory cost is O(N^2) for full attention; Lesson 12 covers the fixes.")


if __name__ == "__main__":
    main()
    
"""Exercises"""""

"""Exercise 1: """    

N = 10_000
d = 64
xs = [0.001 * (i % 17) for i in range(N)]

# 1. Pure Python: Scalar (d=1) vs Vector (d=64)
t0 = time.perf_counter()
h1 = 0.0
for x in xs:
    h1 = 0.9 * h1 + x
t_py_1 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
h64 = [0.0] * d
for x in xs:
    for k in range(d):
        h64[k] = 0.9 * h64[k] + x
t_py_64 = (time.perf_counter() - t0) * 1000

# 2. Vectorized (PyTorch): Scalar (d=1) vs Vector (d=64)
x_t = torch.tensor(xs)
t0 = time.perf_counter()
ht_1 = torch.zeros(1)
for i in range(N):
    ht_1 = 0.9 * ht_1 + x_t[i]
t_vec_1 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
ht_64 = torch.zeros(d)
for i in range(N):
    ht_64 = 0.9 * ht_64 + x_t[i]
t_vec_64 = (time.perf_counter() - t0) * 1000

print(f"Pure Python (d=1):  {t_py_1:6.2f} ms")
print(f"Pure Python (d=64): {t_py_64:6.2f} ms  --> {t_py_64/t_py_1:.1f}x slower (CPU does 64 steps serially)")
print("-" * 65)
print(f"PyTorch Vec (d=1):  {t_vec_1:6.2f} ms")
print(f"PyTorch Vec (d=64): {t_vec_64:6.2f} ms  --> {t_vec_64/t_vec_1:.1f}x (64 dims run in parallel, but N-loop still bottlenecks!)")


"""Exercise 2:"""
def serial_scan(xs):
    out = []
    acc = 0.0
    for x in xs:
        acc += x
        out.append(acc)
    return out

def parallel_scan(xs):
    n = len(xs)
    out = list(xs)
    step = 1
    depth = 0
    while step < n:
        # new = list(out)
        for i in range(step, n):
            out[i] = out[i] + out[i-step]
        # out = new
        step *= 2
        depth += 1
    return out, depth
        

xs = [float(i % 13) * 0.1 for i in range(1024)]
xs = [1,2,3,4]

serial_out = serial_scan(xs)
parallel_out, depth_count = parallel_scan(xs)

print(f'sequence length: {len(xs)}')
print(f'serial scan output: {serial_out}')
print(f'serial depth: {len(xs)} steps')
print(f'parallel scan depth: {depth_count} steps (2^{depth_count} = {2**depth_count})')
print(f'parallel scan output: {parallel_out}')