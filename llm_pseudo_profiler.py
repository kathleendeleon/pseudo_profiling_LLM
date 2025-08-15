#!/usr/bin/env python3
"""
Tiny LLM pseudo-profiler (commented)
Inputs: context length, target tokens, batch size, layers/heads/dim, dtype.
    layers, d_model, n_heads, kv_heads, d_ff --> model architecture specs.
    prompt --> context tokens fed in.
    gen --> number of tokens to generate.
    batch --> number of sequences processed at once.
    dtype --> numeric precision (bf16, fp16, etc.).
    cluster_tflops --> total TFLOPs across your GPU cluster.
    utilization --> fraction of peak performance you realistically get.
    fudge --> overhead factor for non-GEMM ops.
Outputs: CSV row(s) with KV cache memory and a rough latency budget.

Each function below includes:
- What it computes
- Meaning/units of inputs and outputs
- Any modeling assumptions or simplifications
"""
import argparse, math, csv, sys, json


# ----------------------------- Core utilities -----------------------------

def bytes_per_dtype(dtype: str) -> int:
    """
    Map a numeric dtype to bytes-per-element.

    Args:
        dtype: 'bf16'|'fp16'|'fp32'|'fp8'|'int8' (case-insensitive)

    Returns:
        int: number of bytes per element for tensors stored in this dtype.
             e.g., bf16 -> 2 bytes, fp32 -> 4 bytes
    """
    d = dtype.lower()
    if d in ('bf16','bfloat16','fp16','float16'): return 2
    if d in ('fp8','e5m2','e4m3','int8'): return 1
    if d in ('fp32','float32'): return 4
    raise ValueError(f'Unknown dtype: {dtype}')


def kv_cache_bytes(seq_len: int, layers: int, d_model: int,
                   n_heads: int, kv_heads: int, dtype_bytes: int) -> int:
    """
    Estimate KV cache size (bytes) for ONE sequence at a given sequence length.

    Intuition:
      - For each decoder layer we store K and V for every token produced so far.
      - With GQA/MQA, we have fewer KV heads (kv_heads) than attention heads (n_heads).
      - Size per token per layer ≈ 2 * kv_heads * d_head * bytes, where d_head = d_model / n_heads.

    Args:
      seq_len: current sequence length (prompt + generated so far).
      layers: number of transformer layers (L).
      d_model: model hidden size.
      n_heads: attention heads for queries (Q).
      kv_heads: attention heads for K/V (GQA). If no GQA, kv_heads == n_heads.
      dtype_bytes: bytes per element used to store K/V (e.g., 2 for bf16).

    Returns:
      Total KV cache size in bytes for one sequence at seq_len.
    """
    d_head = d_model // n_heads                      # per-head channel dim
    return 2 * layers * seq_len * kv_heads * d_head * dtype_bytes  # 2 = K and V


def prefill_flops(n_prompt: int, layers: int, d_model: int, d_ff: int,
                  n_heads: int, kv_heads: int, fudge: float = 1.2) -> float:
    """
    Rough FLOPs for the PREFILL pass of ONE sequence.

    Model:
      MACs per layer ≈
        2 * n^2 * d_model                      # attention matmuls QK^T and Attn*V
        + n * d_model^2 * (2 + 2*kv_ratio)     # Q,K,V,O projections (GQA lowers K/V cost)
        + 2 * n * d_model * d_ff               # MLP (two big matmuls)
      FLOPs ≈ 2 * MACs; multiply by L layers; inflate by 'fudge' for non-GEMM ops.

    Args:
      n_prompt: tokens in input context.
      d_ff: MLP hidden size (often ~4*d_model).
      kv_heads/n_heads: used to compute kv_ratio for GQA effect.
      fudge: multiplicative overhead factor (kernel launches, norms, etc).

    Returns:
      float: estimated FLOPs for prefill of ONE sequence.
    """
    kv_ratio = kv_heads / n_heads
    macs = (2*(n_prompt**2)*d_model) \
         + (n_prompt*d_model*d_model*(2 + 2*kv_ratio)) \
         + (2*n_prompt*d_model*d_ff)
    return 2 * macs * layers * fudge


def decode_flops(n_prompt: int, T: int, layers: int, d_model: int, d_ff: int,
                 n_heads: int, kv_heads: int, fudge: float = 1.2) -> float:
    """
    Rough FLOPs for DECODING T new tokens for ONE sequence.

    Model:
      At step i (0..T-1) current length n_i = n_prompt + i.
      Sum over all steps:
        MACs per layer ≈
          2 * d_model * sum(n_i)                 # attention against entire cache
          + T * d_model^2 * (2 + 2*kv_ratio)     # Q,K,V,O per generated token
          + T * 2 * d_model * d_ff               # MLP per generated token
      FLOPs ≈ 2 * MACs; multiply by L; inflate by 'fudge'.

    Args:
      n_prompt: original prompt length.
      T: number of tokens to generate.

    Returns:
      float: estimated FLOPs for decoding T tokens of ONE sequence.
    """
    kv_ratio = kv_heads / n_heads
    sum_n = T*n_prompt + (T*(T-1))//2  # sum_{i=0..T-1} (n_prompt + i)
    macs = (2*d_model*sum_n) \
         + (T*d_model*d_model*(2 + 2*kv_ratio)) \
         + (T*2*d_model*d_ff)
    return 2 * macs * layers * fudge


def seconds_from_flops(total_flops: float, cluster_tflops: float, utilization: float) -> float:
    """
    Convert FLOPs to seconds using available cluster compute.

    Args:
      total_flops: workload FLOPs.
      cluster_tflops: aggregate available TFLOPs at target dtype (e.g., 8*H100 BF16).
      utilization: effective fraction of peak compute you actually realize (0..1).

    Returns:
      float: seconds to complete (prefill or decode) for the given workload.
    """
    eff = cluster_tflops * 1e12 * utilization       # effective FLOPs/s
    return total_flops / eff if eff > 0 else float('inf')


def human_bytes(n: float) -> str:
    """
    Pretty-print a byte count using binary units.
    """
    u = ['B','KiB','MiB','GiB','TiB']
    i = 0
    while n >= 1024 and i < len(u)-1:
        n /= 1024; i += 1
    return f"{n:,.2f} {u[i]}"


# ------------------------------ Scenario run ------------------------------

def run_scenario(s: dict) -> dict:
    """
    Compute key serving metrics for a single scenario.

    Inputs (s):
      layers, d_model, n_heads, kv_heads, d_ff       # model shape
      prompt (n_prompt), gen (T)                     # token counts
      batch                                          # number of sequences processed together
      dtype                                          # 'bf16' | 'fp16' | ...
      cluster_tflops                                 # hardware budget at dtype
      utilization                                    # effective fraction of peak (0..1)
      fudge                                          # overhead multiplier

    Intermediate variables:
      b           : bytes per element based on dtype (for KV memory)
      pf / df     : FLOPs for prefill/decode for the WHOLE batch
      prefill_s   : seconds to run prefill for this batch
      decode_s    : seconds to run decode for this batch
      total_s     : total latency (prefill + decode) for this batch
      peak_len    : prompt + generated (end-of-generation sequence length)
      kv_seq      : KV bytes for ONE sequence at peak_len
      kv_total    : total KV for the entire batch (×1.1 small overhead)

    Returns (row dict; all values are final outputs written to CSV):
      'Prefill_s', 'Decode_s', 'Total_s'             # seconds
      'Agg_tok_per_s'                                # total tokens/sec across batch (based on decode time)
      'Tok_per_s_per_seq'                            # tokens/sec per sequence (decode time)
      'KV_per_seq_peak', 'KV_total_batch'            # human-readable memory sizes
      plus identifying fields (model shape, batch, dtype, etc).
    """
    # 1) Bytes/elt for K/V cache at chosen dtype
    b = bytes_per_dtype(s['dtype'])

    # 2) FLOPs for the batch: multiply per-sequence FLOPs by batch size
    pf = prefill_flops(
        s['prompt'], s['layers'], s['d_model'], s['d_ff'],
        s['n_heads'], s['kv_heads'], s['fudge']
    ) * s['batch']

    df = decode_flops(
        s['prompt'], s['gen'], s['layers'], s['d_model'], s['d_ff'],
        s['n_heads'], s['kv_heads'], s['fudge']
    ) * s['batch']

    # 3) Convert FLOPs → time using effective cluster performance
    prefill_s = seconds_from_flops(pf, s['cluster_tflops'], s['utilization'])
    decode_s  = seconds_from_flops(df, s['cluster_tflops'], s['utilization'])
    total_s   = prefill_s + decode_s

    # 4) KV memory at peak sequence length (end of generation)
    peak_len  = s['prompt'] + s['gen']
    kv_seq    = kv_cache_bytes(peak_len, s['layers'], s['d_model'],
                               s['n_heads'], s['kv_heads'], b)
    kv_total  = kv_seq * s['batch'] * 1.1   # small paging/fragmentation overhead

    # 5) Throughput derived from decode time (prefill is one-shot setup)
    agg_tok_per_s = (s['batch']*s['gen'])/decode_s if decode_s > 0 else float('inf')
    tok_per_s_seq = (s['gen']/decode_s) if decode_s > 0 else float('inf')

    return {
        # Descriptive fields
        'Model': f"{s['layers']}L d={s['d_model']} h={s['n_heads']} (kv={s['kv_heads']}) d_ff={s['d_ff']}",
        'Prompt_to_Gen': f"{s['prompt']} → +{s['gen']}",
        'Batch': s['batch'],
        'Dtype': s['dtype'],
        'Cluster_TFLOPs': s['cluster_tflops'],
        'Utilization': s['utilization'],

        # Latency (seconds)
        'Prefill_s': round(prefill_s, 2),
        'Decode_s':  round(decode_s,  2),
        'Total_s':   round(total_s,   2),

        # Throughput (tokens/sec)
        'Agg_tok_per_s':     round(agg_tok_per_s, 1) if agg_tok_per_s != float('inf') else agg_tok_per_s,
        'Tok_per_s_per_seq': round(tok_per_s_seq, 2) if tok_per_s_seq != float('inf') else tok_per_s_seq,

        # Memory (human-readable strings)
        'KV_per_seq_peak': human_bytes(kv_seq),
        'KV_total_batch':  human_bytes(kv_total),
    }


# ----------------------------------- CLI -----------------------------------

def main():
    """
    Parse CLI flags and emit a CSV with one or more scenario rows.

    Outputs:
      - Prints a confirmation like "Wrote results.csv with N scenario(s)."
      - Writes 'results.csv' (or --out-csv path) with the columns returned by run_scenario().

    Usage (single scenario):
      python llm_pseudo_profiler.py \
        --layers 80 --d-model 8192 --n-heads 64 --kv-heads 8 --d-ff 32768 \
        --prompt 8000 --gen 512 --batch 8 --dtype bf16 \
        --cluster-tflops 7912 --utilization 0.45 --fudge 1.2 \
        --out-csv results.csv

    Usage (batch via JSON list):
      python llm_pseudo_profiler.py --scenarios-json scenarios.json --out-csv results.csv

      where scenarios.json is:
      [
        {
          "layers": 80, "d_model": 8192, "n_heads": 64, "kv_heads": 8, "d_ff": 32768,
          "prompt": 8000, "gen": 512, "batch": 8,
          "dtype": "bf16", "cluster_tflops": 7912, "utilization": 0.45, "fudge": 1.2
        },
        ...
      ]
    """
    ap = argparse.ArgumentParser(description='Tiny LLM pseudo-profiler')
    # ---- Model shape ----
    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--d-model', type=int, required=True)
    ap.add_argument('--n-heads', type=int, required=True)
    ap.add_argument('--kv-heads', type=int, default=None, help='defaults to n-heads (no GQA)')
    ap.add_argument('--d-ff', type=int, default=None, help='defaults to 4*d-model')

    # ---- Workload ----
    ap.add_argument('--prompt', type=int, required=True, help='context length (tokens)')
    ap.add_argument('--gen', type=int, required=True, help='target new tokens')
    ap.add_argument('--batch', type=int, required=True)

    # ---- Numeric type & hardware ----
    ap.add_argument('--dtype', type=str, default='bf16')
    ap.add_argument('--cluster-tflops', type=float, default=989.0,
                    help='total available TFLOPs at chosen dtype (sum across all GPUs)')
    ap.add_argument('--utilization', type=float, default=0.35,
                    help='effective fraction of theoretical peak (0..1)')

    # ---- Modeling fudge factor ----
    ap.add_argument('--fudge', type=float, default=1.2,
                    help='inflate FLOPs for non-GEMM ops, kernel overheads, etc.')

    # ---- Output / batch scenarios ----
    ap.add_argument('--out-csv', type=str, default='results.csv')
    ap.add_argument('--scenarios-json', type=str, help='optional JSON list of scenarios to batch-run')

    args = ap.parse_args()

    # Construct scenarios either from JSON or from CLI flags
    if args.scenarios_json:
        with open(args.scenarios_json,'r') as f:
            scenarios = json.load(f)  # list[dict] with same keys as below
    else:
        # Fill defaults for kv_heads and d_ff if not provided
        scenarios = [{
            'layers': args.layers,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'kv_heads': (args.kv_heads if args.kv_heads is not None else args.n_heads),
            'd_ff': (args.d_ff if args.d_ff is not None else 4*args.d_model),
            'prompt': args.prompt,
            'gen': args.gen,
            'batch': args.batch,
            'dtype': args.dtype,
            'cluster_tflops': args.cluster_tflops,
            'utilization': args.utilization,
            'fudge': args.fudge,
        }]

    # Compute metrics for each scenario and write CSV
    rows = [run_scenario(s) for s in scenarios]
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    print(f'Wrote {args.out_csv} with {len(rows)} scenario(s).')


if __name__ == '__main__':
    main()
