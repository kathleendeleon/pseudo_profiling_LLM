#!/usr/bin/env python3
"""Tiny LLM pseudo-profiler
Inputs: context length, target tokens, batch size, layers/heads/dim, dtype.
Outputs: KV cache memory and a rough latency budget.
"""
import argparse, math, csv, sys, json

def bytes_per_dtype(dtype: str) -> int:
    d = dtype.lower()
    if d in ('bf16','bfloat16','fp16','float16'): return 2
    if d in ('fp8','e5m2','e4m3','int8'): return 1
    if d in ('fp32','float32'): return 4
    raise ValueError(f'Unknown dtype: {dtype}')

def kv_cache_bytes(seq_len, layers, d_model, n_heads, kv_heads, dtype_bytes):
    d_head = d_model // n_heads
    return 2 * layers * seq_len * kv_heads * d_head * dtype_bytes

def prefill_flops(n_prompt, layers, d_model, d_ff, n_heads, kv_heads, fudge=1.2):
    kv_ratio = kv_heads / n_heads
    macs = (2*(n_prompt**2)*d_model) + (n_prompt*d_model*d_model*(2+2*kv_ratio)) + (2*n_prompt*d_model*d_ff)
    return 2 * macs * layers * fudge

def decode_flops(n_prompt, T, layers, d_model, d_ff, n_heads, kv_heads, fudge=1.2):
    kv_ratio = kv_heads / n_heads
    sum_n = T*n_prompt + (T*(T-1))//2
    macs = (2*d_model*sum_n) + (T*d_model*d_model*(2+2*kv_ratio)) + (T*2*d_model*d_ff)
    return 2 * macs * layers * fudge

def seconds_from_flops(total_flops, cluster_tflops, utilization):
    eff = cluster_tflops * 1e12 * utilization
    return total_flops/eff if eff>0 else float('inf')

def human_bytes(n):
    u = ['B','KiB','MiB','GiB','TiB']
    i = 0
    while n >= 1024 and i < len(u)-1:
        n /= 1024; i += 1
    return f"{n:,.2f} {u[i]}"

def run_scenario(s):
    b = bytes_per_dtype(s['dtype'])
    pf = prefill_flops(s['prompt'], s['layers'], s['d_model'], s['d_ff'], s['n_heads'], s['kv_heads'], s['fudge']) * s['batch']
    df = decode_flops(s['prompt'], s['gen'], s['layers'], s['d_model'], s['d_ff'], s['n_heads'], s['kv_heads'], s['fudge']) * s['batch']
    prefill_s = seconds_from_flops(pf, s['cluster_tflops'], s['utilization'])
    decode_s  = seconds_from_flops(df, s['cluster_tflops'], s['utilization'])
    total_s   = prefill_s + decode_s
    peak_len  = s['prompt'] + s['gen']
    kv_seq    = kv_cache_bytes(peak_len, s['layers'], s['d_model'], s['n_heads'], s['kv_heads'], b)
    kv_total  = kv_seq * s['batch'] * 1.1
    return {
        'Model': f"{s['layers']}L d={s['d_model']} h={s['n_heads']} (kv={s['kv_heads']}) d_ff={s['d_ff']}",
        'Prompt_to_Gen': f"{s['prompt']} → +{s['gen']}",
        'Batch': s['batch'],
        'Dtype': s['dtype'],
        'Cluster_TFLOPs': s['cluster_tflops'],
        'Utilization': s['utilization'],
        'Prefill_s': round(prefill_s,2),
        'Decode_s': round(decode_s,2),
        'Total_s': round(total_s,2),
        'Agg_tok_per_s': round((s['batch']*s['gen'])/decode_s,1) if decode_s>0 else float('inf'),
        'Tok_per_s_per_seq': round(s['gen']/decode_s,2) if decode_s>0 else float('inf'),
        'KV_per_seq_peak': human_bytes(kv_seq),
        'KV_total_batch': human_bytes(kv_total),
    }

def main():
    ap = argparse.ArgumentParser(description='Tiny LLM pseudo-profiler')
    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--d-model', type=int, required=True)
    ap.add_argument('--n-heads', type=int, required=True)
    ap.add_argument('--kv-heads', type=int, default=None, help='defaults to n-heads (no GQA)')
    ap.add_argument('--d-ff', type=int, default=None, help='defaults to 4*d-model')
    ap.add_argument('--prompt', type=int, required=True, help='context length (tokens)')
    ap.add_argument('--gen', type=int, required=True, help='target new tokens')
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--dtype', type=str, default='bf16')
    ap.add_argument('--cluster-tflops', type=float, default=989.0, help='total available TFLOPs at chosen dtype')
    ap.add_argument('--utilization', type=float, default=0.35, help='effective fraction of peak (0-1)')
    ap.add_argument('--fudge', type=float, default=1.2, help='inflate FLOPs for non-GEMM ops')
    ap.add_argument('--out-csv', type=str, default='results.csv')
    ap.add_argument('--scenarios-json', type=str, help='optional JSON list of scenarios to batch-run')
    args = ap.parse_args()

    scenarios = []
    if args.scenarios_json:
        import json
        with open(args.scenarios_json,'r') as f:
            scenarios = json.load(f)
    else:
        scenarios = [{
            'layers': args.layers, 'd_model': args.d_model, 'n_heads': args.n_heads,
            'kv_heads': (args.kv_heads if args.kv_heads is not None else args.n_heads),
            'd_ff': (args.d_ff if args.d_ff is not None else 4*args.d_model),
            'prompt': args.prompt, 'gen': args.gen, 'batch': args.batch,
            'dtype': args.dtype, 'cluster_tflops': args.cluster_tflops,
            'utilization': args.utilization, 'fudge': args.fudge,
        }]

    rows = [run_scenario(s) for s in scenarios]
    with open(args.out_csv, 'w', newline='') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {args.out_csv} with {len(rows)} scenario(s).')

if __name__ == '__main__':
    main()
