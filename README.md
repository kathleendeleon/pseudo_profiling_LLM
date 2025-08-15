# Pseudo Profiling LLM

'''
# Big Picture Data Flow (Pseudo Profiling LLM)

[Client] <br>
   │  JSON RPC / gRPC
   ▼
[API Gateway] ── auth / rate limit / quotas
   │
   ▼
[Request Router] ──> picks an inference pool (model X, quant Y)
   │
   ▼
[Tokenizer] (BPE) ──> int token IDs
   │
   ▼
[Scheduler/Batcher]
   │   merges compatible requests → “micro-batches”
   ▼
[Model Server]
   ├─ Prefill phase (build KV cache for prompt)
   ├─ Decode loop (one or few tokens/step)
   │     ├─ optional speculative decode w/ draft model
   │     └─ sampling (top-p/top-k/temperature, penalties)
   └─ Streaming partials back to client
   │
   ▼
[Post-processing]
   ├─ detokenize to UTF-8 text
   ├─ safety/harm filters, formatters, tool call validators
   └─ usage metering, logs (PII-minimized)

     

Attention with rotary positional embeddings (RoPE)
ASCII inside the block:
    h_{in} ──┬─────────────► LN ─► [Q,K,V]=hW  ─► RoPE(Q,K) ─► Masked Attn ─► h + Δ1
             │
             └─────────────► LN ─► W1 ─► SwiGLU ─► W2 ─────────► (h + Δ1) + Δ2 = h_{out}
'''
