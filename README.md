# Big Picture Data Flow (Pseudo Profiling LLM)

[Client] <br>
   │  JSON RPC / gRPC <br>
   ▼ <br>
[API Gateway] ── auth / rate limit / quotas <br>
   │ <br>
   ▼ <br>
[Request Router] ──> picks an inference pool (model X, quant Y) <br>
   │ <br>
   ▼ <br>
[Tokenizer] (BPE) ──> int token IDs <br>
   │ <br>
   ▼ <br>
[Scheduler/Batcher] <br>
   │   merges compatible requests → “micro-batches” <br>
   ▼ <br>
[Model Server] <br>
   ├─ Prefill phase (build KV cache for prompt) <br>
   ├─ Decode loop (one or few tokens/step) <br>
   │     ├─ optional speculative decode w/ draft model <br>
   │     └─ sampling (top-p/top-k/temperature, penalties) <br>
   └─ Streaming partials back to client <br>
   │ <br>
   ▼ <br>
[Post-processing] <br>
   ├─ detokenize to UTF-8 text <br>
   ├─ safety/harm filters, formatters, tool call validators <br>
   └─ usage metering, logs (PII-minimized) <br>

     

Attention with rotary positional embeddings (RoPE) <br>
ASCII inside the block: <br>
    h_{in} ──┬─────────────► LN ─► [Q,K,V]=hW  ─► RoPE(Q,K) ─► Masked Attn ─► h + Δ1 <br>
             │ <br>
             └─────────────► LN ─► W1 ─► SwiGLU ─► W2 ─────────► (h + Δ1) + Δ2 = h_{out} <br>

