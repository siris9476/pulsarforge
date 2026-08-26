#!/usr/bin/env python3
"""Concurrent throughput bench (M8, continuous batching): N parallel
clients with different prompts against a listening OpenAI-compatible
server (nf serve OR llama-server, same script for both — that's the
point), measures aggregate tok/s and per-request latency. This is NOT
an automatic gate (no pass/fail exit code): it's a measurement tool,
the verdict is written by whoever reads the numbers.

Usage: python tools/bench_concurrent_serve.py <host> <port> <n_conc> <max_tokens>

Recorded baseline (DeepSeek-V2-Lite-Chat.Q4_K_M, 8 clients, 100
tokens, greedy): llama-server (-np 8 -cb) 13.71 tok/s aggregate; nf
serve (Phase 2, compute still serial) 8.14 tok/s (59.4%). Phase 3b
target: close a significant part of the gap.
"""
import http.client
import json
import sys
import threading
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18093
N_CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAX_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 100

PROMPTS = [
    f"Write a short paragraph (a few sentences) about the number {i} "
    f"and its mathematical history." for i in range(N_CONC)
]


def send_request(prompt, results, idx):
    t0 = time.time()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                        "stream": False, "temperature": 0,
                        "max_tokens": MAX_TOK})
    try:
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        n_tok = data.get("usage", {}).get("completion_tokens", 0)
        results[idx] = (n_tok, time.time() - t0)
    except Exception as e:
        results[idx] = (0, time.time() - t0)
        print(f"  request {idx} error: {e}")
    finally:
        conn.close()


def main():
    results = [None] * N_CONC
    threads = [threading.Thread(target=send_request, args=(p, results, i))
               for i, p in enumerate(PROMPTS)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    total_tok = sum(r[0] for r in results)
    latencies = [r[1] for r in results]
    print(f"host={HOST}:{PORT}  N={N_CONC}  max_tokens={MAX_TOK}")
    print(f"  total tokens generated: {total_tok}")
    print(f"  wall-clock: {wall:.2f}s")
    print(f"  AGGREGATE throughput: {total_tok / wall:.2f} tok/s")
    print(f"  per-request latency: min={min(latencies):.2f}s "
          f"max={max(latencies):.2f}s mean={sum(latencies)/len(latencies):.2f}s")


if __name__ == "__main__":
    main()
