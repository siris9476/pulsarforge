#!/usr/bin/env python3
"""M8 Phase 1 regression test (concurrent server): the thread-per-
connection for I/O (accept/read/tokenization) must not corrupt or
alter responses compared to today. Checks: N requests with DISTINCT
prompts (independent sessions, each in its own slot), sent first ONE
AT A TIME (baseline) and then all CONCURRENTLY against a fresh
server — with temperature 0 (greedy) and no numeric interaction
between sessions at this stage, each prompt's response must be
EXACTLY the same in both cases. Corruption of the shared reqbuf
(the bug fixed in Phase 1) or a race on slot selection would show up
here as a garbled or missing response.

Also includes a "double" round (two concurrent passes back to back on
the SAME server, reused prompts) — the scenario that triggered a real
bug in `nf_session_copy_prefix` (Qwen K/V layout used on a
deepseek2/MLA model, silent heap corruption, the compute thread would
hang with no crash or log): the prefix transplant had never been
exercised on deepseek2 before. Short timeouts on the second round: if
it regresses, the test fails in a few seconds instead of hanging the
suite.

TWO REGIMES, TWO DIFFERENT GATES (mixed ubatch):
- The byte-exact gates above run with NF_MIXBATCH_OFF=1, i.e. on the
  DETERMINISTIC path (blocking prefill): there, a request must always
  return the exact same bytes, regardless of load. None of that value
  is given up.
- The last gate runs on the DEFAULT path (mixed ubatch: the prefill
  chunk travels in the same batch as other sessions' decode steps,
  +30% throughput, -25% latency). There, floating-point summation
  order depends on who else is in the batch, so tokens CAN change
  under load — that's a property of the design, not a bug (llama has
  the same behavior). Demanding byte-for-byte identity there would be
  demanding something the design cannot give: instead we check that a
  response arrives, makes sense, and there's no hang or socket
  mix-up. Numeric QUALITY is guarded by PPL (25.0974 vs 25.0572 for
  the baseline, within the already-accepted delta family:
  `NF_PPL_ROWS=1 nf perplexity ...`), not by this test.

Usage: python tests/check_serve_concurrency.py [gguf_path]
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q4_K_M.gguf"
NF = ".\\nf.exe" if sys.platform == "win32" else "./nf"
HOST = "127.0.0.1"
PORT = 18099
N_CONC = 6
ROUNDS = 3
# M8 Phase 4: NF_SRV_SLOTS is 8 (nf.c) - 10 concurrent requests force
# slot eviction/reuse (jobs queue until one frees up), a path never
# exercised by the test above (N_CONC=6 < 8 slots).
N_OVERFLOW = 10

PROMPTS = [f"Rispondi solo col numero {i}, e nient'altro." for i in range(N_CONC)]
OVERFLOW_PROMPTS = [f"Rispondi solo col numero {i}, e nient'altro."
                    for i in range(N_OVERFLOW)]


def wait_ready(port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = socket.create_connection((HOST, port), timeout=0.5)
            s.close()
            return True
        except OSError:
            time.sleep(0.2)
    return False


def start_server(port, mixbatch=False, max_tokens=12, no_think=False):
    """mixbatch=False -> NF_MIXBATCH_OFF=1: blocking prefill,
    DETERMINISTIC path (a request always returns the same bytes,
    regardless of load) — that's where the byte-exact gate below
    lives.
    mixbatch=True -> DEFAULT path (mixed ubatch): the prefill travels
    in the same batch as other sessions' decode steps, so
    floating-point summation order depends on batch composition and
    tokens can change under different load (same property as llama;
    PPL validated at 25.0974 vs 25.0572, within the already-accepted
    delta family). On that path we check MEANING, not bytes."""
    env = dict(os.environ)
    if not mixbatch:
        env["NF_MIXBATCH_OFF"] = "1"
    else:
        env.pop("NF_MIXBATCH_OFF", None)
    cmd = [NF, "serve", GGUF, "--port", str(port),
           "--max-tokens", str(max_tokens), "--temp", "0"]
    if no_think:
        cmd.append("--no-think")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    if not wait_ready(port):
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError("server not ready within timeout")
    return proc


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def send_request(port, prompt, timeout=60, max_tokens=12):
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                        "stream": False, "temperature": 0,
                        "max_tokens": max_tokens})
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    obj = json.loads(data)
    return obj["choices"][0]["message"]["content"]


def main() -> int:
    ok = True

    # baseline: sequential, a fresh server (reference behavior)
    proc = start_server(PORT)
    baseline = {}
    try:
        for p in PROMPTS:
            baseline[p] = send_request(PORT, p)
    finally:
        stop_server(proc)

    n_valid = sum(1 for v in baseline.values() if v)
    c0 = n_valid == len(PROMPTS)
    print(f"[{'OK' if c0 else 'FAIL'}] sequential baseline: "
          f"{n_valid}/{len(PROMPTS)} valid responses")
    ok &= c0

    # concurrent, fresh server each round, repeated to catch rare races
    for r in range(ROUNDS):
        proc = start_server(PORT)
        results = {}
        lock = threading.Lock()

        def worker(p):
            try:
                out = send_request(PORT, p)
            except Exception:
                out = None
            with lock:
                results[p] = out

        threads = [threading.Thread(target=worker, args=(p,)) for p in PROMPTS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop_server(proc)

        n_ok = sum(1 for p in PROMPTS if results.get(p) == baseline[p])
        match = n_ok == len(PROMPTS)
        print(f"[{'OK' if match else 'FAIL'}] round {r + 1}/{ROUNDS} "
              f"concurrent: {n_ok}/{len(PROMPTS)} responses identical "
              f"to baseline")
        ok &= match
        if not match:
            for p in PROMPTS:
                if results.get(p) != baseline[p]:
                    print(f"       | prompt={p!r}")
                    print(f"       | expected ={baseline[p]!r}")
                    print(f"       | got      ={results.get(p)!r}")

    # M8 Phase 4: N_OVERFLOW > 8 slots - some requests must wait in
    # queue for a slot to free up (eviction/reuse). Fresh sequential
    # baseline, then all concurrent against a fresh server - same
    # criterion (identical response), but now actually exercises the
    # queue.
    proc = start_server(PORT)
    overflow_baseline = {}
    try:
        for p in OVERFLOW_PROMPTS:
            overflow_baseline[p] = send_request(PORT, p)
    finally:
        stop_server(proc)
    n_valid = sum(1 for v in overflow_baseline.values() if v)
    c_ov0 = n_valid == len(OVERFLOW_PROMPTS)
    print(f"[{'OK' if c_ov0 else 'FAIL'}] overflow, sequential baseline: "
          f"{n_valid}/{len(OVERFLOW_PROMPTS)} valid responses")
    ok &= c_ov0

    proc = start_server(PORT)
    results = {}
    lock = threading.Lock()

    def worker_ov(p):
        try:
            out = send_request(PORT, p)
        except Exception:
            out = None
        with lock:
            results[p] = out

    threads = [threading.Thread(target=worker_ov, args=(p,))
               for p in OVERFLOW_PROMPTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_server(proc)

    n_ok = sum(1 for p in OVERFLOW_PROMPTS
               if results.get(p) == overflow_baseline[p])
    match_ov = n_ok == len(OVERFLOW_PROMPTS)
    print(f"[{'OK' if match_ov else 'FAIL'}] concurrent overflow "
          f"({N_OVERFLOW} requests, 8 slots): {n_ok}/{len(OVERFLOW_PROMPTS)} "
          f"responses identical to baseline")
    ok &= match_ov
    if not match_ov:
        for p in OVERFLOW_PROMPTS:
            if results.get(p) != overflow_baseline[p]:
                print(f"       | prompt={p!r}")
                print(f"       | expected ={overflow_baseline[p]!r}")
                print(f"       | got      ={results.get(p)!r}")

    # M8 (real bug found): a SECOND concurrent round on the SAME
    # server, right after the first, with prompts that share a long
    # prefix with the previous round - the scenario that triggered
    # nf_session_copy_prefix with the Qwen offset on a deepseek2 (MLA)
    # layout, corrupting the heap and hanging the compute thread (no
    # crash, no log - just a silent hang). Short timeout: if it
    # regresses, the test FAILS in a few seconds instead of hanging
    # the suite forever.
    proc = start_server(PORT)
    try:
        round1 = {}
        for p in PROMPTS:
            round1[p] = send_request(PORT, p)
        n_valid = sum(1 for v in round1.values() if v)
        c_r1 = n_valid == len(PROMPTS)
        print(f"[{'OK' if c_r1 else 'FAIL'}] double-round, first pass: "
              f"{n_valid}/{len(PROMPTS)} valid responses")
        ok &= c_r1

        results2 = {}
        lock2 = threading.Lock()

        def worker_r2(p):
            try:
                out = send_request(PORT, p, timeout=15)
            except Exception as e:
                out = f"ERROR: {e}"
            with lock2:
                results2[p] = out

        threads2 = [threading.Thread(target=worker_r2, args=(p,))
                    for p in PROMPTS]
        for t in threads2:
            t.start()
        for t in threads2:
            t.join(timeout=20)
        n_ok2 = sum(1 for p in PROMPTS if results2.get(p) == round1[p])
        match_r2 = n_ok2 == len(PROMPTS)
        print(f"[{'OK' if match_r2 else 'FAIL'}] double-round, second pass "
              f"(same server, reused prompts): {n_ok2}/{len(PROMPTS)} "
              f"responses identical to the first pass")
        ok &= match_r2
        if not match_r2:
            for p in PROMPTS:
                if results2.get(p) != round1[p]:
                    print(f"       | prompt={p!r}")
                    print(f"       | expected ={round1[p]!r}")
                    print(f"       | got      ={results2.get(p)!r}")
    finally:
        stop_server(proc)

    # MIXED ubatch, the DEFAULT path: the prefill chunk of a new
    # request travels in the SAME batch as other sessions' decode
    # steps (+30% throughput, -25% latency). Unavoidable consequence
    # — and the same one llama has — floating-point summation order
    # depends on WHO ELSE is in the batch, so tokens can change under
    # load. Here we therefore CANNOT demand byte-for-byte identity
    # (the gates above demand it, and get it, on the deterministic
    # path). Instead we check what must hold true regardless: every
    # request gets a response, the response MAKES SENSE (contains the
    # requested number), no session hangs, and no response ends up on
    # the wrong socket. Numeric quality is guarded by PPL (25.0974 vs
    # 25.0572, within the delta family already accepted by the
    # project), not by this test.
    # Higher max_tokens HERE (24 instead of 12): this model repeats
    # the prompt before answering, and with 12 tokens the digit
    # sometimes lands just past the budget. On the deterministic path
    # it barely makes it, on the mixed path the tokens are slightly
    # different and sometimes it doesn't — an artifact of the test
    # budget, not of batching (the truncated output is consistent, no
    # corruption or socket mix-up). We give the response room instead
    # of lowering the bar.
    # --no-think: the suite runs by default on Qwen3, which is a
    # REASONING model (it answers "<think>\nOkay, the user asked me
    # to..." and the digit doesn't appear in the first tokens). The
    # semantic criterion measures CONTENT, not the reasoning preamble:
    # without this, the test would fail a perfectly correct response.
    # (M8 Phase 4: mixed batching is now also active on Qwen —
    # qwen_eval_batched_rows — not just deepseek2; the semantic gate
    # applies to every model in the suite precisely because none of
    # them is a special case anymore.)
    MIX_TOKENS = 24
    proc = start_server(PORT, mixbatch=True, max_tokens=MIX_TOKENS,
                        no_think=True)
    results = {}
    lock = threading.Lock()

    def worker_mix(p):
        try:
            # the budget must ALSO be passed in the JSON body: that's
            # what wins over the server's --max-tokens
            out = send_request(PORT, p, timeout=40, max_tokens=MIX_TOKENS)
        except Exception as e:
            out = f"ERROR: {e}"
        with lock:
            results[p] = out

    threads = [threading.Thread(target=worker_mix, args=(p,)) for p in PROMPTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    stop_server(proc)

    # Real bug found (extended to Qwen): the scheduler's emission
    # loop could echo the TAIL OF THE PROMPT (including chat-template
    # markers) as if it were generated output, when a job's last chunk
    # spanned more than one line (almost always) — visible at a glance
    # on Qwen (<|im_start|>...), masked on deepseek2 by its plain-text
    # template ("User: ...Assistant:", it looked like a plausible
    # dialogue). The "contains the digit" check alone NEVER caught it
    # (the echo still includes the original prompt, which contains the
    # digit) — added an explicit check that the response doesn't
    # contain chat-template markers: a correct response never contains
    # them, an echo of the prompt (or its tail) almost always does.
    TEMPLATE_LEAK = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    n_ok = 0
    for i, p in enumerate(PROMPTS):
        out = results.get(p)
        # the prompt asks "answer with just the number i": the response
        # must contain THAT number, not be an error/hang, and not
        # contain an echo of the chat template (a sign of corruption).
        if (out and not out.startswith("ERROR") and str(i) in out
                and not any(m in out for m in TEMPLATE_LEAK)):
            n_ok += 1
    match_mix = n_ok == len(PROMPTS)
    print(f"[{'OK' if match_mix else 'FAIL'}] mixed ubatch (default), "
          f"SEMANTIC check: {n_ok}/{len(PROMPTS)} sensible responses "
          f"(byte-exactness not guaranteed by design — see docstring)")
    ok &= match_mix
    if not match_mix:
        for i, p in enumerate(PROMPTS):
            out = results.get(p)
            sane = (out and not out.startswith("ERROR") and str(i) in out
                    and not any(m in out for m in TEMPLATE_LEAK))
            if not sane:
                print(f"       | prompt={p!r}")
                print(f"       | got={out!r}")

    print(f"\n{'CONCURRENCY VALIDATED' if ok else 'CONCURRENCY FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
