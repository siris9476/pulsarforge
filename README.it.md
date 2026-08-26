# pulsarforge

> Versione italiana operativa. La presentazione pubblica del progetto,
> coi numeri finali e i confronti (colibrì, DwarfStar, llama.cpp), è
> in inglese: [README.md](README.md). Una traduzione inglese aggiornata
> di QUESTO documento (con i riferimenti stale sistemati) è
> [OPERATIONS.md](OPERATIONS.md).

Motore di inferenza LLM **didattico**, scritto in C, CPU-only, per un solo
modello alla volta (attualmente **Qwen3**). Ispirato alle lezioni di
[DwarfStar](https://github.com/antirez) e llama.cpp, ma con un obiettivo
diverso: **capire costruendo**, non competere.

## Cosa è (e cosa non è)

- **È** un motore piccolo e leggibile che carica un GGUF di Qwen3, lo
  tokenizza, esegue il forward pass e genera testo — validato a ogni
  passo contro l'implementazione ufficiale HuggingFace.
- **Non è** un runner GGUF generico: un modello solo, validato a fondo
  (la "scommessa stretta" di DwarfStar, in miniatura).
- **Non punta** a battere llama.cpp in velocità: su questa classe di
  hardware i suoi kernel sono anni di tuning SIMD. Punta a essere
  spiegabile riga per riga.

## Hardware di riferimento

Laptop x86-64 senza GPU dedicata: HP EliteBook 840 G5 (i7-8550U, 4 core /
8 thread, AVX2, 32GB RAM). Sviluppo e validazione su **Qwen3-0.6B**
(veloce da iterare); lo stesso codice deve far girare **Qwen3-8B
quantizzato** entro i 32GB.

## Setup (modelli e ambiente Python)

`models/` e `venv-tools/` sono volutamente fuori da git (8.9GB di GGUF,
un intero venv) — si ricreano così, prima di build/test:

```
# venv Python per i test (torch CPU — il torch globale su questa
# macchina crasha all'import):
python -m venv venv-tools
venv-tools/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
venv-tools/Scripts/pip install transformers tokenizers gguf

# modelli (mkdir models se non esiste già):
curl -sL -o models/Qwen3-0.6B-Q8_0.gguf   "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
curl -sL -o models/Qwen3-0.6B-Q4_K_M.gguf "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
curl -sL -o models/Qwen3-0.6B-Q5_K_M.gguf "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q5_K_M.gguf"
curl -sL -o models/Qwen3-0.6B-Q6_K.gguf   "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q6_K.gguf"
curl -sL -o models/Qwen3-4B-Q4_K_M.gguf   "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
curl -sL -o models/Qwen3-8B-Q4_K_M.gguf   "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"

# MoE (Qwen3-30B-A3B, 18.6GB — 128 esperti, 8 attivi per token):
curl -sL -o models/Qwen3-30B-A3B-Q4_K_M.gguf "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"

# oracolo tokenizer, uno per membro della famiglia (check_m1.py valida
# ognuno contro il SUO tokenizer.json, non dà per scontato che sia
# identico tra 0.6B/4B/8B; nessuna cache HuggingFace coinvolta):
curl -sL -o models/Qwen3-0.6B-tokenizer.json "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
curl -sL -o models/Qwen3-4B-tokenizer.json   "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json"
curl -sL -o models/Qwen3-8B-tokenizer.json   "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json"
```

## Roadmap

| Milestone | Contenuto | Criterio di successo |
|---|---|---|
| **M0** ✅ | Lettore GGUF: header, metadati, tensori | ispeziona un GGUF reale senza errori |
| **M1** ✅ | Tokenizer BPE byte-level | 11/11 prompt identici all'oracolo HF |
| **M2** ✅ | Forward pass fp32, un token | argmax uguale, coseno 0.9998, top-10 10/10 |
| **M3** ✅ | KV cache + generazione | selfcheck bit-esatto; greedy 5/20 = oracolo poi diverge (drift Q8_0) |
| **M4** ✅ | Quantizzazione: pesi mai espansi, dequant inline | Q8_0 603.9MB/coseno 0.9998; Q4_K_M 372.4MB/coseno 0.9758 |
| **M5** ✅ | mmap + OpenMP + AVX2 | 4.25x (ctx 128) → 2.91x (ctx 512); caricamento 0.19s→0.04s |
| **M6** ✅ | Chat template, REPL, KV su disco | conversazione multi-turno, sessione ripresa tra processi separati |

Oltre la roadmap originale (vedi [RETROSPECTIVE.md](RETROSPECTIVE.md)
per il dettaglio): penalità di ripetizione, vettorializzazione dell'attenzione, RoPE precalcolato,
prefill batched, **kernel interi** (attivazioni Q8 + maddubs), **KV
cache f16**, testa di output separata (Qwen3 8B+), stop testuale,
`--no-think`, decodifica speculativa (`--draft`), **MoE** (Qwen3-30B-A3B:
router + top-8/128 esperti).

La disciplina (da DwarfStar): **nessuna milestone è finita finché non è
validata contro l'oracolo**. I vettori golden si generano con
`tools/golden.py` (richiede `pip install torch transformers`).

## Modelli provati e raccomandazioni (macchina di riferimento)

| Modello | RAM | Decode | Uso consigliato |
|---|---|---|---|
| Qwen3-0.6B Q4_K_M | 372MB | ~15 tok/s | velocità pura, **solo con --no-think** (il thinking collassa a Q4 su 0.6B) |
| Qwen3-0.6B Q6_K | 472MB | ~13 tok/s | il miglior 0.6B tuttofare |
| **Qwen3-4B Q4_K_M** | 2.5GB | ~2.6 tok/s | **il punto dolce**: --no-think per l'uso quotidiano, thinking per i compiti di ragionamento |
| Qwen3-8B Q4_K_M | 4.8GB | ~1.2 tok/s | massima qualità, con pazienza |
| Qwen3-30B-A3B Q4_K_M (MoE) | 17.7GB | ~3.1 tok/s | qualità di un modello grande, più veloce del 4B (solo 8/128 esperti attivi per token) |

(Questa tabella è precedente a diverse ottimizzazioni che riguardano
le MoE descritte in [RETROSPECTIVE.md](RETROSPECTIVE.md); il numero di
decode del 30B-A3B è più alto sul motore attuale — vedi i numeri
principali di README.md per la misura aggiornata.)

## Test

Tutta la batteria di regressione in un comando:

```
venv-tools/Scripts/python.exe tests/run_all.py          # suite base (~2-3 min)
venv-tools/Scripts/python.exe tests/run_all.py --full   # + selfcheck/dequant su 4B e 8B
```

## Build

**Windows (MSVC / Visual Studio 2022):**
```
build.bat
```

**Linux (gcc) — rispecchia build.bat, produce `./nf`:**
```
sh build_posix.sh nf
```

## Uso

```
# M0 — ispezione (test rapido senza modello: GGUF sintetico):
python tools/make_test_gguf.py tests/test.gguf
nf.exe inspect tests/test.gguf --tensors

# M0 — con il modello vero (in models/, scaricato da Qwen/Qwen3-0.6B-GGUF):
nf.exe inspect models/Qwen3-0.6B-Q8_0.gguf

# M1 — tokenizzazione (testo ASCII inline, oppure --file per UTF-8):
nf.exe tokenize models/Qwen3-0.6B-Q8_0.gguf "ciao mondo"
nf.exe tokenize models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt

# M1 — regressione contro l'oracolo HuggingFace:
python tests/check_m1.py

# M2 — logit dell'ultima posizione + confronto con l'oracolo fp32:
nf.exe logits models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt
venv-tools/Scripts/python.exe tests/check_m2.py

# M3 — generazione (greedy di default; --temp/--top-p per il sampling):
nf.exe generate models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt --n-predict 30
nf.exe generate models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt --n-predict 30 --temp 0.7 --top-p 0.9 --seed 7

# Decodifica speculativa (solo greedy): un modello piccolo propone K token,
# il grande li verifica in una passata — output GARANTITO identico al
# non-speculativo. Nota onesta: su questa macchina non e' piu' veloce
# (il decode e' compute-bound dopo i kernel interi); resta per
# macchine/kernel memory-bound.
nf.exe generate models/Qwen3-8B-Q4_K_M.gguf --file prompt.txt --n-predict 64 \
    --draft models/Qwen3-0.6B-Q4_K_M.gguf --draft-k 6

# M3 — auto-consistenza (cache incrementale vs ricalcolo completo, nessun oracolo):
nf.exe selfcheck models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt

# M3 — confronto greedy con l'oracolo HuggingFace:
venv-tools/Scripts/python.exe tests/check_m3.py

# M4 — stesso identico comando di M2, ma su un GGUF Q4_K_M invece di Q8_0
# (i pesi restano quantizzati in RAM, dequant inline nel dot product):
venv-tools/Scripts/python.exe tests/check_m2.py models/Qwen3-0.6B-Q4_K_M.gguf

# M4 — verifica il dequantizzatore Q4_K/Q6_K contro il riferimento ufficiale:
venv-tools/Scripts/python.exe tests/check_m4_dequant.py

# M5 — benchmark prefill/decode per frontiera di contesto (CSV):
python tools/make_bench_prompt.py tests/bench_prompt.txt
nf.exe bench models/Qwen3-0.6B-Q4_K_M.gguf --file tests/bench_prompt.txt \
    --ctx-start 128 --ctx-max 1024 --step-mul 2 --gen-tokens 16

# M6 — chat interattiva (ChatML, KV riusata tra i turni). Default greedy
# (--temp 0): su un modello da 0.6B il campionamento (temp>0) puo' far
# deragliare il ragionamento su compiti di codice/logica — verificato
# confrontando greedy contro l'oracolo HF fp32.
nf.exe chat models/Qwen3-0.6B-Q4_K_M.gguf --system "Sei un assistente conciso."
# sui quant aggressivi (Q4_K_M) il ragionamento esteso puo' avvitarsi e non
# convergere mai: --no-think lo salta e il modello risponde diretto (stessa
# domanda: da 900 token bruciati senza risposta a risposta corretta in 32).
# Il thinking affidabile richiede Q5_K_M o superiori.
nf.exe chat models/Qwen3-0.6B-Q4_K_M.gguf --system "Sei un assistente conciso." --no-think
# per varieta' invece di affidabilita', attiva il campionamento esplicitamente:
nf.exe chat models/Qwen3-0.6B-Q4_K_M.gguf --temp 0.7 --top-p 0.9
# /quit o /exit per uscire. --session file.bin per salvare/riprendere:
nf.exe chat models/Qwen3-0.6B-Q4_K_M.gguf --session conversazione.bin
# (la seconda invocazione, anche in un processo nuovo, riprende da dove
# la prima si era fermata — vedi tests/check_m6.py)

# M6 — regressione chat multi-turno + sessione, tra due processi separati:
python tests/check_m6.py

# MoE (Qwen3-30B-A3B) — stesso identico comando di generate/chat/bench,
# l'architettura (qwen3 denso o qwen3moe) si rileva dal GGUF:
nf.exe generate models/Qwen3-30B-A3B-Q4_K_M.gguf --file prompt.txt --n-predict 30
nf.exe chat models/Qwen3-30B-A3B-Q4_K_M.gguf --system "Sei un assistente conciso."

# MoE — validazione router+esperti contro una reimplementazione Python
# indipendente sui pesi reali (niente oracolo HF: il modello non entra
# in RAM in bf16 su questa macchina):
venv-tools/Scripts/python.exe tests/check_moe.py
```

## Documentazione concettuale

I concetti dietro ogni milestone (token, tensori, KV cache, logit,
softmax, quantizzazione, GGUF/GGML) sono nati dallo studio di
[DwarfStar](https://github.com/antirez), il progetto che ha preceduto
questo — tenuti come appunti di studio privati, non parte di questo
repository.
