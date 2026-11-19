<p align="center"><img src="brand/delbina-logo-transparent-1024.png" width="160" alt="delbina"></p>

# delbina

A context-compression layer that sits in front of existing LLMs to cut the token cost of long chats. Each chat turn re-sends the whole history, so cost grows roughly quadratically with chat length; delbina measures that and compresses history before it reaches the model.

## Status

Early. Baseline token benchmark only.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python bench\run_bench.py
```

## Layout

- `src/delbina/` - library (token counting, compression strategies)
- `bench/` - benchmarks and sample chats
- `tests/` - pytest tests
