# Local GPU Reranker

This is a host-native FastAPI service for `BAAI/bge-reranker-v2-m3`. It uses
the same `POST /rerank` wire contract as Text Embeddings Inference, so the
existing EchoMind retrieval code can call it without an adapter change.

It is intentionally outside the Docker application image. The service loads
the model once on the Windows host GPU and Docker Desktop reaches it through
`host.docker.internal:18080`.

## Install and start

Run PowerShell from the repository root:

```powershell
.\tools\install_local_reranker.ps1
.\tools\start_local_reranker.ps1
```

The installer creates `.venv-reranker`, installs a CUDA 13.0 PyTorch wheel,
then verifies CUDA is available before installing the server dependencies. It
fails rather than silently falling back to CPU. If the host uses a different
Python executable, pass it explicitly:

```powershell
.\tools\install_local_reranker.ps1 -PythonExecutable C:\Python312\python.exe
```

The service pins `transformers==4.57.3`. Do not upgrade Transformers to v5:
that major version removed a tokenizer API still used by FlagReranker.

When the service is ready, verify it from the host:

```powershell
Invoke-RestMethod http://127.0.0.1:18080/health
```

Expected response includes `backend: FlagEmbedding` and `device: cuda:0`.

## Docker connection

The default Compose configuration calls
`http://host.docker.internal:18080/rerank`. Start this service before starting
or restarting EchoMind. If an older TEI container is still running, stop it
first so it does not continue to consume resources:

```powershell
docker compose --profile tei stop reranker
docker compose up -d --build echomind
docker compose exec echomind curl -sS http://host.docker.internal:18080/health
```

The service uses one Uvicorn worker and serializes GPU inference. This avoids
loading duplicate models or allowing concurrent cross-encoder calls to exhaust
the GPU. Each EchoMind request still scores all 12 recall candidates in one
model batch. Scores are sigmoid-normalized to the 0-to-1 range so they match
the existing TEI-compatible threshold semantics.

## Optional TEI fallback

TEI remains available only as an explicit Compose profile. To use it instead
of the host service, set the container URL before recreating EchoMind:

```powershell
$env:RERANK_URL = "http://reranker:80/rerank"
docker compose --profile tei up -d reranker echomind
```

Remove `RERANK_URL` from the shell and recreate `echomind` to restore the
host-native default.
