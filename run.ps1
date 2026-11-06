# Starts OmniRAG (API + built UI) on http://localhost:8000
# -Fake runs without any model weights (deterministic stand-ins), useful for UI rehearsal.
param([switch]$Fake, [int]$Port = 8000)
Set-Location "$PSScriptRoot\backend"
if ($Fake) { $env:OMNIRAG_PROVIDERS = "fake" }
else { $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1" }  # prove it: no network calls at runtime
& .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $Port
