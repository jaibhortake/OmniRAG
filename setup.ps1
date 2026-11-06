# OmniRAG setup for the demo machine (Windows, NVIDIA GPU).
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1 [-Cpu] [-SkipModels]
param([switch]$Cpu, [switch]$SkipModels)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Python environment" -ForegroundColor Cyan
if (-not (Test-Path backend\.venv)) { py -3.12 -m venv backend\.venv }
$py = "backend\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
if ($Cpu) { & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu }
else      { & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu121 }
& $py -m pip install -r backend\requirements.txt -r backend\requirements-ml.txt

Write-Host "==> Frontend build" -ForegroundColor Cyan
Push-Location frontend; npm ci --no-audit --no-fund; npm run build; Pop-Location

if (-not (Test-Path backend\.env)) { Copy-Item backend\.env.example backend\.env }

if (-not $SkipModels) {
  Write-Host "==> Downloading local models (one time, ~6 GB)" -ForegroundColor Cyan
  ollama pull qwen2.5:3b
  ollama pull moondream
  & $py -m spacy download en_core_web_sm
  # warm the HuggingFace cache so the demo works with the network unplugged
  & $py -c "from app.config import get_settings; from app.providers.real import build; p=build(get_settings()); p.text.embed(['warmup']); p.reranker.score('a',['b']); p.clip.embed_texts(['warmup']); p.stt.get(); p.ocr.get(); print('models cached')"
}
Write-Host "Done. Start with:  .\run.ps1   then open http://localhost:8000" -ForegroundColor Green
