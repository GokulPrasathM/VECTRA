param(
  [string]$RepoUrl = "https://github.com/GokulPrasathM/VECTRA.git",
  [string]$Branch = "main",
  [string]$WorkDir = "$PWD/vectra_run",

  # Python
  [string]$Python = "python3",
  [string]$VenvDir = ".venv",

  # Model + run params
  [string]$ModelId = "openai/gpt-oss-20b",
  [int]$SamplesPerSuite = 25,
  [int]$Seed = 7,

  # Throughput
  [int]$BatchMaxSize = 8,
  [int]$BatchMaxWaitMs = 8,

  # Device placement
  [switch]$ForceSingleCudaDevice,
  [int]$CudaDeviceIndex = 0,
  [string]$DeviceMap = "",

  # VECTRA params
  [int]$VectraAttempts = 4,
  [int]$VectraEarlyStop = 2,
  [int]$VectraMaxTurns = 16,
  [int]$VectraProblemConcurrency = 2,
  [int]$BaselineConcurrency = 8
)

$ErrorActionPreference = "Stop"

function Write-Header([string]$msg) {
  $ts = (Get-Date).ToString("s")
  Write-Host "`n[$ts] $msg`n"
}

function Exec([string]$cmd) {
  Write-Host "> $cmd"
  iex $cmd
}

Write-Header "VECTRA terminal demo bootstrap (pwsh)"

# 1) Prepare directories
$WorkDir = (Resolve-Path -Path $WorkDir -ErrorAction SilentlyContinue) ?? $WorkDir
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Set-Location $WorkDir

# Timestamped run folder (keeps all evidence artifacts)
$RunId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$OutDir = Join-Path $WorkDir "runs/$RunId"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Tee ALL console output into a transcript for evidence logging
$TranscriptPath = Join-Path $OutDir "terminal_transcript.txt"
Start-Transcript -Path $TranscriptPath -Append | Out-Null

try {
  Write-Header "System snapshot"
  "RunId=$RunId" | Out-File -FilePath (Join-Path $OutDir "run_id.txt") -Encoding utf8

  # Basic system info
  Exec "uname -a | tee $OutDir/uname.txt"
  Exec "$Python --version | tee $OutDir/python_version.txt"

  # GPU snapshot (best-effort)
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Exec "nvidia-smi -L | tee $OutDir/nvidia_smi_L.txt"
    Exec "nvidia-smi | tee $OutDir/nvidia_smi.txt"
  } else {
    "nvidia-smi not found" | Out-File -FilePath (Join-Path $OutDir "nvidia_smi_missing.txt") -Encoding utf8
  }

  Write-Header "Clone/update repo"
  $RepoDir = Join-Path $WorkDir "VECTRA"
  if (Test-Path $RepoDir) {
    Set-Location $RepoDir
    Exec "git fetch origin $Branch"
    Exec "git reset --hard origin/$Branch"
  } else {
    Exec "git clone --depth 1 --branch $Branch $RepoUrl $RepoDir"
    Set-Location $RepoDir
  }
  Exec "git rev-parse HEAD | tee $OutDir/git_rev.txt"

  Write-Header "Create venv + install dependencies"
  Exec "$Python -m venv $VenvDir"

  # Activate
  Exec "source $VenvDir/bin/activate"
  Exec "python -m pip install -U pip wheel setuptools"

  # IMPORTANT: torch install is machine-specific (CUDA build). We do not auto-install torch here.
  # Install the rest.
  Exec "python -m pip install -U transformers accelerate datasets httpx"

  # Capture pip freeze
  Exec "python -m pip freeze | tee $OutDir/pip_freeze.txt"

  Write-Header "Run impact demo (terminal CLI)"

  $Args = @(
    "tools/run_vectra_impact_cli.py",
    "--model-id", $ModelId,
    "--samples-per-suite", "$SamplesPerSuite",
    "--seed", "$Seed",
    "--batch-max-size", "$BatchMaxSize",
    "--batch-max-wait-ms", "$BatchMaxWaitMs",
    "--baseline-concurrency", "$BaselineConcurrency",
    "--vectra-attempts", "$VectraAttempts",
    "--vectra-early-stop", "$VectraEarlyStop",
    "--vectra-max-turns", "$VectraMaxTurns",
    "--vectra-problem-concurrency", "$VectraProblemConcurrency",
    "--cuda-device-index", "$CudaDeviceIndex",
    "--log-dir", $OutDir
  )

  if ($DeviceMap -ne "") {
    $Args += @("--device-map", $DeviceMap)
  } else {
    # Default behavior: avoid mixed CPU/GPU sharding issues by forcing single CUDA device.
    if ($ForceSingleCudaDevice.IsPresent) {
      $Args += "--force-single-cuda-device"
    } else {
      $Args += "--force-single-cuda-device"
    }
  }

  # Run and tee to a separate log in addition to transcript
  $CliLog = Join-Path $OutDir "cli_stdout.txt"
  Exec ("python " + ($Args -join " ") + " | tee $CliLog")

  Write-Header "Done"
  Write-Host "Evidence folder: $OutDir"
  Write-Host "Key files:"
  Write-Host "- terminal_transcript.txt"
  Write-Host "- meta.json / summary.json / run.jsonl"
  Write-Host "- nvidia_smi.txt / pip_freeze.txt / git_rev.txt"
}
finally {
  try { Stop-Transcript | Out-Null } catch {}
}
