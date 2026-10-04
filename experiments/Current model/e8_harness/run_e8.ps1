# Windows PowerShell version of run_e8.sh.  Usage:  .\e8_harness\run_e8.ps1 [-Modes fedavg,adaptive_krum] [-DataDir D:\data] [-TrainRows 40000]
param([string[]]$Modes = @("fedavg","adaptive_krum","calibrated_krum","dp_calibrated_krum"), [string[]]$Profiles = @("1.0","0.5"),
      [string]$DataDir = "", [int]$TrainRows = 0, [int]$TestRows = 0, [string]$DataSource = "auto", [string]$Image = "flids-e8:latest",
      [string]$ResultsRoot = (Join-Path (Get-Location) "e8_results"))
$ErrorActionPreference = "Stop"
$Harness = $PSScriptRoot; $Project = Split-Path $Harness -Parent; $Net = "e8bench-net"
docker image inspect $Image *> $null; if ($LASTEXITCODE -ne 0) { docker build -t $Image $Harness }
docker network inspect $Net *> $null; if ($LASTEXITCODE -ne 0) { docker network create $Net | Out-Null }
foreach ($mode in $Modes) { foreach ($prof in $Profiles) {
  $tag = "${mode}_${prof}vcpu"; $out = Join-Path $ResultsRoot $tag; New-Item -ItemType Directory -Force $out | Out-Null
  $cfg = "/app/e8_harness/configs/$tag.json"; Write-Host "=== $tag ==="
  $lim = @("--cpus=$prof","--memory=2048m","--memory-swap=2048m"); $common = @("--rm","--network",$Net,"-e","PROJECT_ROOT=/app","-v","${Project}:/app:ro")
  $srv = Start-Process docker -PassThru -NoNewWindow -RedirectStandardOutput "$out\server.log" -RedirectStandardError "$out\server.err" -ArgumentList (@("run") + $common + $lim + @("--name","e8-server-$tag","-v","${out}:/results",$Image,"python","/app/e8_harness/docker_bench_server.py","--config",$cfg,"--port","9000","--out-dir","/results"))
  Start-Sleep 3; $procs = @()
  foreach ($cid in 0,1) {
    $dargs = @(); if ($DataDir) { $dargs = @("-v","${DataDir}:${DataDir}:ro","-e","DATA_DIR=$DataDir") }
    $extra = @("--data-source",$DataSource); if ($TrainRows -gt 0) { $extra += @("--train-rows",$TrainRows) }; if ($TestRows -gt 0) { $extra += @("--test-rows",$TestRows) }
    $procs += Start-Process docker -PassThru -NoNewWindow -RedirectStandardOutput "$out\client$cid.log" -RedirectStandardError "$out\client$cid.err" -ArgumentList (@("run") + $common + $lim + $dargs + @("--name","e8-client$cid-$tag","-v","${out}:/results",$Image,"python","/app/e8_harness/docker_bench_client.py","--config",$cfg,"--client-id",$cid,"--server-host","e8-server-$tag","--server-port","9000","--out-dir","/results") + $extra)
  }
  $procs | Wait-Process; $srv | Wait-Process
  if (Test-Path "$out\server_${mode}_results.json") { Write-Host "done: $out" } else { Write-Host "NO SERVER RESULT for $tag (see $out\server.err)" }
}}
docker network rm $Net *> $null; Write-Host "Finished. Send back: $ResultsRoot"
