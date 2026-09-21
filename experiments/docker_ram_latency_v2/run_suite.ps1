# PowerShell equivalent of run_suite.sh.
#
# Runs the full RAM/timing test matrix:
#   modes:    baseline, he_full, he_partial, he_partial_zkp, dp
#   profiles: unthrottled (2GB/1.0 vCPU), throttled (2GB/0.5 vCPU)
# for the given MODEL_TYPE (default: network).
#
# ORCHESTRATION NOTE (fixed from an earlier version): client0 and
# client1 almost never finish at exactly the same time (different
# partition sizes -> different training time). This script brings the
# communication daemon (`server`) up first with --wait, then runs both
# clients as independent one-shot `run --rm` processes (via
# Start-Process, so output streams live) instead of `up
# --abort-on-container-exit`, which kills every other service the
# instant the FIRST one exits -- that was silently truncating whichever
# client finished second, mid-run.
#
# Usage:
#   .\run_suite.ps1                                                    # everything
#   .\run_suite.ps1 -ModelType network -Modes dp                       # just dp, both profiles
#   .\run_suite.ps1 -ModelType network -Modes dp -Profiles throttled   # just dp, throttled only
#   .\run_suite.ps1 -ModelType network -Modes he_partial,he_partial_zkp # both, both profiles
#
# Prereqs:
#   1. Build partitions first (once per model type) -- see offline\build_docker_partitions.py
#   2. Docker Desktop running.

param(
    [string]$ModelType = "network",
    [string[]]$Modes = @("baseline", "he_full", "he_partial", "he_partial_zkp", "dp"),
    [string[]]$Profiles = @("unthrottled", "throttled")
)

$ErrorActionPreference = "Stop"

$ManifestPath = "partitions\$ModelType\manifest.json"
if (-not (Test-Path $ManifestPath)) {
    Write-Host "ERROR: $ManifestPath not found."
    Write-Host "Run the offline partition builder first -- see README.md."
    exit 1
}

New-Item -ItemType Directory -Force -Path "results" | Out-Null

Write-Host "Building images once up front..."
docker compose build

foreach ($mode in $Modes) {
    foreach ($profile in $Profiles) {
        $RunTag = "${ModelType}_${mode}_${profile}"
        Write-Host ""
        Write-Host "=============================================="
        Write-Host " RUN: $RunTag"
        Write-Host "=============================================="

        $env:MODE = $mode
        $env:MODEL_TYPE = $ModelType
        $env:RUN_TAG = $RunTag

        if ($profile -eq "throttled") {
            $ComposeFiles = @("-f", "docker-compose.yml", "-f", "docker-compose.throttled.yml")
        } else {
            $ComposeFiles = @("-f", "docker-compose.yml")
        }

        New-Item -ItemType Directory -Force -Path "results\$RunTag" | Out-Null

        # Phase 1a: bring the communication daemon up and wait for it
        # to be healthy BEFORE either client starts.
        docker compose @ComposeFiles up -d --wait server

        # Phase 1b: run both clients as independent one-shot processes,
        # concurrently, neither one able to kill the other on exit.
        # -NoNewWindow streams their output live into this console,
        # interleaved (same as bash backgrounding + wait).
        $client0Args = @("compose") + $ComposeFiles + @("run", "--rm", "client0")
        $client1Args = @("compose") + $ComposeFiles + @("run", "--rm", "client1")

        $p0 = Start-Process -FilePath "docker" -ArgumentList $client0Args -NoNewWindow -PassThru
        $p1 = Start-Process -FilePath "docker" -ArgumentList $client1Args -NoNewWindow -PassThru
        Wait-Process -Id $p0.Id, $p1.Id

        if ($p0.ExitCode -ne 0 -or $p1.ExitCode -ne 0) {
            Write-Host "WARNING: client0 exit=$($p0.ExitCode) client1 exit=$($p1.ExitCode) -- check output above."
        }

        # Phase 2: server-side aggregation timing (Krum / HE-aggregate+
        # decrypt / ZKP-verify+threshold). The daemon should have
        # already self-exited (it shuts down once it's received every
        # expected submission) -- `down -v` afterward cleans up
        # regardless.
        docker compose @ComposeFiles --profile aggregate run --rm aggregator

        docker compose @ComposeFiles down -v

        Write-Host " Done: $RunTag  ->  results\$RunTag\"
    }
}

Write-Host ""
Write-Host "All runs complete. Results in .\results\<model>_<mode>_<profile>\"
Write-Host "Run: python consolidate_results.py results"
