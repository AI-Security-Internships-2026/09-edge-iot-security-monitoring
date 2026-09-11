<#
.SYNOPSIS
    scripts/run_task4_federated_sweep.ps1
    Native PowerShell version -- no bash, no WSL, no tmux. Runs everything
    sequentially in the foreground using your Windows python.exe / venv.

.DESCRIPTION
    Task 4 (E1 baseline table) -- FEDERATED CONDITIONS ONLY.
    NOT covered here: the CENTRALIZED conditions (CNN-LSTM, MLP, RF,
    XGBoost) -- those come from scripts/train_centralized.py.

    Runs, per the ticket's schedule, for BOTH models (network,
    application) x 5 seeds (42,123,456,789,2024):
      1. FedAvg            -- PROX_MU=0, AGGREGATOR=fedavg
      2. FedProx mu-sweep  -- PROX_MU in {0, 0.005, 0.02, 0.05, 0.1}
      3. Arch-swap variant -- run ONLY after reading the argmax-Macro-F1
                                mu off stage 2's aggregated VALIDATION
                                results. --force-dp-safe-arch, USE_DP
                                stays False.

    Make sure your venv/conda env is ACTIVATED in this PowerShell session
    before running this script, so `python` resolves to the right
    interpreter with this project's dependencies installed.

.PARAMETER Stage
    One of: timing_probe, fedavg_and_prox_sweep, arch_swap

.PARAMETER BestMu
    Required for arch_swap. The argmax mu from stage 2's VALIDATION
    results (see aggregate_task4_results.py --stage prox_sweep).

.PARAMETER Model
    Optional, for arch_swap only. One of: network, application.
    Needed when network and application argmax to DIFFERENT mu values --
    run arch_swap twice, once per model, each with its own -BestMu.
    Omit to run both models with the same -BestMu.

.EXAMPLE
    .\run_task4_federated_sweep.ps1 -Stage timing_probe

.EXAMPLE
    .\run_task4_federated_sweep.ps1 -Stage fedavg_and_prox_sweep

.EXAMPLE
    .\run_task4_federated_sweep.ps1 -Stage arch_swap -BestMu 0.005 -Model network
    .\run_task4_federated_sweep.ps1 -Stage arch_swap -BestMu 0     -Model application
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("timing_probe", "fedavg_and_prox_sweep", "arch_swap")]
    [string]$Stage,

    [string]$BestMu,

    [ValidateSet("network", "application")]
    [string]$Model
)

$ErrorActionPreference = "Stop"
# PowerShell 7.3+ treats native-command STDERR output as terminating errors
# when $ErrorActionPreference is "Stop" -- that would kill this script on
# the FIRST line python writes to stderr (even mid-traceback), hiding the
# real error. Disable that specifically so python's own output streams
# through untouched; PowerShell cmdlet errors in this script still stop
# normally.
$PSNativeCommandUseErrorActionPreference = $false

$Seeds = @(42, 123, 456, 789, 2024)
$Models = if ($Model) { @($Model) } else { @("network", "application") }
$MuValues = @("0", "0.005", "0.02", "0.05", "0.1")
$ResultsDir = "experiments/results/task4_federated"

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

# Windows + OpenBLAS is prone to "Memory allocation still failed after N
# retries" when many BLAS thread pools get spawned (e.g. once per
# simulated FL client) and oversubscribe available memory/threads. Cap
# thread counts unless you've already set these yourself in this session.
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "4" }
if (-not $env:OMP_NUM_THREADS)      { $env:OMP_NUM_THREADS = "4" }
if (-not $env:MKL_NUM_THREADS)      { $env:MKL_NUM_THREADS = "4" }

function Show-Usage {
    Write-Host "Usage: .\run_task4_federated_sweep.ps1 -Stage {timing_probe|fedavg_and_prox_sweep|arch_swap}"
    Write-Host "  1. .\run_task4_federated_sweep.ps1 -Stage timing_probe"
    Write-Host "  2. .\run_task4_federated_sweep.ps1 -Stage fedavg_and_prox_sweep"
    Write-Host "  3. python scripts/aggregate_task4_results.py --stage prox_sweep --results-dir $ResultsDir"
    Write-Host "  4. .\run_task4_federated_sweep.ps1 -Stage arch_swap -BestMu <argmax from step 3>"
    Write-Host "     If network and application argmax to DIFFERENT mu, run arch_swap twice instead:"
    Write-Host "       .\run_task4_federated_sweep.ps1 -Stage arch_swap -BestMu <network argmax>     -Model network"
    Write-Host "       .\run_task4_federated_sweep.ps1 -Stage arch_swap -BestMu <application argmax> -Model application"
    Write-Host "  5. python scripts/aggregate_task4_results.py --stage full --results-dir $ResultsDir"
}

if (-not $Stage) {
    Show-Usage
    exit 1
}

if ($Stage -eq "timing_probe") {
    Write-Host "Running a single foreground timing probe (network, seed=42, FedAvg)..."
    Write-Host "Get this number before committing to the full schedule."
    Measure-Command {
        python main.py network --ablation-mode baseline --aggregator fedavg `
            --prox-mu 0 --seed 42 --tag task4_timing_probe
    } | Select-Object TotalSeconds
    exit 0
}

if ($Stage -eq "fedavg_and_prox_sweep") {
    Write-Host "Running FedAvg + FedProx mu-sweep sequentially in the foreground."
    Write-Host ("Models: {0} | Seeds: {1} | Mu values: {2}" -f ($Models -join ","), ($Seeds -join ","), ($MuValues -join ","))

    foreach ($model in $Models) {
        foreach ($seed in $Seeds) {
            # Condition 1: FedAvg (mu=0)
            $tag = "task4_fedavg"
            Write-Host "=== $tag ($model, seed=$seed) ==="
            $logPath = "$ResultsDir/log_${tag}_${model}_seed${seed}.txt"
            python main.py $model --ablation-mode baseline --aggregator fedavg `
                --prox-mu 0 --seed $seed --tag $tag 2>&1 | Tee-Object -FilePath $logPath

            # Condition 2: FedProx mu-sweep
            foreach ($mu in $MuValues) {
                $tag = "task4_fedprox_mu$mu"
                Write-Host "=== $tag ($model, seed=$seed) ==="
                $logPath = "$ResultsDir/log_${tag}_${model}_seed${seed}.txt"
                python main.py $model --ablation-mode baseline --aggregator fedavg `
                    --prox-mu $mu --seed $seed --tag $tag 2>&1 | Tee-Object -FilePath $logPath
            }
        }
    }

    Write-Host "Stage 1+2 complete. Now run:"
    Write-Host "  python scripts/aggregate_task4_results.py --stage prox_sweep --results-dir $ResultsDir"
    Write-Host "to get the argmax-Macro-F1 mu, then re-run this script with -Stage arch_swap and"
    Write-Host "-BestMu <that value> (per-model if they differ -- see header comment)."
    exit 0
}

if ($Stage -eq "arch_swap") {
    if (-not $BestMu) {
        Write-Host "ERROR: pass -BestMu <argmax mu from stage 2's aggregated results> first."
        Write-Host "  e.g. .\run_task4_federated_sweep.ps1 -Stage arch_swap -BestMu 0.02"
        Write-Host "  or, for a per-model mu:"
        Write-Host "  .\run_task4_federated_sweep.ps1 -Stage arch_swap -BestMu 0.005 -Model network"
        Write-Host "  .\run_task4_federated_sweep.ps1 -Stage arch_swap -BestMu 0     -Model application"
        exit 1
    }

    Write-Host ("Running arch-swap variant (PROX_MU={0}, Models={1}) sequentially in the foreground." -f $BestMu, ($Models -join ","))

    foreach ($model in $Models) {
        foreach ($seed in $Seeds) {
            $tag = "task4_fedprox_dpsafe_arch_no_dp"
            Write-Host "=== $tag ($model, seed=$seed, PROX_MU=$BestMu, --force-dp-safe-arch, USE_DP stays False) ==="
            $logPath = "$ResultsDir/log_${tag}_${model}_seed${seed}.txt"
            python main.py $model --ablation-mode baseline --aggregator fedavg `
                --prox-mu $BestMu --force-dp-safe-arch `
                --seed $seed --tag $tag 2>&1 | Tee-Object -FilePath $logPath
        }
    }

    Write-Host ("arch_swap (Models={0}) complete." -f ($Models -join ","))
    exit 0
}

Show-Usage
exit 1
