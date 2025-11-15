# run_all.ps1
Write-Host "===== STARTING FULL PIPELINE ====="

$steps = @(
    "python src/preprocess.py",
    "python src/inspect_data.py",
    "python src/train.py",
    "python src/shap_explain.py",
    "python src/federated_sim.py",
    "python src/federated_robust.py",
    "python src/federated_secure.py"
)

foreach ($step in $steps) {
    Write-Host "`n=== EXECUTING: $step ==="
    $start = Get-Date
    Invoke-Expression $step
    $end = Get-Date
    $duration = New-TimeSpan -Start $start -End $end
    Write-Host "Completed in $($duration.TotalSeconds) seconds"
}

Write-Host "`n===== ALL STEPS COMPLETED ====="
