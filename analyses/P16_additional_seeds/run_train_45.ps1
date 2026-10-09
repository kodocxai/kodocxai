# P16 seed 4/5 training runner (2026-10-05)
# Encoding-proof: no non-ASCII literals; paths derived from $PSScriptRoot.
# Log = train_log.txt, completion marker = train_done.txt (one line per seed).
# "Continue": native stderr (e.g., transformers' harmless init notice) must not
# abort the run; real failures are caught via $LASTEXITCODE below.
$ErrorActionPreference = "Continue"
$here = $PSScriptRoot                     # ...\_exp\P16_<folder>
$exp = Split-Path $here -Parent           # ...\_exp
$py = "D:\kodx-venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $exp
$log = Join-Path $here "train_log.txt"
$done = Join-Path $here "train_done.txt"

foreach ($seed in 4, 5) {
    "=== seed $seed start $(Get-Date -Format 'HH:mm:ss') ===" | Add-Content $log
    & $py -W ignore -X utf8 kodx_train.py `
        --train data\train.jsonl --images data\images224 `
        --out "runs\seed$seed" --seed $seed *>> $log
    if ($LASTEXITCODE -ne 0) {
        "seed $seed FAILED (exit $LASTEXITCODE)" | Add-Content $log
        "FAIL seed$seed $(Get-Date -Format 'HH:mm:ss')" | Add-Content $done
        exit 1
    }
    "=== seed $seed done $(Get-Date -Format 'HH:mm:ss') ===" | Add-Content $log
    "OK seed$seed $(Get-Date -Format 'HH:mm:ss')" | Add-Content $done
}
"ALL_DONE $(Get-Date -Format 'HH:mm:ss')" | Add-Content $done
