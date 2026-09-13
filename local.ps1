param(
    [ValidateSet('serve', 'codex', 'claude')][string]$Action = 'serve',
    [ValidateSet('measured', 'whole-layer', 'stock-speculative', 'target-only')][string]$Profile = 'measured',
    [int]$Port = 8080,
    [string]$Workspace = (Get-Location).Path,
    [ValidateSet('manual', 'auto')][string]$ClaudeCompaction = 'manual',
    [switch]$PrintCommand
)
$sessionPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $sessionPython)) { throw 'Run local.ps1 from the main checkout with its measured .venv.' }
$sessionArgs = @((Join-Path $PSScriptRoot 'scripts/local_session.py'), $Action, '--root', $PSScriptRoot,
    '--profile', $Profile, '--port', "$Port", '--workspace', $Workspace,
    '--claude-compaction', $ClaudeCompaction)
if ($PrintCommand) { $sessionArgs += '--print-command' }
& $sessionPython @sessionArgs
exit $LASTEXITCODE
