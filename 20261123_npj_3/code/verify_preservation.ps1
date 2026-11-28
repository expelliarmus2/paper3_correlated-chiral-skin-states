param()
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$items = Get-Content -LiteralPath (Join-Path $root 'source_manifest.json') -Raw | ConvertFrom-Json
foreach ($item in $items) {
    $path = Join-Path $root $item.current_path
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing: $($item.current_path)" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $item.sha256) {
        throw "Changed source: $($item.current_path)"
    }
    if ($path.EndsWith('.ipynb')) {
        $notebook = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        if ($notebook.nbformat -ne 4) { throw "Unexpected notebook format: $path" }
    }
}
Write-Output "Verified $($items.Count) original source files; notebook JSON is valid."
