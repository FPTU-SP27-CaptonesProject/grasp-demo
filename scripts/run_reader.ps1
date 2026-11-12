param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $LlamaServer = "llama-server.exe",
    [int] $Port = 8080,
    [int] $Context = 8192,
    [int] $Parallel = 4
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Model)) { throw "GGUF model not found: $Model" }
& $LlamaServer -m $Model -ngl 99 -c $Context --parallel $Parallel --port $Port
