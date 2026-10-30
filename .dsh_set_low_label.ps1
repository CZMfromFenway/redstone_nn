# Restore the Low mandatory integrity label on a directory subtree.
# Diagnosis: the DSH confined token runs at Low integrity. E:\redstone_nn itself carries
# "Mandatory Label\Low Mandatory Level:(OI)(CI)(NW)" but its children do not, so a Low-IL
# writer is blocked by the no-write-up policy on every child path.
# This script sets the same label on directories only (no file contents touched, no owner,
# no DACL change). Reversible with: icacls <path> /setintegritylevel Medium
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Path,
  [switch]$IncludeFiles,
  [int]$Limit = 20000
)
$ErrorActionPreference = 'Stop'
$full = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')

function Set-LowLabel([string]$p) {
  $out = & icacls $p /setintegritylevel '(OI)(CI)L' 2>&1
  if ($LASTEXITCODE -ne 0) { throw "icacls failed on $p : $($out -join ' ')" }
}

# Root first, then a bounded breadth-first walk over directories.
Set-LowLabel $full
Write-Output "LABELED $full"

$q = [System.Collections.Generic.Queue[string]]::new()
try { foreach ($d in [System.IO.Directory]::EnumerateDirectories($full)) { $q.Enqueue($d) } } catch { }
$n = 0
while ($q.Count -gt 0 -and $n -lt $Limit) {
  $cur = $q.Dequeue()
  $item = Get-Item -LiteralPath $cur -Force -ErrorAction SilentlyContinue
  if ($null -eq $item) { continue }
  if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
  try { Set-LowLabel $cur; $n++ } catch { Write-Output "  SKIP $cur :: $($_.Exception.Message)"; continue }
  try { foreach ($d in [System.IO.Directory]::EnumerateDirectories($cur)) { $q.Enqueue($d) } } catch { }
}
Write-Output "LABELED_DIRS $($n + 1)"

if ($IncludeFiles) {
  $fn = 0
  foreach ($f in [System.IO.Directory]::EnumerateFiles($full, '*', [System.IO.SearchOption]::AllDirectories)) {
    try { & icacls $f /setintegritylevel L > $null 2>&1; if ($LASTEXITCODE -eq 0) { $fn++ } } catch { }
    if ($fn -ge $Limit) { break }
  }
  Write-Output "LABELED_FILES $fn"
}
Write-Output "DONE"
