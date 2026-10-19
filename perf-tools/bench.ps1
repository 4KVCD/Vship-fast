# Paired speed runs: alternates builds so drift affects them alike. Prints fps per run and the median per build.
param([string]$Tool = 'butterbench', [string[]]$Builds = @('C:\Code\Vship\build-local', 'C:\Code\Vship\wt-opt\build-clean'), [int]$Rounds = 3, [int]$Threads = 1)
$o = "$env:TEMP\vship-bench"
$cases = @(
    @{name = '1080p'; args = @(1920, 1080, "$o\tg1080_ref.yuv", "$o\tg1080_dis.yuv", 8, '-r', 40)},
    @{name = '4K'; args = @(3840, 2160, "$o\bee4k_ref.yuv", "$o\bee4k_dis.yuv", 6, '-r', 15)}
)
$results = @{}
foreach ($case in $cases) {
    for ($r = 0; $r -lt $Rounds; $r++) {
        foreach ($b in $Builds) {
            $line = (& "$b\$Tool.exe" @($case.args) -g $Threads -q | Select-String 'fps').Line
            $fps = [double](($line -split ', ')[-1] -replace ' fps', '')
            $key = "$($case.name) $b"
            if (-not $results.ContainsKey($key)) { $results[$key] = @() }
            $results[$key] += $fps
        }
    }
}
foreach ($k in $results.Keys | Sort-Object) {
    $v = $results[$k] | Sort-Object
    "{0,-55} median {1,7:N1} fps   runs {2}" -f $k, $v[[int][Math]::Floor($v.Count / 2)], (($results[$k] | ForEach-Object { $_.ToString('N1') }) -join ' ')
}
