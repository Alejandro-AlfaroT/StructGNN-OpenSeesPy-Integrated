# Independent metadata inventory when the selected Python path is unset.
# Does not replace preflight.py or import generator code.
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$profilePath = Join-Path $workspace 'config/desktop.local.json'
$started = [DateTime]::UtcNow
$destination = Join-Path $workspace ('outputs/' + $started.ToString('yyyyMMddTHHmmssZ') + '_desktop_setup_' + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $destination | Out-Null
$evidence = [System.Collections.Generic.List[object]]::new()
$errorsFound = [System.Collections.Generic.List[object]]::new()
function Read-EvidenceJson([string]$path) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    try {
        $before = Get-Item -LiteralPath $path
        $size = $before.Length
        $ticks = $before.LastWriteTimeUtc.Ticks
        if ($size -gt 33554432) { throw 'Exceeds 32 MiB metadata bound' }
        $raw = [IO.File]::ReadAllBytes($path)
        $after = Get-Item -LiteralPath $path
        $changed = $size -ne $after.Length -or $ticks -ne $after.LastWriteTimeUtc.Ticks
        $sha = [Security.Cryptography.SHA256]::Create()
        $hash = [BitConverter]::ToString($sha.ComputeHash($raw)).Replace('-','').ToLower()
        $sha.Dispose()
        $evidence.Add([pscustomobject]@{path=$path; bytes=$size; mtime_ticks=$ticks; sha256=$hash; changed_during_read=$changed})
        if ($changed) { throw 'Changed during read' }
        return [Text.Encoding]::UTF8.GetString($raw).TrimStart([char]0xfeff) | ConvertFrom-Json
    } catch { $errorsFound.Add([pscustomobject]@{path=$path; error=$_.Exception.Message}); return $null }
}
$profile = Read-EvidenceJson $profilePath
$base = Join-Path $profile.project_root $profile.dataset_base
$rows = @()
$plans = @()
$samples = @()
foreach ($root in (Get-ChildItem -LiteralPath $base -Directory | Sort-Object Name)) {
    $planPath = Join-Path $root.FullName 'parameter_plan.json'
    if (!(Test-Path -LiteralPath $planPath)) { continue }
    $plan = Read-EvidenceJson $planPath
    $manifest = @(Read-EvidenceJson (Join-Path $root.FullName 'parameterized_manifest.json'))
    $state = Read-EvidenceJson (Join-Path $root.FullName 'generation_state.json')
    $metadata = [ordered]@{}
    foreach ($prop in $plan.PSObject.Properties) { if ($prop.Name -ne 'cases') { $metadata[$prop.Name] = $prop.Value } }
    $plans += [pscustomobject]@{root=$root.FullName; metadata=$metadata; first_case=@($plan.cases | Select-Object -First 1); comparison_admitted=$false}
    $caseRoot = Join-Path $root.FullName 'cases'
    $cases = @(if (Test-Path -LiteralPath $caseRoot) { Get-ChildItem -LiteralPath $caseRoot -Directory -Filter 'case_*' | Sort-Object Name })
    $configured = @($profile.datasets | Where-Object path -EQ $root.Name).Count -gt 0
    $nthaCount = 0; $npzCount = 0; $metaCount = 0
    if ($configured) {
        foreach ($case in $cases) {
            $ntha = Join-Path $case.FullName 'ntha'
            if (Test-Path -LiteralPath $ntha) { $nthaCount += @(Get-ChildItem -LiteralPath $ntha -Directory).Count }
            $ds = Join-Path $case.FullName 'dataset'
            if (Test-Path -LiteralPath $ds) {
                foreach ($run in (Get-ChildItem -LiteralPath $ds -Directory)) {
                    if (Test-Path -LiteralPath (Join-Path $run.FullName 'hybrid_sample.npz') -PathType Leaf) { $npzCount++ }
                    $mp = Join-Path $run.FullName 'hybrid_metadata.json'
                    if (Test-Path -LiteralPath $mp -PathType Leaf) {
                        $metaCount++
                        if ($case.Name -in @($cases | Select-Object -First 3 -ExpandProperty Name)) {
                            $m = Read-EvidenceJson $mp
                            $samples += [pscustomobject]@{root=$root.Name; case_id=$case.Name; run_name=$run.Name; source=$mp; schema_version=$m.schema_version; npz_present=(Test-Path -LiteralPath (Join-Path $run.FullName 'hybrid_sample.npz'))}
                        }
                    }
                }
            }
        }
    }
    $claims = [ordered]@{}
    foreach ($g in ($manifest | Where-Object { $null -ne $_ } | Group-Object status)) { $claims[$g.Name] = $g.Count }
    $plannedIds = @($plan.cases | ForEach-Object case_id)
    $observedIds = @($cases.Name)
    $rows += [pscustomobject]@{root=$root.Name; path=$root.FullName; configured=$configured; plan_version=$plan.version; plan_cases=@($plan.cases).Count; case_directories=$cases.Count; planned_absent=@($plannedIds | Where-Object {$_ -notin $observedIds}).Count; manifest_rows=@($manifest | Where-Object {$null -ne $_}).Count; manifest_status_claims=($claims | ConvertTo-Json -Compress); ntha_run_directories=$(if($configured){$nthaCount}else{$null}); metadata_files=$(if($configured){$metaCount}else{$null}); npz_files=$(if($configured){$npzCount}else{$null}); state_claim=$state.status; state_updated_at=$state.updated_at}
}
$calibration = Read-EvidenceJson (Join-Path $base 'intensity_calibration_pilot30.json')
$changes = @()
foreach ($item in $evidence) {
    $now = Get-Item -LiteralPath $item.path
    if ($now.Length -ne $item.bytes -or $now.LastWriteTimeUtc.Ticks -ne $item.mtime_ticks) { $changes += $item.path }
}
$report = [ordered]@{scan_kind='independent_powershell_metadata_inventory_not_python_preflight'; started_at=$started.ToString('o'); finished_at=[DateTime]::UtcNow.ToString('o'); hostname=$env:COMPUTERNAME; profile=$profile; preflight_executed=$false; blocked_fields=@('python_executable'); dependencies='Unknown: configured interpreter is null'; roots=$rows; schema_samples=$samples; candidate_plans=$plans; calibration_artifact=$calibration; evidence=$evidence; changed_by_scan_end=$changes; errors=$errorsFound; sampling='Configured roots: all immediate case/ntha/dataset directories, metadata in first 3 lexical cases. Other plan roots: plan, manifest, state and case-directory inventory only. No NPZ contents or solver histories inspected.'}
$report | ConvertTo-Json -Depth 60 | Set-Content -LiteralPath (Join-Path $destination 'inventory.json') -Encoding utf8
$rows | Export-Csv -LiteralPath (Join-Path $destination 'coverage.csv') -NoTypeInformation
$samples | Export-Csv -LiteralPath (Join-Path $destination 'schema_samples.csv') -NoTypeInformation
Copy-Item -LiteralPath $profilePath -Destination (Join-Path $destination 'selected_profile.json')
$rows | Format-Table root,configured,plan_cases,case_directories,manifest_status_claims,npz_files,state_claim -AutoSize | Out-String -Width 220 | Write-Output
Write-Output $destination
