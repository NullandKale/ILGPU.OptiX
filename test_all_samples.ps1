# Test all 15 ILGPU.OptiX samples
# Usage: powershell -ExecutionPolicy Bypass -File test_all_samples.ps1

$ErrorActionPreference = "Continue"
$results = @()
$wpfSamples = @(10, 11, 12, 13)

Write-Host "=== ILGPU.OptiX Sample Test Suite ===" -ForegroundColor Cyan
Write-Host "Testing samples 1-15..."
Write-Host ""

for ($num = 1; $num -le 15; $num++) {
    $sampleName = "Sample$('{0:D2}' -f $num)"

    if ($wpfSamples -contains $num) {
        # WPF samples need manual verification
        Write-Host "$sampleName (WPF Interactive):" -ForegroundColor Cyan
        Write-Host "  A window should appear with a ray-traced scene."
        Write-Host "  Check:"
        Write-Host "    - Window opens and displays correctly"
        Write-Host "    - No crashes or visual artifacts"
        Write-Host "    - Window is responsive"
        Write-Host "  Launching now... (close window when done testing)"
        Write-Host ""

        try {
            $exe = Get-Item "Samples\$sampleName\bin\Release\*\$sampleName.exe" | Select-Object -First 1
            if ($exe) {
                & $exe.FullName 2>&1 | ForEach-Object { Write-Host "    $_" }
                $results += @{Sample=$num; Name=$sampleName; Status="Manual Verification"; ExitCode=$LASTEXITCODE}
                Write-Host "✓ $sampleName: Window closed (exit code: $LASTEXITCODE)" -ForegroundColor Green
            } else {
                $results += @{Sample=$num; Name=$sampleName; Status="FAIL"; Reason="Executable not found"}
                Write-Host "✗ $sampleName: Executable not found" -ForegroundColor Red
            }
        } catch {
            $results += @{Sample=$num; Name=$sampleName; Status="ERROR"; Error=$_.Exception.Message}
            Write-Host "✗ $sampleName: ERROR - $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        # Console samples
        Write-Host "$sampleName (Console):" -ForegroundColor Cyan

        try {
            $exe = Get-Item "Samples\$sampleName\bin\Release\*\$sampleName.exe" | Select-Object -First 1
            if ($exe) {
                Write-Host "  Running: $($exe.FullName)"
                $output = & $exe.FullName 2>&1
                Write-Host $output | ForEach-Object { Write-Host "    $_" }

                if ($LASTEXITCODE -eq 0) {
                    $results += @{Sample=$num; Name=$sampleName; Status="PASS"; ExitCode=0}
                    Write-Host "✓ $sampleName: PASS" -ForegroundColor Green
                } else {
                    $results += @{Sample=$num; Name=$sampleName; Status="FAIL"; ExitCode=$LASTEXITCODE}
                    Write-Host "✗ $sampleName: FAIL (exit code: $LASTEXITCODE)" -ForegroundColor Red
                }
            } else {
                $results += @{Sample=$num; Name=$sampleName; Status="FAIL"; Reason="Executable not found"}
                Write-Host "✗ $sampleName: Executable not found" -ForegroundColor Red
            }
        } catch {
            $results += @{Sample=$num; Name=$sampleName; Status="ERROR"; Error=$_.Exception.Message}
            Write-Host "✗ $sampleName: ERROR - $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    Write-Host ""
}

# Print summary
Write-Host "=== TEST SUMMARY ===" -ForegroundColor Yellow
$passed = ($results | Where-Object { $_.Status -eq "PASS" }).Count
$failed = ($results | Where-Object { $_.Status -eq "FAIL" }).Count
$errors = ($results | Where-Object { $_.Status -eq "ERROR" }).Count
$manual = ($results | Where-Object { $_.Status -eq "Manual Verification" }).Count

foreach ($result in $results) {
    switch ($result.Status) {
        "PASS" { $color = "Green"; $icon = "✓" }
        "FAIL" { $color = "Red"; $icon = "✗" }
        "ERROR" { $color = "Red"; $icon = "✗" }
        "Manual Verification" { $color = "Cyan"; $icon = "?" }
        default { $color = "White"; $icon = "?" }
    }

    $reason = ""
    if ($result.Reason) { $reason = " ($($result.Reason))" }
    if ($result.Error) { $reason = " - $($result.Error)" }

    Write-Host "$icon $($result.Name): $($result.Status)$reason" -ForegroundColor $color
}

Write-Host ""
Write-Host "Summary: $passed passed, $failed failed, $errors errors, $manual manual tests" -ForegroundColor Yellow
Write-Host ""

if ($failed -eq 0 -and $errors -eq 0) {
    Write-Host "All tests completed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some tests failed. See details above." -ForegroundColor Red
    exit 1
}
