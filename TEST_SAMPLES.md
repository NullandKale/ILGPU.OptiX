# Sample Testing Guide

This document provides instructions for testing all 15 ILGPU.OptiX samples.

## Sample Categories

### Console Samples (Samples 1-9, 14-15)
These are command-line applications that should run and output results to the console.

### WPF Samples (Samples 10-13)  
These are Windows WPF applications with GUI windows that display rendering results.

## Sample Descriptions

**Sample01**: Basic CUDA + OptiX initialization (no rendering)
- Expected: Prints "Initializing CUDA + OptiX..." and exits successfully

**Sample02**: Simple ray tracing with custom ray generation
- Expected: Prints output info and exits

**Sample03-09**: Progressive ray tracing samples with increasing complexity
- Sample03: Basic triangle mesh rendering
- Sample04: Triangle mesh with perspective camera
- Sample05: Textured meshes with lighting
- Sample06: Physically-based rendering materials
- Sample07: Model loading
- Sample08: Multi-material scenes
- Sample09: Advanced lighting
- Expected: WPF windows showing rendered scenes

**Sample10-13**: Advanced rendering with interactive controls
- Expected: WPF windows with interactive ray-traced scenes

**Sample14**: OptiX + OpenGL interop (should display a rendered scene)
- Expected: OpenGL window showing ray-traced rendering

**Sample15**: PBR path tracer with advanced features
- Expected: Console app that generates path-traced imagery

## Testing Procedure

### For Console Samples (1-2, 14-15):
1. Open PowerShell in the Samples directory
2. Run: `.\SampleN\bin\Release\netX.Y\SampleN.exe`
3. Check for:
   - No crashes or exceptions
   - Appropriate console output
   - Clean exit

### For WPF Samples (3-13):
1. Run the executable: `.\SampleN\bin\Release\netX.Y\SampleN.exe`
2. A window should appear showing a ray-traced scene
3. Check for:
   - Window appears and stays responsive
   - Scene displays correctly
   - No visual glitches or artifacts
   - Can close window without crashing

## Quick Test Script

Run all samples sequentially (save as `test_all.ps1`):

```powershell
$samples = @(1..15)
$results = @()

foreach ($num in $samples) {
    $pad = [string]$num
    if ($num -lt 10) { $pad = "0$num" }
    
    $wpfSamples = @(10, 11, 12, 13)
    if ($wpfSamples -contains $num) {
        Write-Host "Sample$pad (WPF): Please verify window opens correctly" -ForegroundColor Cyan
        Write-Host "  Press Enter when done testing..."
        Read-Host
        $results += @{Sample=$num; Status="Manual verification"}
    } else {
        Write-Host "Testing Sample$pad..." -ForegroundColor Green
        
        $exe = Get-Item "Samples\Sample$pad\bin\Release\*\Sample$pad.exe" | Select-Object -First 1
        if ($exe) {
            try {
                & $exe.FullName 2>&1 | Tee-Object -Variable output | Write-Host
                if ($LASTEXITCODE -eq 0) {
                    $results += @{Sample=$num; Status="PASS"}
                    Write-Host "✓ Sample$pad: PASS" -ForegroundColor Green
                } else {
                    $results += @{Sample=$num; Status="FAIL"; ExitCode=$LASTEXITCODE}
                    Write-Host "✗ Sample$pad: FAIL (exit code: $LASTEXITCODE)" -ForegroundColor Red
                }
            } catch {
                $results += @{Sample=$num; Status="ERROR"; Error=$_.Exception.Message}
                Write-Host "✗ Sample$pad: ERROR - $_" -ForegroundColor Red
            }
        } else {
            Write-Host "✗ Sample$pad: Executable not found" -ForegroundColor Red
        }
    }
    Write-Host ""
}

Write-Host "=== TEST SUMMARY ===" -ForegroundColor Yellow
$results | ForEach-Object { Write-Host "Sample $('{0:D2}' -f $_.Sample): $($_.Status)" }
```

Save this file and run with: `powershell -ExecutionPolicy Bypass -File test_all.ps1`
