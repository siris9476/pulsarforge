# CPU power (package) logger via LibreHardwareMonitorLib — requires
# ADMIN to read the RAPL counters (Start-Process powershell -Verb RunAs).
# On a power-limited machine, sustained throughput is governed by
# joules/token, and this logger is what made it possible to measure them.
# Usage:  powershell -ExecutionPolicy Bypass -File power_logger.ps1 <dll> [minutes] [out.csv]
param(
    [Parameter(Mandatory=$true)][string]$DllPath,
    [int]$Minutes = 45,
    [string]$OutFile = "power_log.csv"
)
Add-Type -Path $DllPath
$c = New-Object LibreHardwareMonitor.Hardware.Computer
$c.IsCpuEnabled = $true
$c.Open()
"timestamp,sensor,watt" | Out-File -FilePath $OutFile -Encoding utf8
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while ($sw.Elapsed.TotalMinutes -lt $Minutes) {
    foreach ($hw in $c.Hardware) {
        $hw.Update()
        foreach ($s in $hw.Sensors) {
            if ($s.SensorType -eq "Power" -and $s.Value -ne $null) {
                "{0},{1},{2:N2}" -f (Get-Date -Format "HH:mm:ss.fff"), $s.Name, $s.Value |
                    Out-File -FilePath $OutFile -Append -Encoding utf8
            }
        }
    }
    Start-Sleep -Milliseconds 500
}
$c.Close()
