# Checks that the whole system works, end to end.
# Run this after: docker compose up --build -d
#
#   powershell -ExecutionPolicy Bypass -File scripts\check-system.ps1
#
# It sends readings through the gateway and checks what comes back, so it proves
# all four services are running and talking to each other.

$GW = if ($env:GATEWAY_URL) { $env:GATEWAY_URL } else { "http://localhost:8080" }
$zone = "test-" + [int][double]::Parse((Get-Date -UFormat %s))
$failed = 0

function Check($name, $got, $want) {
  if ("$got" -eq "$want") { Write-Host "PASS  $name" -ForegroundColor Green }
  else { Write-Host "FAIL  $name (got '$got', expected '$want')" -ForegroundColor Red
         $script:failed = 1 }
}
function Send($body) {
  Invoke-RestMethod -Method Post -Uri "$GW/api/readings" -ContentType "application/json" `
    -Body ($body | ConvertTo-Json -Depth 5)
}

# One timestamp, worked out once. Test 3 sends the same reading again and expects
# it to be rejected as a duplicate, which only works if both use the same time.
$stamp = (Get-Date).ToUniversalTime().AddHours(-2).ToString("yyyy-MM-ddTHH:mm:ssZ")
$healthy = @{ readings = @(@{
  zone_id = $zone; recorded_at = $stamp
  temperature = 27.6; soil_humidity = 57.0; soil_moisture = 44.9
  air_humidity = 59.6; ph = 6.62; soil_ec = 0.94; pressure = 101.14; rainfall = 88.1 })}

Write-Host "`n1. all four services are running"
Check "health check" (Invoke-RestMethod "$GW/health/all").status "ok"

Write-Host "`n2. a healthy reading is classified as Optimal"
Check "optimal" (Send $healthy).results[0].label "Optimal"

Write-Host "`n3. sending the same reading twice does not store it twice"
Check "no duplicate" (Send $healthy).results[0].duplicate "True"

Write-Host "`n4. an impossible reading is rejected"
try {
  $bad = @{ readings = @(@{ zone_id = $zone; recorded_at = $stamp; soil_moisture = 999 })}
  Invoke-RestMethod -Method Post -Uri "$GW/api/readings" -ContentType "application/json" `
    -Body ($bad | ConvertTo-Json -Depth 5) | Out-Null
  Check "rejected" "accepted" "422"
} catch { Check "rejected" $_.Exception.Response.StatusCode.value__ 422 }

Write-Host "`n5. the trained model is loaded"
$model = Invoke-RestMethod "$GW/api/model"
Check "model loaded" ($model.version -ne $null) "True"
Write-Host "      $($model.version), finds $([math]::Round($model.critical_recall*100))% of Critical readings"

Write-Host "`n6. lights off at night is normal, not a fault"
$night = @{ temperature=27.6; soil_humidity=57; soil_moisture=44.9; air_humidity=59.6
            ph=6.62; soil_ec=0.94; pressure=101.14; rainfall=88.1
            nitrogen=54.3; phosphorus=38.3; potassium=52.0
            light_intensity=40; hour=2 }
Check "night" (Invoke-RestMethod -Method Post -Uri "$GW/api/predict" `
  -ContentType "application/json" -Body ($night | ConvertTo-Json)).label "Optimal"

Write-Host "`n7. the same darkness at midday is a fault"
$day = $night.Clone(); $day.hour = 12
$result = Invoke-RestMethod -Method Post -Uri "$GW/api/predict" `
  -ContentType "application/json" -Body ($day | ConvertTo-Json)
Check "midday" $result.label "Critical"
Write-Host "      cause: $($result.probable_cause)"

Write-Host ""
if ($failed -eq 0) { Write-Host "All checks passed." -ForegroundColor Green }
else { Write-Host "Some checks failed." -ForegroundColor Red }
exit $failed
