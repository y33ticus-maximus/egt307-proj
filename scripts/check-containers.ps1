# Checks every container in the running Docker Compose application.
# Run this after: docker compose up --build -d
#
#   powershell -ExecutionPolicy Bypass -File scripts\check-containers.ps1

$passed = 0
$failed = 0

function Show-Pass($message) {
  Write-Host "PASS  $message" -ForegroundColor Green
  $script:passed++
}

function Show-Fail($message) {
  Write-Host "FAIL  $message" -ForegroundColor Red
  $script:failed++
}

# PostgreSQL does not serve HTTP, so use its own readiness command.
docker compose exec -T postgres pg_isready -U farm -d farmdb 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Show-Pass "postgres is ready" }
else { Show-Fail "postgres is not ready" }

# Each Python container checks its own local endpoint. Ingestion and dashboard
# use /ready because this also verifies their PostgreSQL connection.
$checks = @{
  "gateway"   = "/health"
  "ingestion" = "/ready"
  "inference" = "/ready"
  "dashboard" = "/ready"
  "simulator" = "/ready"
}

foreach ($name in $checks.Keys) {
  $path = $checks[$name]
  $code = "import json,urllib.request; d=json.load(urllib.request.urlopen('http://localhost:8000$path',timeout=5)); assert d.get('status') in ('ok','ready') and d.get('model',True) is not False, d"
  docker compose exec -T $name python -c $code 2>&1 | Out-Null

  if ($LASTEXITCODE -eq 0) { Show-Pass "$name is ready" }
  else { Show-Fail "$name is not ready or not running" }
}

Write-Host "`n$passed passed, $failed failed"
if ($failed -eq 0) {
  Write-Host "All Docker containers are working correctly." -ForegroundColor Green
  exit 0
}

Write-Host "Run: docker compose logs --tail 30" -ForegroundColor Yellow
exit 1
