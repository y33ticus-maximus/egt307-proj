# Checks the Kubernetes deployment, the way check-containers.ps1 checks Compose.
# Run after: kubectl apply -f k8s\
#
#   powershell -ExecutionPolicy Bypass -File scripts\check-k8s.ps1
#
# It proves the four things the deployment is meant to demonstrate: every pod
# starts, the scaled service really runs several replicas, the cluster replaces
# a pod you delete, and the system still answers from outside the cluster.

$NS = "smart-farm"
$failed = 0

function Check($name, $got, $want) {
  if ("$got" -eq "$want") { Write-Host "PASS  $name" -ForegroundColor Green }
  else { Write-Host "FAIL  $name (got '$got', expected '$want')" -ForegroundColor Red
         $script:failed++ }
}

Write-Host "`n1. every pod reaches Ready"
kubectl -n $NS wait --for=condition=Ready pod --all --timeout=300s
Check "all pods ready" $LASTEXITCODE 0
kubectl -n $NS get pods

Write-Host "`n2. what is deployed"
kubectl -n $NS get deploy,svc,hpa,pvc

Write-Host "`n3. inference is the scaled service"
$ready = kubectl -n $NS get deploy inference -o jsonpath='{.status.readyReplicas}'
Check "three inference replicas" $ready 3

Write-Host "`n4. the autoscaler is configured"
$min = kubectl -n $NS get hpa inference -o jsonpath='{.spec.minReplicas}'
$max = kubectl -n $NS get hpa inference -o jsonpath='{.spec.maxReplicas}'
Check "hpa range 2 to 6" "$min-$max" "2-6"
Write-Host "      (TARGETS showing <unknown> means metrics-server is not enabled)"

Write-Host "`n5. storage survives the database pod being deleted"
kubectl -n $NS delete pod -l app=postgres --wait=$false | Out-Null
Start-Sleep 5
kubectl -n $NS rollout status deployment/postgres --timeout=180s | Out-Null
$bound = kubectl -n $NS get pvc postgres-pvc -o jsonpath='{.status.phase}'
Check "claim still bound" $bound "Bound"

Write-Host "`n6. killing an inference pod does not take the service down"
$pod = kubectl -n $NS get pods -l app=inference -o jsonpath='{.items[0].metadata.name}'
Write-Host "      deleting $pod"
kubectl -n $NS delete pod $pod --wait=$false | Out-Null
Start-Sleep 5
kubectl -n $NS rollout status deployment/inference --timeout=180s | Out-Null
$ready = kubectl -n $NS get deploy inference -o jsonpath='{.status.readyReplicas}'
Check "replaced automatically" $ready 3

Write-Host "`n7. the system answers from outside the cluster"
$pf = Start-Process kubectl -ArgumentList "-n",$NS,"port-forward","svc/gateway","8080:8000" `
      -PassThru -WindowStyle Hidden
Start-Sleep 6
try {
  Check "health check" (Invoke-RestMethod "http://localhost:8080/health/all").status "ok"
  $model = Invoke-RestMethod "http://localhost:8080/api/model"
  Check "model loaded" ($null -ne $model.version) "True"
  Write-Host "      $($model.version)"
} catch {
  Check "health check" $_.Exception.Message "ok"
} finally {
  Stop-Process -Id $pf.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failed -eq 0) { Write-Host "All Kubernetes checks passed." -ForegroundColor Green }
else { Write-Host "$failed check(s) failed." -ForegroundColor Red }
exit $failed
