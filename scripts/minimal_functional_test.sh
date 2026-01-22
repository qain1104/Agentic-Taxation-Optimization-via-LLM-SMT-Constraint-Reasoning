#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-http://localhost:32770}"
OUTDIR="${OUTDIR:-ae_outputs}"

mkdir -p "${OUTDIR}"

echo "[1/5] Checking service liveness at ${HOST}/openapi.json ..."
until curl -sSf "${HOST}/openapi.json" >/dev/null; do
  sleep 1
done
echo "OK"

echo "[2/5] Saving OpenAPI spec to ${OUTDIR}/openapi.json"
curl -sSf "${HOST}/openapi.json" > "${OUTDIR}/openapi.json"

cat <<'EOF'

[3/5] Manual UI Minimal Functional Test (Income tax)

1) Ensure the service is running (docker compose up).
2) Open the UI: http://localhost:32770
3) Tell the system which tax to compute (or click the right-side button):
   - 我要算所得稅
4) Phase 1 input:
   - 我是一名公司高管，年收400萬、太太在家當貴婦，我們有兩個小孩。
   - 下一步
5) Phase 2 input:
   - 我想嘗試調整一些配置，所得總和400萬。
   - 下一步
6) Wait for Phase 3 result (report/summary).

When you see the result in the UI, come back here and press ENTER.

EOF

read -r

echo "[4/5] Capturing docker logs tail (best effort) -> ${OUTDIR}/docker_logs_tail.txt"
if [ -d "source_code" ] && [ -f "source_code/docker-compose.yml" ]; then
  (cd source_code && docker compose logs --tail=200) > "${OUTDIR}/docker_logs_tail.txt" || true
else
  echo "WARN: cannot find ./source_code/docker-compose.yml; skipping docker logs capture." > "${OUTDIR}/docker_logs_tail.txt"
fi

echo "[5/5] Capturing modified files under source_code/ (best effort)"
if [ -d "source_code" ]; then
  find source_code -maxdepth 5 -type f -mmin -10 | sort > "${OUTDIR}/recent_files.txt" || true
  find source_code -maxdepth 5 -type f | grep -E "reports|output|outputs|logs|artifacts" | sort | head -n 200 > "${OUTDIR}/common_outputs.txt" || true
else
  echo "WARN: cannot find ./source_code; run this script from the artifact root." > "${OUTDIR}/recent_files.txt"
  echo "WARN: cannot find ./source_code; run this script from the artifact root." > "${OUTDIR}/common_outputs.txt"
fi

echo
echo "Done."
echo "Check ${OUTDIR}/ for: openapi.json, docker_logs_tail.txt, recent_files.txt, common_outputs.txt"
