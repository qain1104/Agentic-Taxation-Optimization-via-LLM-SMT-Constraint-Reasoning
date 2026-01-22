#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-http://localhost:32770}"

echo "[1/4] Checking service liveness at ${HOST}/openapi.json ..."
until curl -sSf "${HOST}/openapi.json" >/dev/null; do
  sleep 1
done
echo "OK"

cat <<'EOF'

[2/4] Manual UI Minimal Functional Test (Income tax)

1) Open the UI: http://localhost:32770
2) Tell the system which tax to compute (or click the right-side button):
   - 我要算所得稅
3) Phase 1 input:
   - 我是一名公司高管，年收400萬、太太在家當貴婦，我們有兩個小孩。
   - 下一步
4) Phase 2 input:
   - 我想嘗試調整一些配置，所得總和400萬。
   - 下一步
5) Wait for Phase 3 result (report/summary).

When you see the result in the UI, come back here and press ENTER.

EOF

read -r

echo "[3/4] Recently modified files (last ~10 minutes) under source_code/"
if [ -d "source_code" ]; then
  find source_code -maxdepth 5 -type f -mmin -10 | sort || true
else
  echo "WARN: cannot find ./source_code; run this script from the artifact root."
fi

echo
echo "[4/4] Common output folders (logs/reports):"
if [ -d "source_code" ]; then
  find source_code -maxdepth 5 -type f | grep -E "reports|output|outputs|logs|artifacts" | sort | head -n 200 || true
fi

echo
echo "Done. If you see updated files under logs/ and reports/, the functional test succeeded."
