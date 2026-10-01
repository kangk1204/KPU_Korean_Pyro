#!/usr/bin/env python3
"""KBDS beta access evidence helper.

This script is a narrow evidence harness for the KBDS CRC cohorts under
public-access review.

Current state:
- Public project pages, file-info endpoints, and RAON config/handler are reachable.
- The tested public HTML5 download returned HTTP 401; the cause is unresolved.
- No processed_beta bytes have been retrieved yet.

The script intentionally avoids panel scoring or outcome computation.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "review" / "kbds_access" / "access_report.md"
INVENTORY = ROOT / "data" / "raw" / "kbds" / "kbds_public_access_inventory.json"


def main() -> int:
    print(f"report={REPORT}")
    print(f"inventory={INVENTORY}")
    if REPORT.exists():
        print("report_status=present")
    else:
        print("report_status=missing")
    if INVENTORY.exists():
        data = json.loads(INVENTORY.read_text())
        print(f"request_count={len(data.get('requests', []))}")
        print(f"project_count={len(data.get('projects', []))}")
    else:
        print("inventory_status=missing")
    print("processed_beta_retrieved=false")
    print("download_route=http401_invalid_request_cause_unresolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
