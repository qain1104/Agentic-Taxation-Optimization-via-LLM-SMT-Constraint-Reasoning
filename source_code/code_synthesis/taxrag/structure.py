from __future__ import annotations
import datetime
from typing import Dict, Any, List

def build_all_laws_payload(laws: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "metadata": {
            "processed_date": datetime.date.today().isoformat(),
            "source": "PDF files",
            "language": "繁體中文",
        },
        "laws": laws,
    }
