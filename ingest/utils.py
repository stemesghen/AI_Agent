#helpers

import hashlib, re
from datetime import datetime, timezone

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def make_doc_id(title: str, url: str, text: str) -> str:
    blob = (title or "")[:200] + (url or "") + (text or "")[:1000]
    return "sha256:" + hashlib.sha256(blob.encode("utf-8", "ignore")).hexdigest()

MARITIME_HINTS = re.compile(r'\b(vessel|ship|port|terminal|berth|IMO|container|grounding|piracy|hull|draft|anchorage)\b', re.I)

def looks_maritime(text: str) -> bool:
    return bool(MARITIME_HINTS.search(text or ""))

