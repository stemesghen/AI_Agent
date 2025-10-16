import os, json
from .base import Classifier

# Optional: pip install openai==1.* tenacity
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None  # so imports don't break when you don't have it yet

SYSTEM_PROMPT = """You are a cautious maritime risk analyst. Decide if the text describes a real maritime incident.
Incident types ONLY: "grounding","collision","fire","piracy","weather","port_closure","strike","spill".
Rules:
- Policy/sanctions/market news ≠ incident.
- Forecasts/rumors without an event ≠ incident.
- “Prevented/averted” counts as incident, set near_miss=true.
Return STRICT JSON only:
{ "is_incident": <bool>, "incident_types": <array>, "near_miss": <bool>, "confidence": <0..1>, "rationale": "<≤12 words>" }
"""

ALLOWED = {"grounding","collision","fire","piracy","weather","port_closure","strike","spill"}

def _coerce_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() in ("true","1","yes")
    return False

def _sanitize(d):
    types = [t for t in d.get("incident_types", []) if t in ALLOWED]
    out = {
        "is_incident": _coerce_bool(d.get("is_incident", False)),
        "incident_types": types,
        "near_miss": _coerce_bool(d.get("near_miss", False)),
        "confidence": float(d.get("confidence", 0.5)),
        "rationale": str(d.get("rationale","")).strip()[:60],
    }
    if not out["is_incident"]:
        out["incident_types"] = []
        out["near_miss"] = False
        out["confidence"] = min(out["confidence"], 0.5)
    return out

class AzureOpenAIClassifier(Classifier):
    def __init__(self):
        if AzureOpenAI is None:
            raise RuntimeError("openai client not installed; pip install openai")
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION","2024-10-21"),
        )
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    def classify(self, text: str):
        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0.0,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":f"Text:\n{text or ''}"}
            ],
            timeout=30,
        )
        data = json.loads(resp.choices[0].message.content)
        return _sanitize(data)

