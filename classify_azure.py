import os, json, glob
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION","2024-10-21"),
)

DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

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

def _bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.lower().strip() in ("true","1","yes")
    return False

def _sanitize(d):
    types = [t for t in d.get("incident_types", []) if t in ALLOWED]
    out = {
        "is_incident": _bool(d.get("is_incident", False)),
        "incident_types": types,
        "near_miss": _bool(d.get("near_miss", False)),
        "confidence": float(d.get("confidence", 0.5)),
        "rationale": str(d.get("rationale", "")).strip()[:60]
    }
    if not out["is_incident"]:
        out["incident_types"] = []
        out["near_miss"] = False
        out["confidence"] = min(out["confidence"], 0.5)
    return out

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def classify_text(text: str) -> dict:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        temperature=0.0,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":f"Text:\n{text}"}
        ],
        timeout=30,
    )
    data = json.loads(resp.choices[0].message.content)
    return _sanitize(data)

def run(in_dir="data/normalized", out_dir="data/classified"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{in_dir}/*.json")
    total, incidents = 0, 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        title = doc.get("title","")
        content = (doc.get("content_text","") or "")[:1000]
        text = f"{title}\n{content}"
        try:
            res = classify_text(text)
        except Exception as e:
            res = {"is_incident": False, "incident_types": [], "near_miss": False, "confidence": 0.0, "rationale": "error"}
        out = {
            "doc_id": doc["doc_id"],
            "url": doc.get("url",""),
            "title": title,
            "published_at": doc.get("published_at",""),
            **res
        }
        with open(Path(out_dir)/f"{doc['doc_id']}.classify.json","w",encoding="utf-8") as oh:
            json.dump(out, oh, ensure_ascii=False, indent=2)
        total += 1
        incidents += int(out["is_incident"])
    print(f"Classified {total} docs → {out_dir} | incidents: {incidents}")

if __name__ == "__main__":
    run()

