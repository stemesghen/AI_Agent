import json, glob, re
from pathlib import Path
import dateparser
import spacy

# folders
IN_DIR = "data/classified"
NORM_DIR = "data/normalized"
OUT_DIR = "data/extracted"

# load spaCy model once
nlp = spacy.load("en_core_web_sm")

# simple regex helpers
RE_IMO = re.compile(r"\bIMO\s*([0-9]{7})\b", re.I)
RE_VESSEL_HINT = re.compile(r"\b(MV|M\/V|MS|M\.S\.|SS|M\/T|MT)\s+[A-Z0-9\- ]{3,}\b")
PORT_WORDS = {"port", "harbor", "harbour", "terminal", "anchorage", "bay"}


def load_doc(norm_path):
    with open(norm_path, "r", encoding="utf-8") as f:
        return json.load(f)


NON_VESSEL_TERMS = {"tug", "tugs", "pilot", "pilots", "harbor", "harbour", "port", "authority"}
RE_PORT_NAME = re.compile(r"\bPort\s+[A-Z][A-Za-z.\- ]{2,}\b")  # e.g., Port Hedland, Port Said

def normalize_ship_candidate(name: str) -> str | None:
    if not name: return None
    tokens = [t.strip(" .,-").lower() for t in name.split()]
    if any(t in NON_VESSEL_TERMS for t in tokens):
        return None
    # Heuristic: vessel names are Title Case or ALLCAPS, 2–5 tokens
    if not any(w[0].isupper() for w in name.split()):
        return None
    return " ".join(name.split())


def choose_date(text):
    dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "past"})
    return dt.date().isoformat() if dt else None


def extract_entities(title, text):
    doc = nlp(f"{title}\n{text[:3000]}")
    vessel = imo = port = date_iso = None

    # 1) IMO
    if m := RE_IMO.search(text):
        imo = m.group(1)

    # 2) PORT: explicit "Port X" beats everything
    if m := RE_PORT_NAME.search(title):
        port = m.group(0)
    if not port:
        if m := RE_PORT_NAME.search(text):
            port = m.group(0)

    # fallback: GPE/LOC with port cue nearby
    if not port:
        gpes = [ent.text for ent in doc.ents if ent.label_ in ("GPE","LOC")]
        for g in gpes:
            if re.search(rf"\b{re.escape(g)}\b.*\b({'|'.join(PORT_WORDS)})\b", text, re.I) or "port" in title.lower():
                port = g
                break
        if not port and gpes:
            port = gpes[0]

    # 3) VESSEL:
    # 3a) Prefix pattern (MV/MS/SS/MT ...)
    v = None
    if m := RE_VESSEL_HINT.search(text.upper()):
        v = m.group(0).title()

    # 3b) If still none, use proper-noun runs but drop obvious non-vessels
    if not v:
        proper_runs, current = [], []
        for t in doc:
            if t.pos_ == "PROPN" and t.text[0].isupper():
                current.append(t.text)
            else:
                if len(current) >= 2:
                    proper_runs.append(" ".join(current))
                current = []
        if len(current) >= 2:
            proper_runs.append(" ".join(current))
        for cand in proper_runs[:6]:
            nc = normalize_ship_candidate(cand)
            if nc:
                v = nc
                break

    vessel = v

    # 4) DATE: prefer explicit in text; fallback to None (we’ll fill from published_at outside)
    date_iso = choose_date(text) or choose_date(title)

    return {"vessel": vessel, "imo": imo, "port": port, "date": date_iso}
def run():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    cls_files = glob.glob(f"{IN_DIR}/*.classify.json")
    count = 0
    for cf in cls_files:
        with open(cf, "r", encoding="utf-8") as f:
            cls = json.load(f)
        if not cls.get("is_incident"):
            continue

        norm_path = Path(NORM_DIR) / Path(cf).name.replace(".classify.json", ".json")
        if not norm_path.exists():
            continue
        norm = load_doc(norm_path)

        title = norm.get("title", "")
        text = norm.get("content_text", "")
        ents = extract_entities(title, text)

        date_final = ents["date"] or (norm.get("published_at", "")[:10] or None)
        out = {
            "doc_id": norm["doc_id"],
            "vessel": ents["vessel"],
            "imo": ents["imo"],
            "port": ents["port"],
            "date": date_final
        }

        out_path = Path(OUT_DIR) / f"{norm['doc_id']}.extract.json"
        with open(out_path, "w", encoding="utf-8") as oh:
            json.dump(out, oh, ensure_ascii=False, indent=2)
        count += 1

    print(f"Extracted entities for {count} incident docs → {OUT_DIR}")


if __name__ == "__main__":
    run()

