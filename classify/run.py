
import os, glob, json, sys
from pathlib import Path
sys.path.append(os.path.dirname(__file__))


from .providers.mock_provider import MockClassifier

def get_provider():
    provider = os.environ.get("LLM_PROVIDER","mock").lower()
    if provider == "azure":
        from providers.azure_provider import AzureOpenAIClassifier
        return AzureOpenAIClassifier()
    return MockClassifier()

def run(in_dir="../data/normalized", out_dir="../data/classified"):
    # Allow running from repo root or from classify/ dir
    here = Path(__file__).parent
    in_path  = (here / in_dir).resolve()
    out_path = (here / out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    clf = get_provider()
    files = glob.glob(str(in_path / "*.json"))
    total, incidents = 0, 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        title = doc.get("title","")
        content = (doc.get("content_text","") or "")[:1000]
        text = f"{title}\n{content}"

        res = clf.classify(text)
        out = {
            "doc_id": doc["doc_id"],
            "url": doc.get("url",""),
            "title": title,
            "published_at": doc.get("published_at",""),
            **res
        }
        out_file = out_path / f"{doc['doc_id']}.classify.json"
        with open(out_file, "w", encoding="utf-8") as oh:
            json.dump(out, oh, ensure_ascii=False, indent=2)

        total += 1
        incidents += int(res["is_incident"])

    print(f"[LLM_PROVIDER={os.environ.get('LLM_PROVIDER','mock')}] "
          f"Classified {total} docs → {out_path} | incidents: {incidents}")

if __name__ == "__main__":
    run()

