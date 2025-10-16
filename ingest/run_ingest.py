import csv, json, os, time
from pathlib import Path
import feedparser, requests, trafilatura
from dateutil import parser as dtp
from langdetect import detect as lang_detect
from utils import iso_now, make_doc_id, looks_maritime
from bs4 import BeautifulSoup



ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
NORM = DATA / "normalized"
CATALOG = DATA / "catalog.jsonl"
for p in (RAW, NORM): p.mkdir(parents=True, exist_ok=True)

def read_sources():
    with open(Path(__file__).parent / "sources.csv") as f:
        for row in csv.DictReader(f):
            yield row

def clean_html_to_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        text = trafilatura.extract(r.text) or ""
        return text.strip()
    except Exception:
        return ""

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def append_catalog(line_obj):
    with open(CATALOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

def already_seen(doc_id: str) -> bool:
    # cheap check: look for existing file
    return (NORM / f"{doc_id}.json").exists()

def norm_item(item, source_id, reliability, default_lang):
    url = item.get("link") or item.get("id")
    title = (item.get("title") or "").strip()
    if not url or not title:
        return None

    # full text
    text = clean_html_to_text(url)
    if not text:
        # fallback to feed summary
        text = (item.get("summary") or "").strip()

    # basic filters (skip if not maritime-ish)
    if not looks_maritime(f"{title}\n{text}"):
        return None

    # language guess
    try:
        lang = default_lang or lang_detect((title + " " + text)[:5000])
    except Exception:
        lang = default_lang or "en"

    # published time
    pub = item.get("published") or item.get("updated") or ""
    try:
        published_at = dtp.parse(pub).astimezone().astimezone(tz=None).astimezone().isoformat()
    except Exception:
        published_at = iso_now()

    doc_id = make_doc_id(title, url, text)
    doc = {
        "doc_id": doc_id,
        "source_id": source_id,
        "url": url,
        "title": title,
        "published_at": published_at,
        "fetched_at": iso_now(),
        "language": lang,
        "reliability": float(reliability or 0.7),
        "content_text": text
    }
    return doc


def list_page_links(base_url: str, item_selector: str, link_selector: str, max_pages: int = 1):
    headers = {"User-Agent": "Mozilla/5.0"}
    links = []
    for p in range(1, max_pages + 1):
        url = base_url if p == 1 else (base_url.rstrip("/") + f"/page/{p}/")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
        except Exception:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for card in soup.select(item_selector):
            a = card.select_one(link_selector)
            if a and a.get("href"):
                links.append((a.get("href"), a.get_text(strip=True)))
    # de-dup, preserve order
    seen=set(); out=[]
    for href,title in links:
        if href in seen: continue
        seen.add(href); out.append((href,title))
    return out

def ingest_once():
    new_count, dupes = 0, 0
    for src in read_sources():
        kind = src["kind"]
        if kind == "rss":
            feed = feedparser.parse(
                src["url"],
                request_headers={"User-Agent": "Mozilla/5.0"}
            )
            print(f"→ {src['source_id']} fetched {len(feed.entries)} entries (bozo={getattr(feed,'bozo',0)})")
            entries = [{"link": e.get("link") or e.get("id"),
                        "title": e.get("title",""),
                        "summary": e.get("summary","")} for e in feed.entries[:200]]
        elif kind == "html":
            item_sel = src.get("item_selector") or "article"
            link_sel = src.get("link_selector") or "a"
            max_pages = int(src.get("max_pages") or 1)
            pairs = list_page_links(src["url"], item_sel, link_sel, max_pages)
            print(f"→ {src['source_id']} scraped {len(pairs)} links from HTML")
            entries = [{"link": href, "title": title, "summary": ""} for href,title in pairs]
        else:
            continue

        for entry in entries:
            doc = norm_item(entry, src["source_id"], src["reliability"], src["lang"])
            if not doc: continue
            if already_seen(doc["doc_id"]):
                dupes += 1; continue
            save_json(doc, NORM / f"{doc['doc_id']}.json")
            append_catalog({
                "doc_id": doc["doc_id"], "url": doc["url"],
                "source_id": doc["source_id"], "title": doc["title"],
                "published_at": doc["published_at"]
            })
            new_count += 1
            time.sleep(0.2)  # polite
    print(f" new: {new_count} | dupes skipped: {dupes}")

if __name__ == "__main__":
    ingest_once()
