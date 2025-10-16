import csv, feedparser
from pathlib import Path

SOURCES = Path(__file__).parent / "sources.csv"

with open(SOURCES) as f:
    for row in csv.DictReader(f):
        if row["kind"] != "rss":
            continue

        d = feedparser.parse(
            row["url"],
            request_headers={"User-Agent": "Mozilla/5.0"}
        )

        # Inspect results
        bozo = getattr(d, "bozo", 0)
        err = getattr(d, "bozo_exception", None)
        print(f"{row['source_id']:<24} items={len(d.entries):>4}  bozo={bozo}  url={row['url']}")
        if bozo and err:
            print(f"  ↳ parse warning: {err}")

