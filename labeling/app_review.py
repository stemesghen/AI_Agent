import json, glob
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")
NORM_DIR = DATA_DIR / "normalized"
CLS_DIR  = DATA_DIR / "classified"
EXT_DIR  = DATA_DIR / "extracted"
LABELS_F = DATA_DIR / "labels" / "review.csv"
LABELS_F.parent.mkdir(parents=True, exist_ok=True)

INCIDENT_TYPES = ["grounding","collision","fire","piracy","weather","port_closure","strike","spill"]

@st.cache_data
def load_rows():
    rows = []
    cls_files = glob.glob(str(CLS_DIR / "*.classify.json"))
    for cf in cls_files:
        with open(cf, "r", encoding="utf-8") as f:
            cls = json.load(f)
        doc_id = cls["doc_id"]

        nf = NORM_DIR / (Path(cf).name.replace(".classify.json", ".json"))
        if not nf.exists():
            continue
        with open(nf, "r", encoding="utf-8") as f:
            norm = json.load(f)

        ef = EXT_DIR / (Path(cf).name.replace(".classify.json", ".extract.json"))
        extracted = {}
        if ef.exists():
            with open(ef, "r", encoding="utf-8") as f:
                extracted = json.load(f)

        rows.append({
            "doc_id": doc_id,
            "title": norm.get("title",""),
            "url": norm.get("url",""),
            "published_at": (norm.get("published_at","") or "")[:10],
            "source_id": norm.get("source_id",""),
            "is_incident_pred": bool(cls.get("is_incident", False)),
            "incident_types_pred": ",".join(cls.get("incident_types", [])),
            "vessel_pred": (extracted.get("vessel") if extracted else None),
            "imo_pred": (extracted.get("imo") if extracted else None),
            "port_pred": (extracted.get("port") if extracted else None),
            "date_pred": (extracted.get("date") if extracted else None),
            "content_text": (norm.get("content_text","") or "")[:2000],
        })
    return pd.DataFrame(rows)

def load_labels():
    if LABELS_F.exists():
        return pd.read_csv(LABELS_F)
    return pd.DataFrame(columns=[
        "doc_id","is_incident_true","incident_types_true","vessel_true","imo_true","port_true","date_true","notes"
    ])

def upsert_label(row):
    df = load_labels()
    idx = df.index[df["doc_id"] == row["doc_id"]]
    if len(idx):
        df.loc[idx[0]] = row
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LABELS_F, index=False)

st.set_page_config(page_title="Incident Review", layout="wide")
st.title("Incident Review & Labeling")

data = load_rows()
labels = load_labels()

left, right = st.columns([2,3])

with left:
    st.subheader("Filters")
    only_inc = st.checkbox("Only show predicted incidents", value=True)
    q = st.text_input("Search title")
    df = data.copy()
    if only_inc:
        df = df[df["is_incident_pred"] == True]
    if q:
        df = df[df["title"].str.contains(q, case=False, na=False)]
    st.caption(f"{len(df)} items")
    idx = st.number_input("Row", min_value=0, max_value=max(len(df)-1,0), value=0, step=1)

with right:
    if len(df) == 0:
        st.info("No rows match your filter.")
    else:
        row = df.iloc[int(idx)].to_dict()
        st.subheader(row["title"])
        st.write(f"**Date**: {row['published_at']}  |  **Source**: {row['source_id']}")
        if row["url"]:
            st.write(f"[Open article]({row['url']})")
        with st.expander("Show content", expanded=False):
            st.write(row["content_text"])

        st.markdown("---")
        st.subheader("Predictions")
        p1, p2 = st.columns([1,3])
        p1.metric("Pred is_incident", "YES" if row["is_incident_pred"] else "NO")
        p2.write(f"Types: `{row['incident_types_pred']}` · Vessel: `{row['vessel_pred']}` · "
                 f"Port: `{row['port_pred']}` · Date: `{row['date_pred']}` · IMO: `{row['imo_pred']}`")

        st.markdown("---")
        st.subheader("Your Labels (Ground Truth)")

        # pull prior saved label if exists
        prior = labels[labels["doc_id"] == row["doc_id"]]
        prior = prior.iloc[0].to_dict() if len(prior) else {}

        c1, c2 = st.columns(2)
        is_incident_true = c1.selectbox("Is incident?", [True, False],
                                        index=0 if row["is_incident_pred"] else 1,
                                        key=f"is_{row['doc_id']}")
        incident_types_true = c1.multiselect("Incident types", INCIDENT_TYPES,
                                             default=(row["incident_types_pred"].split(",") if row["incident_types_pred"] else []),
                                             key=f"types_{row['doc_id']}")
        vessel_true = c2.text_input("Vessel", value=prior.get("vessel_true", row.get("vessel_pred") or ""), key=f"v_{row['doc_id']}")
        port_true   = c2.text_input("Port", value=prior.get("port_true", row.get("port_pred") or ""), key=f"p_{row['doc_id']}")
        date_true   = c2.text_input("Date (YYYY-MM-DD)", value=prior.get("date_true", row.get("date_pred") or row["published_at"]), key=f"d_{row['doc_id']}")
        imo_true    = c2.text_input("IMO (7 digits)", value=prior.get("imo_true", row.get("imo_pred") or ""), key=f"imo_{row['doc_id']}")
        notes       = st.text_area("Notes", value=prior.get("notes",""), key=f"n_{row['doc_id']}")

        if st.button("💾 Save label", type="primary"):
            upsert_label({
                "doc_id": row["doc_id"],
                "is_incident_true": bool(is_incident_true),
                "incident_types_true": ",".join(incident_types_true),
                "vessel_true": vessel_true.strip(),
                "imo_true": imo_true.strip(),
                "port_true": port_true.strip(),
                "date_true": date_true.strip(),
                "notes": notes.strip(),
            })
            st.success("Saved!")

