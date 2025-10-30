# app.py
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import io, os, re, traceback
import pdfplumber

app = FastAPI(title="PDF Extractor")

API_KEY = os.getenv("API_KEY")  # optional

PROJECT_COLS = ["#", "Projektnavn", "Roller", "Region",
                "Budget, kr.", "Byggestart", "Bæredygtighed",
                "Seneste opdateringsdato", "Stadie"]

CONTACT_COLS = ["#", "Navn", "Firma / Navn", "Telefon", "Rolle"]

def clean(s: Optional[str]) -> str:
    if s is None: return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def table_settings():
    return dict(
        vertical_strategy="text",
        horizontal_strategy="text",
        snap_tolerance=3,
        join_tolerance=3,
        edge_min_length=3,
        keep_blank_chars=False,
        text_tolerance=2,
        intersection_tolerance=3,
        min_words_horizontal=2,
        min_words_vertical=2
    )

def detect_section(page) -> str:
    top = (page.extract_text() or "")[:600].lower()
    if "projekter" in top: return "projects"
    if "kontakter" in top: return "contacts"
    return "unknown"

def normalize_header(row: List[str], expected: List[str]) -> List[str]:
    r = [clean(c) for c in row]
    out = []
    for c in r:
        c = c.replace(" ,", ",").replace("Firma/ Navn","Firma / Navn")
        out.append(c)
    if len(out) < len(expected):
        out += [""] * (len(expected) - len(out))
    return out[:len(expected)]

def extract_tables_from_page(page, expected_header: List[str]) -> List[List[List[str]]]:
    w, h = page.width, page.height
    content = page.crop((0, 0.07*h, w, h))  # trim top band

    tables = content.extract_tables(table_settings()) or []
    cleaned = []
    for t in tables:
        t = [[clean(c) for c in row] for row in t if any(c and c.strip() for c in row)]
        if not t: 
            continue

        header_idx = -1
        for i, row in enumerate(t[:4]):
            row_join = " ".join(row).lower()
            hits = sum(1 for col in expected_header if col.split(",")[0].lower() in row_join)
            if hits >= max(2, len(expected_header)//3):
                header_idx = i
                break

        if header_idx == -1:
            body = [r for r in t if r and re.match(r"^\d+(\.|)$", r[0])]
            if body:
                cleaned.append([expected_header] + body)
            continue

        header = normalize_header(t[header_idx], expected_header)
        body = t[header_idx+1:]
        if body and header:
            cleaned.append([header] + body)

    return cleaned

def merge_to_dicts(chunks: List[List[List[str]]], expected_header: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for tbl in chunks:
        header, *body = tbl
        for r in body:
            r = (r + [""] * len(expected_header))[:len(expected_header)]
            rows.append({k: clean(v) for k, v in zip(expected_header, r)})
    return rows

def post_projects(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        r["#"] = re.sub(r"\D+$", "", r["#"])
        r["Budget, kr."] = r["Budget, kr."].replace(" mia.", " mia").replace(" mio.", " mio")
        out.append(r)
    seen, uniq = set(), []
    for r in out:
        key = (r["#"], r["Projektnavn"])
        if key in seen: 
            continue
        seen.add(key); uniq.append(r)
    return uniq

def post_contacts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for r in rows:
        if r["#"] and re.match(r"^\d+$", r["#"]):
            merged.append(r)
        elif merged:
            merged[-1]["Rolle"] = clean(merged[-1]["Rolle"] + " " + " ".join([
                r.get("Navn",""), r.get("Firma / Navn",""),
                r.get("Telefon",""), r.get("Rolle","")
            ]))
    return merged

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    which: str = Form("projects,contacts"),
    x_api_key: Optional[str] = Header(default=None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        data = await file.read()
        which_set = {w.strip().lower() for w in which.split(",") if w.strip()}

        projects: List[Dict[str, Any]] = []
        contacts: List[Dict[str, Any]] = []

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                section = detect_section(page)

                if "projects" in which_set and section == "projects":
                    chunks = extract_tables_from_page(page, PROJECT_COLS)
                    if chunks:
                        projects += merge_to_dicts(chunks, PROJECT_COLS)

                if "contacts" in which_set and section == "contacts":
                    chunks = extract_tables_from_page(page, CONTACT_COLS)
                    if chunks:
                        contacts += merge_to_dicts(chunks, CONTACT_COLS)

        if projects: projects = post_projects(projects)
        if contacts: contacts = post_contacts(contacts)

        return JSONResponse({
            "ok": True,
            "counts": {"projects": len(projects), "contacts": len(contacts)},
            "projects": projects if "projects" in which_set else [],
            "contacts": contacts if "contacts" in which_set else []
        })

    except Exception as e:
        # log full traceback to Railway logs
        print("---- EXTRACT ERROR ----")
        print(traceback.format_exc())
        # return readable error to the client (Swagger)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
