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

# ---------- helpers ----------
def clean(s: Optional[str]) -> str:
    if s is None: return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def table_settings():
    # Compatible with pdfplumber on Railway
    return dict(
        vertical_strategy="text",
        horizontal_strategy="text",
        snap_tolerance=3,
        join_tolerance=3,
        edge_min_length=3,
        text_tolerance=2,
        intersection_tolerance=3,
    )

def detect_section(page) -> str:
    top = (page.extract_text() or "")[:600].lower()
    if "projekter" in top: return "projects"
    if "kontakter" in top: return "contacts"
    return "unknown"

def looks_like_contacts(page) -> bool:
    t = (page.extract_text() or "").lower()
    return ("# navn" in t and "rolle" in t) or ("navn firma" in t and "telefon" in t)

def looks_like_projects(page) -> bool:
    t = (page.extract_text() or "").lower()
    return ("projektnavn" in t and "budget" in t) or ("region" in t and "stadie" in t)

def normalize_header(row: List[str], expected: List[str]) -> List[str]:
    r = [clean(c) for c in row]
    out = []
    for c in r:
        c = c.replace(" ,", ",").replace("Firma/ Navn", "Firma / Navn")
        out.append(c)
    if len(out) < len(expected):
        out += [""] * (len(expected) - len(out))
    return out[:len(expected)]

def extract_tables_from_page(page, expected_header: List[str]) -> List[List[List[str]]]:
    w, h = page.width, page.height
    content = page.crop((0, 0.07*h, w, h))  # trim top band with page title

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
            # tolerate chunks without header; keep rows starting with index
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

# ---------- projects ----------
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

# ---------- contacts (company omitted) ----------
def post_contacts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge multi-line rows and return schema:
    { "#", "Navn", "Telefon1", "Telefon2", "TelefonExtra", "Rolle" }
    """
    COMPANY_TOKENS = {"ncc", "danmark", "a/s", "as", "a.s.", "a-s", "a", "s", "a/s,"}

    def strip_company_tokens(text: str) -> str:
        parts = text.split()
        keep = []
        for p in parts:
            if p.lower() in COMPANY_TOKENS:
                continue
            keep.append(p)
        return " ".join(keep)

    def find_phones(*texts: str) -> List[str]:
        # Danish-style 8-digit numbers (allow linebreak merges)
        nums = re.findall(r"\b\d{8}\b", " ".join(t for t in texts if t))
        # de-dup preserving order
        seen, out = set(), []
        for n in nums:
            if n not in seen:
                seen.add(n); out.append(n)
        return out

    def normalize_role(text: str) -> str:
        t = clean(text)
        # ensure a space after dots like "Projektleder." -> "Projektleder. "
        t = re.sub(r"\.(?=[A-Za-zÆØÅæøå])", ". ", t)
        return clean(t)

    def merge_buffer(buf: Dict[str, str]) -> Dict[str, str]:
        # Build Navn from 'Navn' + any surname that slipped into 'Firma / Navn' (but drop company tokens)
        name_core = clean(buf.get("Navn", ""))
        firm_bits = strip_company_tokens(clean(buf.get("Firma / Navn", "")))
        # If firm_bits looks like a name fragment, append it
        if firm_bits and re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ.'\- ]+", firm_bits):
            name = f"{name_core} {firm_bits}".strip()
        else:
            name = name_core

        # Phones: look across all fields in the merged buffer
        phones = find_phones(buf.get("Telefon", ""),
                             buf.get("Firma / Navn", ""),
                             buf.get("Navn", ""),
                             buf.get("Rolle", ""))
        tel1 = phones[0] if len(phones) > 0 else ""
        tel2 = phones[1] if len(phones) > 1 else ""
        tel_extra = ", ".join(phones[2:]) if len(phones) > 2 else ""

        rolle = normalize_role(buf.get("Rolle", ""))

        result = {
            "#": clean(buf.get("#", "")),
            "Navn": name,
            "Telefon1": tel1,
            "Telefon2": tel2,
            "Rolle": rolle
        }
        if tel_extra:
            result["TelefonExtra"] = tel_extra
        return result

    merged: List[Dict[str, str]] = []
    buf: Dict[str, str] = {}

    for r in rows:
        # New logical row when '#' is an integer; otherwise continuation
        if r.get("#") and re.fullmatch(r"\d+", r["#"]):
            if buf:
                merged.append(merge_buffer(buf))
            buf = {k: clean(v) for k, v in r.items()}
        else:
            for k, v in r.items():
                if not v:
                    continue
                buf[k] = (buf.get(k, "") + " " + clean(v)).strip()

    if buf:
        merged.append(merge_buffer(buf))

    return merged

# ---------- routes ----------
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
        contacts_raw: List[Dict[str, Any]] = []

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            current_section = None  # remember last detected section

            for page in pdf.pages:
                section = detect_section(page)

                # fallback heuristics for continuation pages
                if section == "unknown":
                    if looks_like_contacts(page):
                        section = "contacts"
                    elif looks_like_projects(page):
                        section = "projects"
                    elif current_section:
                        section = current_section

                if section != "unknown":
                    current_section = section

                if "projects" in which_set and section == "projects":
                    chunks = extract_tables_from_page(page, PROJECT_COLS)
                    if chunks:
                        projects += merge_to_dicts(chunks, PROJECT_COLS)

                if "contacts" in which_set and section == "contacts":
                    chunks = extract_tables_from_page(page, CONTACT_COLS)
                    if chunks:
                        contacts_raw += merge_to_dicts(chunks, CONTACT_COLS)

        if projects:
            projects = post_projects(projects)
        contacts = post_contacts(contacts_raw) if contacts_raw else []

        return JSONResponse({
            "ok": True,
            "counts": {"projects": len(projects), "contacts": len(contacts)},
            "projects": projects if "projects" in which_set else [],
            "contacts": contacts if "contacts" in which_set else []
        })

    except Exception as e:
        print("---- EXTRACT ERROR ----")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
