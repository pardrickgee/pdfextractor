# app.py
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import io, os, re, traceback
import pdfplumber

app = FastAPI(title="PDF Extractor")

API_KEY = os.getenv("API_KEY")  # optional

PROJECT_COLS = [
    "#", "Projektnavn", "Roller", "Region",
    "Budget, kr.", "Byggestart", "Bæredygtighed",
    "Seneste opdateringsdato", "Stadie"
]

# Raw table shape in the PDF
CONTACT_COLS = ["#", "Navn", "Firma / Navn", "Telefon", "Rolle"]

# --- helpers -----------------------------------------------------------------

def clean(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def unsplit_broken_words(text: str) -> str:
    """
    Repair common mid-word splits introduced by table extraction, e.g. 'Ande rsen' -> 'Andersen'.
    We only join if both parts are purely alphabetic and the second part is lowercase-leading or very short.
    """
    tokens = text.split(" ")
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if i + 1 < len(tokens):
            n = tokens[i + 1]
            if re.fullmatch(r"[A-Za-zÆØÅæøå.\-]+", t) and re.fullmatch(r"[A-Za-zÆØÅæøå.\-]+", n):
                # Heuristic: short right piece OR lowercase-start right piece => join
                if len(n) <= 3 or (n and n[0].islower()):
                    out.append(t + n)
                    i += 2
                    continue
        out.append(t)
        i += 1
    return " ".join(out)

def table_settings() -> Dict[str, Any]:
    # Keep args compatible with current pdfplumber on Railway
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
    top = (page.extract_text() or "")[:800].lower()
    if "projekter" in top:
        return "projects"
    if "kontakter" in top:
        return "contacts"
    return "unknown"

def looks_like_contacts(page) -> bool:
    t = (page.extract_text() or "").lower()
    return ("# navn" in t and "rolle" in t) or ("navn" in t and "telefon" in t)

def looks_like_projects(page) -> bool:
    t = (page.extract_text() or "").lower()
    return ("projektnavn" in t and "budget" in t) or ("stadie" in t)

def normalize_header(row: List[str], expected: List[str]) -> List[str]:
    r = [clean(c) for c in row]
    r = [c.replace(" ,", ",").replace("Firma/ Navn", "Firma / Navn") for c in r]
    if len(r) < len(expected):
        r += [""] * (len(expected) - len(r))
    return r[:len(expected)]

def extract_tables_from_page(page, expected_header: List[str]) -> List[List[List[str]]]:
    w, h = page.width, page.height
    content = page.crop((0, 0.07 * h, w, h))  # trim top band

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
            if hits >= max(2, len(expected_header) // 3):
                header_idx = i
                break

        if header_idx == -1:
            # No visible header, but sometimes rows already start with '#'
            body = [r for r in t if r and re.match(r"^\d+(\.|)$", r[0])]
            if body:
                cleaned.append([expected_header] + body)
            continue

        header = normalize_header(t[header_idx], expected_header)
        body = t[header_idx + 1:]
        if body and header:
            cleaned.append([header] + body)

    return cleaned

def merge_to_dicts(chunks: List[List[List[str]]], expected_header: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for tbl in chunks:
        _, *body = tbl
        for r in body:
            r = (r + [""] * len(expected_header))[:len(expected_header)]
            rows.append({k: clean(v) for k, v in zip(expected_header, r)})
    return rows

# --- post-processing ----------------------------------------------------------

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
        seen.add(key)
        uniq.append(r)
    return uniq

def post_contacts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge continuation rows and return compact schema (no company):
    { "#", "Navn", "Telefon1", "Telefon2", "Rolle" }
    """
    COMPANY_TOKENS = {"ncc", "danmark", "a/s", "a.s.", "as", "a-s"}

    def strip_company_tokens(text: str) -> str:
        toks = [t for t in text.split() if t.lower() not in COMPANY_TOKENS]
        return " ".join(toks)

    def looks_like_name_fragment(text: str) -> bool:
        txt = strip_company_tokens(text)
        return bool(txt) and bool(re.fullmatch(r"[A-Za-zÆØÅæøå.\- ]+", txt))

    def phones_from_text(*parts: str) -> List[str]:
        nums = re.findall(r"\b\d{8}\b", " ".join(p for p in parts if p))
        seen, out = set(), []
        for n in nums:
            if n not in seen:
                seen.add(n); out.append(n)
        return out

    def finalize(buf: Dict[str, str]) -> Dict[str, str]:
        name_core = unsplit_broken_words(clean(buf.get("Navn", "")))
        firm = unsplit_broken_words(strip_company_tokens(clean(buf.get("Firma / Navn", ""))))
        # If firma text is actually a surname fragment, glue it to the name
        name = (name_core + " " + firm).strip() if looks_like_name_fragment(firm) else name_core
        name = re.sub(r"\s+", " ", name).strip()

        # Phones: anywhere in the row (sometimes they sit in other columns)
        phones = phones_from_text(buf.get("Telefon", ""), buf.get("Navn", ""), buf.get("Firma / Navn", ""), buf.get("Rolle", ""))
        tel1 = phones[0] if len(phones) > 0 else ""
        tel2 = phones[1] if len(phones) > 1 else ""

        rolle = unsplit_broken_words(strip_company_tokens(clean(buf.get("Rolle", ""))))
        rolle = rolle.replace("  ", " ").strip(" ,.;")

        return {
            "#": clean(buf.get("#", "")),
            "Navn": name,
            "Telefon1": tel1,
            "Telefon2": tel2,
            "Rolle": rolle
        }

    merged: List[Dict[str, str]] = []
    buf: Dict[str, str] = {}

    for r in rows:
        row_has_index = bool(r.get("#") and re.fullmatch(r"\d+", r["#"]))
        if row_has_index:
            if buf:
                merged.append(finalize(buf))
            buf = {k: clean(v) for k, v in r.items()}
        else:
            # Continuation row: append cell-by-cell
            for k, v in r.items():
                if not v:
                    continue
                buf[k] = (buf.get(k, "") + " " + clean(v)).strip()

    if buf:
        merged.append(finalize(buf))

    # Last polish: drop empty roles that are just company leftovers
    for m in merged:
        if m["Rolle"].lower() in {"", "a/s", "as", "danmark"}:
            m["Rolle"] = ""

    return merged

# --- routes -------------------------------------------------------------------

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
            current_section = None
            for page in pdf.pages:
                section = detect_section(page)
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

        contacts: List[Dict[str, Any]] = []
        if contacts_raw:
            contacts = post_contacts(contacts_raw)

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
