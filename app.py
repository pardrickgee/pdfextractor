# app.py
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Tuple
import io, os, re, traceback
import pdfplumber

app = FastAPI(title="PDF Extractor")

API_KEY = os.getenv("API_KEY")  # optional

PROJECT_COLS = [
    "#", "Projektnavn", "Roller", "Region",
    "Budget, kr.", "Byggestart", "Bæredygtighed",
    "Seneste opdateringsdato", "Stadie"
]

# We still define the “table” header for contacts so we can reuse some utilities,
# but the actual extraction for contacts is now TEXT-BASED (see extract_contacts_from_text).
CONTACT_COLS = ["#", "Navn", "Firma / Navn", "Telefon", "Rolle"]

# ----------------- utils -----------------
def clean(s: Optional[str]) -> str:
    if s is None: return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def table_settings():
    # Compatible with Railway’s pdfplumber build
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
    content = page.crop((0, 0.07*h, w, h))  # trim the top ribbon with title/date

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
            r =
