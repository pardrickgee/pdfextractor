from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
import pdfplumber, io, re, os

app = FastAPI(title="Byggefakta PDF Extractor")

API_KEY = os.getenv("API_KEY")

PROJECT_COLS = ["#", "Projektnavn", "Roller", "Region",
                "Budget, kr.", "Byggestart", "Bæredygtighed",
                "Seneste opdateringsdato", "Stadie"]
CONTACT_COLS = ["#", "Navn", "Firma / Navn", "Telefon", "Rolle"]

def clean(x): return re.sub(r"\s+", " ", x).strip() if x else ""

def table_settings():
    return dict(vertical_strategy="text", horizontal_strategy="text",
                snap_tolerance=3, join_tolerance=3, edge_min_length=3,
                keep_blank_chars=False, text_tolerance=2,
                intersection_tolerance=3, min_words_horizontal=2, min_words_vertical=2)

def detect_section(page):
    t = (page.extract_text() or "")[:400].lower()
    if "projekter" in t: return "projects"
    if "kontakter" in t: return "contacts"
    return "unknown"

def normalize_header(row, expected):
    r = [clean(c) for c in row]
    r = [c.replace(" ,", ",").replace("Firma/ Navn", "Firma / Navn") for c in r]
    if len(r) < len(expected): r += [""] * (len(expected)*
