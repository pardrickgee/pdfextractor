from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import pdfplumber, io, re
from typing import List, Dict, Any, Optional

app = FastAPI(title="PDF Extractor")

PROJECT_COLS = ["#", "Projektnavn", "Roller", "Region",
                "Budget, kr.", "Byggestart", "Bæredygtighed",
                "Seneste opdateringsdato", "Stadie"]
CONTACT_COLS = ["#", "Navn", "Firma / Navn", "Telefon", "Rolle"]

def clean(x: Optional[str]) -> str:
    if x is None: return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def table_settings():
    return dict(vertical_strategy="text", horizontal_strategy="text")

def detect_section(page) -> str:
    text = (page.extract_text() or "").lower()
    if "projekter" in text: return "projects"
    if "kontakter" in text: return "contacts"
    return "unknown"

@app.post("/extract")
async def extract(file: UploadFile = File(...), which: str = Form("projects,contacts")):
    content = await file.read()
    projects, contacts = [], []
    which_set = set(which.split(","))

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            section = detect_section(page)
            if "projects" in which_set and section == "projects":
                text = page.extract_text() or ""
                projects.append({"page": page.page_number, "text": text})
            elif "contacts" in which_set and section == "contacts":
                text = page.extract_text() or ""
                contacts.append({"page": page.page_number, "text": text})

    return JSONResponse({"ok": True, "projects": projects, "contacts": contacts})
