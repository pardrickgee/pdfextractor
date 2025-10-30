# app.py - Fixed version with robust table extraction for NCC PDF
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

# Raw table shape in the PDF - Note: actual columns in PDF are different!
CONTACT_COLS = ["#", "Navn", "Firma / Navn", "Telefon", "Rolle"]

# --- helpers -----------------------------------------------------------------

def clean(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def unsplit_broken_words(text: str) -> str:
    """
    Repair common mid-word splits introduced by table extraction.
    Enhanced for Danish names and common patterns in this PDF.
    """
    if not text:
        return text
    
    # First pass: handle obvious splits with special chars
    text = re.sub(r'(\w+)\s+([a-zæøå]\w*)', r'\1\2', text)  # "Ande rsen" -> "Andersen"
    
    # Handle specific patterns we see in this PDF
    patterns = [
        (r'Ande\s+rsen', 'Andersen'),
        (r'Lavr\s+sen', 'Lavrsen'),
        (r'Ka\s+mal', 'Kamal'),
        (r'Hein\s+ze', 'Heinze'),
        (r'Hjer\s+rild', 'Hjerrild'),
        (r'Møll\s+er', 'Møller'),
        (r'L\s+und', 'Lund'),
        (r'Ha\s+ack', 'Haack'),
        (r'Eng\s+berg', 'Engberg'),
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def fix_table_alignment(rows: List[List[str]]) -> List[List[str]]:
    """
    Fix misaligned table data specific to this PDF's contact table structure.
    The PDF seems to have data shifted across columns.
    """
    fixed = []
    for row in rows:
        if len(row) < 5:
            row = row + [""] * (5 - len(row))
        
        # Detect and fix common misalignments
        # If column 2 looks like a name fragment and column 3 is "NCC"
        if (len(row) >= 4 and 
            row[2] and re.match(r'^[a-zæøå]', row[2]) and  # starts lowercase
            'ncc' in row[3].lower()):
            # This is a split name - merge columns 1 and 2
            new_row = [
                row[0],  # #
                unsplit_broken_words(row[1] + row[2]),  # Fixed name
                "NCC Danmark A/S",  # Company
                "",  # Phone will be in next row
                ""   # Role will be in next row
            ]
            fixed.append(new_row)
        else:
            fixed.append(row[:5])
    
    return fixed

def extract_tables_from_page(page, expected_header: List[str]) -> List[List[List[str]]]:
    """Enhanced table extraction specifically tuned for this PDF format"""
    w, h = page.width, page.height
    content = page.crop((0, 0.07 * h, w, h))  # trim header
    
    # For contacts, we need a very specific approach
    if "Navn" in expected_header:
        # Extract raw text first to understand structure
        raw_text = content.extract_text() or ""
        
        # Use pdfplumber's table detection with custom settings
        settings = dict(
            vertical_strategy="text",
            horizontal_strategy="text",
            snap_tolerance=2,
            join_tolerance=2,
            edge_min_length=2,
            text_tolerance=1,
            intersection_tolerance=2,
        )
        
        tables = content.extract_tables(settings) or []
        
        # Alternative: extract as words and reconstruct
        if not tables or all(len(t) < 2 for t in tables):
            # Fall back to word-based extraction
            words = content.extract_words(
                keep_blank_chars=False,
                use_text_flow=True,
                extra_attrs=['fontname', 'size']
            )
            
            # Group words into lines based on y-position
            lines = {}
            for w in words:
                y = round(w['top'])
                if y not in lines:
                    lines[y] = []
                lines[y].append(w)
            
            # Sort lines by y-position and words by x-position
            sorted_lines = []
            for y in sorted(lines.keys()):
                line_words = sorted(lines[y], key=lambda x: x['x0'])
                line_text = []
                
                # Group words into columns based on x-position gaps
                last_x = -100
                current_col = []
                for w in line_words:
                    if w['x0'] - last_x > 20:  # New column
                        if current_col:
                            line_text.append(" ".join(current_col))
                        current_col = [w['text']]
                    else:
                        current_col.append(w['text'])
                    last_x = w['x1']
                
                if current_col:
                    line_text.append(" ".join(current_col))
                
                if line_text:
                    sorted_lines.append(line_text)
            
            # Convert to table format
            if sorted_lines:
                tables = [sorted_lines]
    else:
        # Projects table - use standard approach
        settings = dict(
            vertical_strategy="text",
            horizontal_strategy="text",
            snap_tolerance=3,
            join_tolerance=3,
            edge_min_length=3,
            text_tolerance=2,
            intersection_tolerance=3,
        )
        tables = content.extract_tables(settings) or []
    
    # Process extracted tables
    cleaned = []
    for t in tables:
        # Clean cells
        t = [[clean(c) if c else "" for c in row] for row in t]
        # Remove empty rows
        t = [row for row in t if any(c.strip() for c in row)]
        
        if not t:
            continue
        
        # Find header row
        header_idx = -1
        for i, row in enumerate(t[:5]):
            row_text = " ".join(row).lower()
            # Check for contact header keywords
            if "navn" in expected_header:
                if "navn" in row_text and ("firma" in row_text or "telefon" in row_text):
                    header_idx = i
                    break
            else:
                # Project header
                if "projektnavn" in row_text or "budget" in row_text:
                    header_idx = i
                    break
        
        if header_idx >= 0:
            body = t[header_idx + 1:]
            if body:
                # Fix alignment issues for contacts
                if "Navn" in expected_header:
                    body = fix_table_alignment(body)
                cleaned.append([expected_header] + body)
        else:
            # No header found, but check if data starts with numbers
            numbered_rows = [r for r in t if r and re.match(r'^\d+\.?$', r[0])]
            if numbered_rows:
                if "Navn" in expected_header:
                    numbered_rows = fix_table_alignment(numbered_rows)
                cleaned.append([expected_header] + numbered_rows)
    
    return cleaned

def merge_to_dicts(chunks: List[List[List[str]]], expected_header: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for tbl in chunks:
        if not tbl:
            continue
        _, *body = tbl  # Skip header
        for r in body:
            # Ensure row has correct length
            r = (r + [""] * len(expected_header))[:len(expected_header)]
            rows.append({k: v for k, v in zip(expected_header, r)})
    return rows

# --- post-processing ----------------------------------------------------------

def post_projects(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        r["#"] = re.sub(r"\D+$", "", r["#"])
        r["Budget, kr."] = r["Budget, kr."].replace(" mia.", " mia").replace(" mio.", " mio")
        out.append(r)
    
    # Remove duplicates
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
    Process and merge continuation rows.
    Returns: { "#", "Navn", "Telefon1", "Telefon2", "Rolle" }
    """
    
    # First, let's examine what we actually got
    print(f"DEBUG: Processing {len(rows)} raw contact rows")
    if rows and len(rows) > 0:
        print(f"DEBUG: Sample row keys: {list(rows[0].keys())}")
        print(f"DEBUG: First 3 rows:")
        for i, r in enumerate(rows[:3]):
            print(f"  Row {i}: {r}")
    
    def extract_phones(text: str) -> List[str]:
        """Extract 8-digit phone numbers from text"""
        if not text:
            return []
        nums = re.findall(r'\b\d{8}\b', text)
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result
    
    def is_new_contact(row: Dict) -> bool:
        """Check if this row starts a new contact"""
        num = str(row.get("#", "")).strip()
        return bool(num and re.match(r'^\d+$', num))
    
    merged = []
    current = None
    
    for row in rows:
        if is_new_contact(row):
            # Save previous contact if exists
            if current:
                # Fix name
                name = unsplit_broken_words(current.get("name", ""))
                
                # Extract phones from all accumulated text
                all_text = " ".join([
                    current.get("name", ""),
                    current.get("firma", ""),
                    current.get("telefon", ""),
                    current.get("rolle", "")
                ])
                phones = extract_phones(all_text)
                
                merged.append({
                    "#": current["#"],
                    "Navn": name,
                    "Telefon1": phones[0] if len(phones) > 0 else "",
                    "Telefon2": phones[1] if len(phones) > 1 else "",
                    "Rolle": current.get("rolle", "").replace("  ", " ").strip()
                })
            
            # Start new contact
            current = {
                "#": row.get("#", "").strip(),
                "name": clean(row.get("Navn", "")),
                "firma": clean(row.get("Firma / Navn", "")),
                "telefon": clean(row.get("Telefon", "")),
                "rolle": clean(row.get("Rolle", ""))
            }
            
            # Handle split names
            if current["firma"] and re.match(r'^[a-zæøå]', current["firma"]):
                # Firma column contains name continuation
                current["name"] = current["name"] + current["firma"]
                current["firma"] = ""
        
        elif current:
            # Continuation row - append data
            for key in ["Navn", "Firma / Navn", "Telefon", "Rolle"]:
                val = clean(row.get(key, ""))
                if val:
                    if key == "Navn":
                        current["name"] = (current["name"] + " " + val).strip()
                    elif key == "Firma / Navn":
                        if not current["firma"] or current["firma"] == "NCC Danmark A/S":
                            current["firma"] = val
                    elif key == "Telefon":
                        current["telefon"] = (current["telefon"] + " " + val).strip()
                    elif key == "Rolle":
                        if current["rolle"]:
                            current["rolle"] += " " + val
                        else:
                            current["rolle"] = val
    
    # Don't forget last contact
    if current:
        name = unsplit_broken_words(current.get("name", ""))
        all_text = " ".join([
            current.get("name", ""),
            current.get("firma", ""),
            current.get("telefon", ""),
            current.get("rolle", "")
        ])
        phones = extract_phones(all_text)
        
        merged.append({
            "#": current["#"],
            "Navn": name,
            "Telefon1": phones[0] if len(phones) > 0 else "",
            "Telefon2": phones[1] if len(phones) > 1 else "",
            "Rolle": current.get("rolle", "").replace("  ", " ").strip()
        })
    
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
            # Process each page
            for page_num, page in enumerate(pdf.pages):
                page_text = (page.extract_text() or "").lower()
                
                # Detect section
                is_projects = "projektnavn" in page_text or "budget" in page_text
                is_contacts = ("navn" in page_text and "telefon" in page_text) or "kontakter" in page_text
                
                if "projects" in which_set and is_projects:
                    chunks = extract_tables_from_page(page, PROJECT_COLS)
                    if chunks:
                        projects += merge_to_dicts(chunks, PROJECT_COLS)
                
                if "contacts" in which_set and is_contacts:
                    chunks = extract_tables_from_page(page, CONTACT_COLS)
                    if chunks:
                        contacts_raw += merge_to_dicts(chunks, CONTACT_COLS)

        # Post-process
        if projects:
            projects = post_projects(projects)

        contacts: List[Dict[str, Any]] = []
        if contacts_raw:
            # Remove debug output in production
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()  # Suppress debug output
            try:
                contacts = post_contacts(contacts_raw)
            finally:
                sys.stdout = old_stdout

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
