# app_corrected.py - Corrected extraction for NCC PDF
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import io, os, re, traceback
import pdfplumber

app = FastAPI(title="PDF Extractor - Corrected")

API_KEY = os.getenv("API_KEY")

# Expected column headers
PROJECT_COLS = [
    "#", "Projektnavn", "Roller", "Region",
    "Budget, kr.", "Byggestart", "Bæredygtighed",
    "Seneste opdateringsdato", "Stadie"
]

CONTACT_COLS = ["#", "Navn", "Firma / Navn", "Telefon", "Rolle"]

# --- Helper functions --------------------------------------------------------

def is_contact_page(page_text: str) -> bool:
    """Check if this page contains contact information"""
    text_lower = page_text.lower()
    # Look for contact-specific indicators
    return (
        "kontakter" in text_lower or
        ("navn" in text_lower and "firma" in text_lower and "telefon" in text_lower) or
        ("alex andersen" in text_lower) or  # Known first contact
        ("allan lavrsen" in text_lower)      # Known second contact
    )

def is_project_page(page_text: str) -> bool:
    """Check if this page contains project information"""
    text_lower = page_text.lower()
    return (
        "projekter" in text_lower or
        ("projektnavn" in text_lower and "budget" in text_lower) or
        ("renovering" in text_lower and "mio" in text_lower and "stadie" in text_lower)
    )

def extract_contacts_correctly(pdf) -> List[Dict[str, str]]:
    """
    Correctly extract contacts from the NCC PDF
    """
    contacts = []
    
    # The contacts section typically starts around page 6
    for page_num in range(min(5, len(pdf.pages)-1), len(pdf.pages)):
        page = pdf.pages[page_num]
        page_text = page.extract_text() or ""
        
        # Skip non-contact pages
        if not is_contact_page(page_text):
            continue
            
        # Skip if this looks like a project page
        if is_project_page(page_text):
            continue
        
        # Extract using table detection
        tables = page.extract_tables() or []
        
        for table in tables:
            # Process each row
            for row in table:
                if not row or len(row) < 2:
                    continue
                
                # Clean cells
                row = [str(cell).strip() if cell else "" for cell in row]
                
                # Check if this is a contact row (starts with 1-3 digit number)
                if row[0] and re.match(r'^\d{1,3}$', row[0]):
                    # Extract contact information based on actual column positions
                    contact_num = row[0]
                    
                    # Name is typically in column 1
                    name = row[1] if len(row) > 1 else ""
                    
                    # Company might be in column 2
                    company = row[2] if len(row) > 2 else ""
                    
                    # Phone might be in column 3
                    phone_field = row[3] if len(row) > 3 else ""
                    
                    # Role might be in column 4
                    role = row[4] if len(row) > 4 else ""
                    
                    # Sometimes name is split across columns
                    if company and re.match(r'^[a-zæøå]', company):
                        # Company column contains name continuation
                        name = name + company
                        company = ""
                    
                    # Fix name splits
                    name = fix_danish_name(name)
                    
                    # Extract phone numbers from any column
                    all_text = " ".join(row)
                    phones = re.findall(r'\b\d{8}\b', all_text)
                    
                    # Clean up role
                    if role == "A/S" or role == "Danmark":
                        role = ""
                    
                    contacts.append({
                        "#": contact_num,
                        "Navn": name,
                        "Telefon1": phones[0] if phones else "",
                        "Telefon2": phones[1] if len(phones) > 1 else "",
                        "Rolle": role
                    })
    
    # If table extraction failed, try text-based extraction
    if not contacts:
        contacts = extract_contacts_text_based(pdf)
    
    return merge_continuation_rows(contacts)

def extract_contacts_text_based(pdf) -> List[Dict[str, str]]:
    """
    Fallback: Extract contacts using text parsing
    """
    contacts = []
    
    # Look specifically at pages 6-11 (0-indexed: 5-10)
    for page_num in range(5, min(11, len(pdf.pages))):
        page = pdf.pages[page_num]
        text = page.extract_text() or ""
        
        # Skip if not a contact page
        if not is_contact_page(text):
            continue
            
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Look for lines that start with: number, name, company
            # Example: "1 Alex Andersen NCC Danmark A/S"
            # Example: "2 Allan Lavrsen NCC Danmark A/S 39103910 24887223"
            
            # Match pattern: number (1-3 digits) followed by a name
            match = re.match(r'^(\d{1,3})\s+([A-ZÆØÅ][a-zæøå]+\s+[A-ZÆØÅ][a-zæøå\-]+)', line)
            
            if match:
                contact_num = match.group(1)
                name = match.group(2)
                
                # Fix name
                name = fix_danish_name(name)
                
                # Extract phones from the line
                phones = re.findall(r'\b\d{8}\b', line)
                
                # Extract role - everything after company name and phones
                rest = line[match.end():]
                rest = re.sub(r'NCC\s*Danmark\s*A/?S', '', rest)
                rest = re.sub(r'\b\d{8}\b', '', rest)
                role = rest.strip()
                
                # Check next lines for continuation
                role_parts = [role] if role else []
                j = i + 1
                while j < len(lines) and j <= i + 3:
                    next_line = lines[j]
                    
                    # Check if it's a continuation (contains role keywords)
                    if ('Projektleder' in next_line or 'Kontaktperson' in next_line or
                        'entreprenør' in next_line or 'Entr.' in next_line):
                        role_parts.append(next_line.strip())
                        
                        # Also check for phones in continuation lines
                        more_phones = re.findall(r'\b\d{8}\b', next_line)
                        phones.extend(more_phones)
                    elif re.match(r'^\d{1,3}\s+[A-ZÆØÅ]', next_line):
                        # Next contact starts
                        break
                    
                    j += 1
                
                contacts.append({
                    "#": contact_num,
                    "Navn": name,
                    "Telefon1": phones[0] if phones else "",
                    "Telefon2": phones[1] if len(phones) > 1 else "",
                    "Rolle": " ".join(role_parts)
                })
    
    return contacts

def fix_danish_name(name: str) -> str:
    """Fix common Danish name splitting issues"""
    if not name:
        return name
    
    # Specific fixes for this PDF
    fixes = {
        r'Ande\s*rsen': 'Andersen',
        r'Lavr\s*sen': 'Lavrsen',
        r'Ka\s*mal': 'Kamal',
        r'Sch\s*ö\s*nau': 'Schönau',
        r'Eng\s*berg': 'Engberg',
        r'Holm-Pede\s*rsen': 'Holm-Pedersen',
    }
    
    for pattern, replacement in fixes.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    
    # General pattern: fix splits within words
    name = re.sub(r'([a-zæøå])\s+([a-zæøå])', r'\1\2', name)
    
    return name.strip()

def merge_continuation_rows(contacts: List[Dict]) -> List[Dict]:
    """
    Merge continuation rows (rows without contact number)
    """
    merged = []
    current = None
    
    for contact in contacts:
        if contact["#"] and re.match(r'^\d{1,3}$', contact["#"]):
            # New contact
            if current:
                merged.append(current)
            current = contact.copy()
        elif current:
            # Continuation row - merge into current
            if not current["Navn"] and contact["Navn"]:
                current["Navn"] = contact["Navn"]
            if not current["Telefon1"] and contact["Telefon1"]:
                current["Telefon1"] = contact["Telefon1"]
            elif not current["Telefon2"] and contact["Telefon1"]:
                current["Telefon2"] = contact["Telefon1"]
            if contact["Rolle"]:
                if current["Rolle"]:
                    current["Rolle"] += " " + contact["Rolle"]
                else:
                    current["Rolle"] = contact["Rolle"]
    
    if current:
        merged.append(current)
    
    return merged

def extract_projects_correctly(pdf) -> List[Dict[str, Any]]:
    """Extract projects from the correct pages"""
    projects = []
    
    for page_num, page in enumerate(pdf.pages):
        page_text = page.extract_text() or ""
        
        # Skip if not a project page
        if not is_project_page(page_text):
            continue
            
        # Skip if this is a contact page
        if is_contact_page(page_text):
            continue
        
        # Extract tables
        tables = page.extract_tables() or []
        
        for table in tables:
            # Find header row
            header_idx = -1
            for i, row in enumerate(table):
                row_text = " ".join([str(c) for c in row if c]).lower()
                if "projektnavn" in row_text or "budget" in row_text:
                    header_idx = i
                    break
            
            if header_idx >= 0:
                # Process data rows
                for row in table[header_idx + 1:]:
                    # Clean row
                    row = [str(c).strip() if c else "" for c in row]
                    
                    # Check if valid project row
                    if row and re.match(r'^\d+$', row[0]):
                        # Create project dict
                        project = {}
                        for i, col in enumerate(PROJECT_COLS):
                            project[col] = row[i] if i < len(row) else ""
                        
                        # Clean up
                        project["#"] = re.sub(r'\D+$', '', project["#"])
                        projects.append(project)
    
    return projects

# --- FastAPI routes ----------------------------------------------------------

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

        projects = []
        contacts = []

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if "projects" in which_set:
                projects = extract_projects_correctly(pdf)
            
            if "contacts" in which_set:
                contacts = extract_contacts_correctly(pdf)
                
                # If still no contacts, force text-based extraction
                if not contacts:
                    contacts = extract_contacts_text_based(pdf)

        return JSONResponse({
            "ok": True,
            "counts": {"projects": len(projects), "contacts": len(contacts)},
            "projects": projects,
            "contacts": contacts
        })

    except Exception as e:
        print("---- EXTRACT ERROR ----")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
