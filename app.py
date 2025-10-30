# app_fixed.py - Final fixed extraction for NCC PDF with proper role boundaries
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import io, os, re, traceback
import pdfplumber

app = FastAPI(title="PDF Extractor - Fixed")

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
    return (
        "kontakter" in text_lower or
        ("navn" in text_lower and "firma" in text_lower and "telefon" in text_lower) or
        ("alex andersen" in text_lower) or
        ("allan lavrsen" in text_lower)
    )

def is_project_page(page_text: str) -> bool:
    """Check if this page contains project information"""
    text_lower = page_text.lower()
    return (
        "projekter" in text_lower or
        ("projektnavn" in text_lower and "budget" in text_lower) or
        ("renovering" in text_lower and "mio" in text_lower and "stadie" in text_lower)
    )

def fix_danish_name(name: str) -> str:
    """Fix common Danish name splitting issues"""
    if not name:
        return name
    
    # Specific fixes for names in this PDF
    fixes = {
        r'Ande\s*rsen': 'Andersen',
        r'Lavr\s*sen': 'Lavrsen', 
        r'Ka\s*mal': 'Kamal',
        r'Sch\s*ö\s*nau': 'Schönau',
        r'Eng\s*berg': 'Engberg',
        r'Berg\s+Rasmussen': 'Berg Rasmussen',
        r'Nørby\s+Hansen': 'Nørby Hansen',
        r'Dam\s+Jensen': 'Dam Jensen',
        r'Møller\s+Christensen': 'Møller Christensen',
    }
    
    for pattern, replacement in fixes.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    
    return name.strip()

def extract_contacts_fixed(pdf) -> List[Dict[str, str]]:
    """
    Extract contacts with proper role boundaries
    """
    all_contacts = []
    
    # Look specifically at pages 6-11 where contacts are located
    for page_num in range(5, min(11, len(pdf.pages))):
        page = pdf.pages[page_num]
        text = page.extract_text() or ""
        
        # Skip if not a contact page
        if not is_contact_page(text):
            continue
        
        # Split into lines
        lines = text.split('\n')
        
        # Process line by line
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for contact start: number (1-3 digits) followed by name
            match = re.match(r'^(\d{1,3})\s+([A-ZÆØÅ][a-zæøå]+(?:\s+[A-ZÆØÅ][a-zæøå\-\.]+)*)', line)
            
            if match:
                contact = {
                    "#": match.group(1),
                    "name": match.group(2),
                    "phones": [],
                    "role_lines": []
                }
                
                # Extract phones from current line
                contact["phones"] = re.findall(r'\b\d{8}\b', line)
                
                # Extract role from current line (after name and company)
                rest_of_line = line[match.end():]
                rest_of_line = re.sub(r'NCC\s*Danmark\s*A/?S', '', rest_of_line)
                rest_of_line = re.sub(r'\b\d{8}\b', '', rest_of_line)
                rest_of_line = rest_of_line.strip()
                
                # Only add if it contains role keywords
                if any(kw in rest_of_line for kw in ['Projektleder', 'Kontaktperson', 'entreprenør', 'Entr.']):
                    contact["role_lines"].append(rest_of_line)
                
                # Look at next lines for additional role/phone info
                j = i + 1
                while j < len(lines) and j <= i + 3:  # Only look 3 lines ahead max
                    next_line = lines[j].strip()
                    
                    # STOP if we hit another contact (starts with 1-3 digit number and name)
                    if re.match(r'^(\d{1,3})\s+[A-ZÆØÅ][a-zæøå]+\s+[A-ZÆØÅ]', next_line):
                        break
                    
                    # STOP if we see another contact number in the line
                    if re.search(r'\b\d{1,3}\s+[A-ZÆØÅ][a-zæøå]+\s+(NCC|Danmark)', next_line):
                        break
                    
                    # Check if line starts with 8-digit phone
                    if re.match(r'^(\d{8})\b', next_line):
                        phone = re.match(r'^(\d{8})\b', next_line).group(1)
                        if phone not in contact["phones"]:
                            contact["phones"].append(phone)
                        
                        # Rest might be role
                        rest = next_line[8:].strip()
                        if rest and any(kw in rest for kw in ['Projektleder', 'entreprenør', 'Entr.']):
                            contact["role_lines"].append(rest)
                    
                    # Check if it's a pure role line (contains role keywords)
                    elif any(kw in next_line for kw in ['Projektleder', 'Kontaktperson', 
                                                         'entreprenør', 'Entr.', 'Ingeniør']):
                        # But make sure it doesn't contain another contact's info
                        if not re.search(r'\b\d{1,3}\s+[A-ZÆØÅ]', next_line):
                            contact["role_lines"].append(next_line)
                    
                    j += 1
                
                # Process the contact
                contact["name"] = fix_danish_name(contact["name"])
                
                # Clean up role - join lines and remove duplicates
                role = " ".join(contact["role_lines"])
                role = re.sub(r'\s+', ' ', role)
                
                # Remove any accidental contact info from role
                role = re.sub(r'\d{1,3}\s+[A-ZÆØÅ][a-zæøå]+\s+[A-ZÆØÅ][a-zæøå]+.*', '', role)
                role = re.sub(r'NCC\s*Danmark\s*A/?S', '', role)
                role = role.strip()
                
                all_contacts.append({
                    "#": contact["#"],
                    "Navn": contact["name"],
                    "Telefon1": contact["phones"][0] if contact["phones"] else "",
                    "Telefon2": contact["phones"][1] if len(contact["phones"]) > 1 else "",
                    "Rolle": role
                })
                
                i = j - 1  # Continue from where we stopped
            
            i += 1
    
    return all_contacts

def extract_projects_simple(pdf) -> List[Dict[str, Any]]:
    """Simple project extraction"""
    projects = []
    
    for page in pdf.pages:
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
            # Process each row
            for row in table:
                if not row:
                    continue
                
                # Clean cells
                row = [str(c).strip() if c else "" for c in row]
                
                # Check if this is a project row (starts with number)
                if row[0] and re.match(r'^\d+$', row[0]):
                    # Create project with available columns
                    project = {
                        "#": row[0],
                        "Projektnavn": row[1] if len(row) > 1 else "",
                        "Roller": row[2] if len(row) > 2 else "",
                        "Region": row[3] if len(row) > 3 else "",
                        "Budget, kr.": row[4] if len(row) > 4 else "",
                        "Byggestart": row[5] if len(row) > 5 else "",
                        "Bæredygtighed": row[6] if len(row) > 6 else "",
                        "Seneste opdateringsdato": row[7] if len(row) > 7 else "",
                        "Stadie": row[8] if len(row) > 8 else ""
                    }
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
                projects = extract_projects_simple(pdf)
            
            if "contacts" in which_set:
                contacts = extract_contacts_fixed(pdf)

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
