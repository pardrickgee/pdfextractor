#!/usr/bin/env python3
"""
Improved Smart Byggefakta PDF Scraper
Better filtering and data quality validation
"""

from pathlib import Path
import sys
import json
import re
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    import camelot
except ImportError:
    print("ERROR: camelot-py not installed. Install with: pip install 'camelot-py[cv]'")
    sys.exit(1)

# ========================================
# Configuration
# ========================================

DEBUG = True

# Patterns for content classification
PHONE_PATTERN = re.compile(r'(?:\+45\s*)?(?:\d{2}\s*){3,4}\d{2,4}')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
MONEY_PATTERN = re.compile(r'(\d+(?:[.,]\d+)?)\s*(mio\.?|kr\.?|million|DKK)', re.I)
CVR_PATTERN = re.compile(r'CVR.*?(\d{8})', re.I)
DATE_PATTERN = re.compile(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b')
DANISH_DATE_PATTERN = re.compile(
    r'\b(?:januar|februar|marts|april|maj|juni|juli|august|'
    r'september|oktober|november|december)\s+\d{4}\b', re.I
)

# Section headers to skip
SECTION_HEADERS = {'kontakter', 'projekter', 'udbud', 'oplysninger', 'arkiv'}

# Common role/type indicators that should NOT be names
ROLE_KEYWORDS = {
    'entr.', 'entr', 'entreprenør', 'totalentreprenør', 'hovedentreprenør',
    'element', 'stål', 'beton', 'tømrer', 'snedker', 'el-entreprenør',
    'vvs', 'jord/kloak', 'fundering', 'belægning', 'facade', 'gulv',
    'tagdækning', 'vindue', 'døre', 'maler', 'privat', 'firma',
    'telefon', 'fax', 'e-mail', 'hjemmeside', 'cvr', 'rolle', 'navn'
}

# ========================================
# Data Models
# ========================================

@dataclass
class Contact:
    name: str = ""
    company: str = ""
    phones: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

@dataclass
class Project:
    name: str = ""
    roles: List[str] = field(default_factory=list)
    region: str = ""
    budget: str = ""
    start_date: str = ""
    stage: str = ""
    update_date: str = ""

@dataclass
class Tender:
    name: str = ""
    role: str = ""
    project_name: str = ""
    contact: str = ""
    docs_date: str = ""
    bid_date: str = ""

@dataclass
class CompanyInfo:
    name: str = ""
    cvr: str = ""
    phone: str = ""
    email: str = ""
    website: str = ""

@dataclass
class ParsedDocument:
    source_file: str
    company_info: CompanyInfo = field(default_factory=CompanyInfo)
    contacts: List[Contact] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    tenders: List[Tender] = field(default_factory=list)

# ========================================
# Utilities
# ========================================

def debug_print(msg: str, level: str = "INFO"):
    if DEBUG:
        print(f"[{level}] {msg}", file=sys.stderr)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_phones(text: str) -> List[str]:
    """Extract phone numbers"""
    return list(set(PHONE_PATTERN.findall(clean_text(text))))

def extract_emails(text: str) -> List[str]:
    """Extract email addresses"""
    return list(set(EMAIL_PATTERN.findall(clean_text(text))))

def extract_money(text: str) -> str:
    """Extract money amount"""
    match = MONEY_PATTERN.search(clean_text(text))
    return match.group(0) if match else ""

def split_multiline(text: str) -> List[str]:
    """Split multiline cell content"""
    parts = re.split(r'[\n•|]|  {2,}', clean_text(text))
    return [p.strip() for p in parts if p.strip()]

def is_header_row(row_text: str) -> bool:
    """Check if row is a table header"""
    row_lower = row_text.lower().strip()
    
    # Check for section headers
    if row_lower in SECTION_HEADERS:
        return True
    
    # Check for common header patterns
    header_patterns = [
        r'^navn\s*$',
        r'^telefon\s*$',
        r'^rolle\s*$',
        r'^firma.*navn',
        r'^projekt.*navn',
        r'^region\s*$',
        r'^budget\s*$',
        r'^#\s*$'
    ]
    
    for pattern in header_patterns:
        if re.search(pattern, row_lower):
            return True
    
    return False

def is_role_type_only(text: str) -> bool:
    """Check if text is just a role/type indicator, not a person/project name"""
    text_lower = text.lower().strip()
    
    # Check against role keywords
    for keyword in ROLE_KEYWORDS:
        if text_lower == keyword or text_lower.endswith(keyword):
            return True
    
    # Check if it's ONLY a role type (no actual name content)
    role_only_patterns = [
        r'^[A-Za-z/\-]+\s+entr\.?\s*$',
        r'^entr\.?\s*$',
        r'^element\s*$',
        r'^stål\s*$',
        r'^beton\s*$'
    ]
    
    for pattern in role_only_patterns:
        if re.search(pattern, text_lower, re.I):
            return True
    
    return False

def is_valid_contact(contact: Contact) -> bool:
    """Validate if contact has enough meaningful data"""
    # Must have a name
    if not contact.name or len(contact.name) < 2:
        return False
    
    # Name should not be a header or role keyword
    if is_header_row(contact.name) or is_role_type_only(contact.name):
        return False
    
    # Filter out row numbers
    if contact.name.isdigit():
        return False
    
    # Filter out company info field labels
    info_labels = ['id nr', 'firmatype', 'telefonnummer', 'e-mail adresse', 
                   'hjemmeside', 'cvr nr', 'fax']
    name_lower = contact.name.lower().strip()
    if any(label in name_lower for label in info_labels):
        return False
    
    # Must have at least one piece of contact info or company
    has_info = (
        contact.phones or 
        contact.emails or 
        (contact.company and len(contact.company) > 3 and not is_role_type_only(contact.company))
    )
    
    if not has_info:
        return False
    
    # Check for garbage patterns
    garbage_patterns = ['fax:', 'e-mail', 'telefon:', 'hjemmeside:', 'privat', 'cvr']
    name_lower = contact.name.lower()
    if any(g in name_lower for g in garbage_patterns):
        return False
    
    # Name should have at least one letter (not just numbers/symbols)
    if not any(c.isalpha() for c in contact.name):
        return False
    
    return True

def is_valid_project(project: Project) -> bool:
    """Validate if project has enough meaningful data"""
    # Must have a name
    if not project.name or len(project.name) < 5:
        return False
    
    # Name should not be a header or section title
    if is_header_row(project.name):
        return False
    
    # Name should not be just a role type
    if is_role_type_only(project.name):
        return False
    
    # Must have at least 2 other fields (region, budget, or dates)
    fields_filled = sum([
        bool(project.region),
        bool(project.budget),
        bool(project.start_date),
        bool(project.stage)
    ])
    
    if fields_filled < 2:
        return False
    
    return True

def is_valid_tender(tender: Tender) -> bool:
    """Validate if tender has enough meaningful data"""
    # Must have a name
    if not tender.name or len(tender.name) < 5:
        return False
    
    # Name should not be a header
    if is_header_row(tender.name):
        return False
    
    # Must have at least role or contact
    if not tender.role and not tender.contact:
        return False
    
    return True

# ========================================
# Table Classification
# ========================================

def classify_table(df: pd.DataFrame) -> str:
    """
    Classify table type based on column names and content
    Returns: 'contacts', 'projects', 'tenders', or 'unknown'
    """
    if df.empty:
        return 'unknown'
    
    # Check first 5 rows for section headers and column names
    sample_rows = min(5, len(df))
    sample = ' '.join(
        ' '.join(df.iloc[i].astype(str)) for i in range(sample_rows)
    ).lower()
    
    # Direct section detection
    if re.search(r'\bkontakter\b', sample) or (re.search(r'\bnavn\b', sample) and re.search(r'\btelefon\b', sample)):
        return 'contacts'
    
    if re.search(r'\bprojekter\b', sample) or re.search(r'\bprojektnavn\b', sample):
        return 'projects'
    
    if re.search(r'\budbud\b', sample) or re.search(r'\budbudsnavn\b', sample):
        return 'tenders'
    
    # Fallback to keyword scoring
    contact_score = 0
    if re.search(r'\bnavn\b', sample):
        contact_score += 1
    if re.search(r'\btelefon\b', sample):
        contact_score += 2
    if re.search(r'\brolle\b', sample):
        contact_score += 1
    
    project_score = 0
    if 'budget' in sample:
        project_score += 2
    if 'region' in sample:
        project_score += 2
    if 'byggestart' in sample:
        project_score += 2
    
    tender_score = 0
    if 'licitationsdato' in sample:
        tender_score += 3
    if 'udbudsrolle' in sample:
        tender_score += 2
    
    # Check for characteristic patterns
    if PHONE_PATTERN.search(sample):
        contact_score += 1
    if EMAIL_PATTERN.search(sample):
        contact_score += 1
    if MONEY_PATTERN.search(sample):
        project_score += 1
    
    scores = {'contacts': contact_score, 'projects': project_score, 'tenders': tender_score}
    max_score = max(scores.values())
    
    if max_score < 2:
        return 'unknown'
    
    return max(scores.items(), key=lambda x: x[1])[0]

# ========================================
# Contact Extraction
# ========================================

def merge_continuation_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rows where first column is empty (continuation rows)
    with the previous row that has data in first column
    """
    if df.empty:
        return df
    
    merged_rows = []
    current_row = None
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        first_col = clean_text(str(row.iloc[0]))
        
        # Check if this is a continuation row (first column is empty or whitespace)
        is_continuation = not first_col or first_col == '' or first_col == 'nan'
        
        if is_continuation and current_row is not None:
            # Merge with current row
            for col_idx in range(len(row)):
                cell_value = clean_text(str(row.iloc[col_idx]))
                if cell_value and cell_value != 'nan':
                    # Append to current row
                    current_val = clean_text(str(current_row.iloc[col_idx]))
                    if current_val and current_val != 'nan':
                        current_row.iloc[col_idx] = current_val + '\n' + cell_value
                    else:
                        current_row.iloc[col_idx] = cell_value
        else:
            # Start new row
            if current_row is not None:
                merged_rows.append(current_row)
            current_row = row.copy()
    
    # Add last row
    if current_row is not None:
        merged_rows.append(current_row)
    
    if merged_rows:
        return pd.DataFrame(merged_rows).reset_index(drop=True)
    return df

def extract_contacts_from_df(df: pd.DataFrame) -> List[Contact]:
    """Extract contacts from DataFrame with improved filtering"""
    contacts = []
    
    debug_print(f"Processing {len(df)} contact rows")
    
    # First, merge continuation rows
    df = merge_continuation_rows(df)
    debug_print(f"After merging continuations: {len(df)} rows")
    
    # Find where data actually starts (skip section title and headers)
    data_start = 0
    for i in range(min(10, len(df))):
        row_text = ' '.join(df.iloc[i].astype(str)).lower()
        if is_header_row(row_text):
            data_start = i + 1
            continue
        # If we found a row with actual name-like content, start here
        if len(row_text) > 5 and not is_header_row(row_text):
            break
    
    debug_print(f"Data starts at row {data_start}")
    
    for idx in range(data_start, len(df)):
        row = df.iloc[idx]
        row_list = [clean_text(str(cell)) for cell in row]
        
        # Skip empty rows
        if not any(cell for cell in row_list if cell and cell != 'nan'):
            continue
        
        # Skip if first column looks like a header or section
        if row_list and is_header_row(row_list[0]):
            continue
        
        contact = Contact()
        
        # Extract name - skip first column if it's just a number (row number)
        name_start_col = 0
        if row_list and row_list[0].isdigit():
            name_start_col = 1  # Skip row number column
        
        # Extract name from appropriate column
        for i in range(name_start_col, min(len(row_list), 3)):
            cell = row_list[i]
            if cell and cell != 'nan' and not is_role_type_only(cell):
                # Check if this looks like a name (not a phone, email, or role indicator)
                if not PHONE_PATTERN.match(cell) and not EMAIL_PATTERN.match(cell):
                    # Additional check: should have at least 2 parts (first + last name)
                    # or be longer than just initials
                    if ' ' in cell or len(cell) > 5:
                        contact.name = cell
                        break
        
        # Extract company
        # Look for company name (usually in first few columns, contains A/S or ApS)
        for i in range(name_start_col, min(len(row_list), 4)):
            cell = row_list[i]
            if cell and cell != 'nan' and cell != contact.name:
                # Check if it looks like a company
                if 'A/S' in cell or 'ApS' in cell or (len(cell) > 10 and not is_role_type_only(cell)):
                    contact.company = cell
                    break
        
        # Extract all phones and emails from entire row
        full_text = ' | '.join(row_list)
        contact.phones = extract_phones(full_text)
        contact.emails = extract_emails(full_text)
        
        # Extract roles from cells that look like roles
        for cell in row_list:
            if cell and cell != 'nan' and cell not in [contact.name, contact.company]:
                # Check if this cell contains role keywords
                cell_lower = cell.lower()
                if any(role in cell_lower for role in ['projektleder', 'kontaktperson', 'indkøber', 'byggeleder', 'entr']):
                    parts = split_multiline(cell)
                    contact.roles.extend(parts)
        
        # Validate and add
        if is_valid_contact(contact):
            contacts.append(contact)
            debug_print(f"  ✓ {contact.name} - {contact.company or '(no company)'}")
        else:
            debug_print(f"  ✗ Filtered out: {contact.name}", "WARN")
    
    return contacts

# ========================================
# Project Extraction
# ========================================

def extract_projects_from_df(df: pd.DataFrame) -> List[Project]:
    """Extract projects from DataFrame with improved filtering"""
    projects = []
    
    debug_print(f"Processing {len(df)} project rows")
    
    # First, merge continuation rows
    df = merge_continuation_rows(df)
    debug_print(f"After merging continuations: {len(df)} rows")
    
    # Find where data starts
    data_start = 0
    for i in range(min(10, len(df))):
        row_text = ' '.join(df.iloc[i].astype(str)).lower()
        if is_header_row(row_text):
            data_start = i + 1
            continue
        if len(row_text) > 10:
            break
    
    debug_print(f"Data starts at row {data_start}")
    
    seen_projects = set()  # Track duplicates
    
    for idx in range(data_start, len(df)):
        row = df.iloc[idx]
        row_list = [clean_text(str(cell)) for cell in row]
        
        # Skip empty rows
        if not any(cell for cell in row_list if cell and cell != 'nan'):
            continue
        
        # Skip if row looks like headers
        row_text = ' '.join(row_list)
        if is_header_row(row_text):
            continue
        
        project = Project()
        
        # Extract project name (first substantial cell that's not a role type)
        for cell in row_list:
            if cell and cell != 'nan' and len(cell) > 5:
                if not is_role_type_only(cell):
                    project.name = cell
                    break
        
        # Skip if no valid name found
        if not project.name:
            continue
        
        # Skip duplicates
        if project.name in seen_projects:
            debug_print(f"  ✗ Duplicate: {project.name}", "WARN")
            continue
        
        # Extract other fields
        full_text = ' | '.join(row_list)
        
        # Region (look for specific regions)
        regions = ['hovedstaden', 'sjælland', 'syddanmark', 'midtjylland', 'nordjylland']
        for region in regions:
            if region in full_text.lower():
                project.region = region.capitalize()
                break
        
        # Budget (look for money)
        project.budget = extract_money(full_text)
        
        # Dates
        danish_dates = DANISH_DATE_PATTERN.findall(full_text)
        if danish_dates:
            project.start_date = danish_dates[0]
        
        # Stage (look for common stage keywords)
        stage_keywords = ['udførelsesproces', 'projektering', 'idéfase', 'afsluttet']
        for stage in stage_keywords:
            if stage in full_text.lower():
                project.stage = stage.capitalize()
                break
        
        # Roles (extract any role-like text)
        for cell in row_list:
            if cell and cell != 'nan' and cell != project.name:
                if any(role in cell.lower() for role in ['entr', 'projektleder', 'kontaktperson']):
                    parts = split_multiline(cell)
                    project.roles.extend(parts)
        
        # Validate and add
        if is_valid_project(project):
            projects.append(project)
            seen_projects.add(project.name)
            debug_print(f"  ✓ {project.name[:60]}... - {project.budget} - {project.region}")
        else:
            debug_print(f"  ✗ Filtered out: {project.name[:60]}", "WARN")
    
    return projects

# ========================================
# Tender Extraction
# ========================================

def extract_tenders_from_df(df: pd.DataFrame) -> List[Tender]:
    """Extract tenders from DataFrame with improved filtering"""
    tenders = []
    
    debug_print(f"Processing {len(df)} tender rows")
    
    # First, merge continuation rows
    df = merge_continuation_rows(df)
    debug_print(f"After merging continuations: {len(df)} rows")
    
    # Find where data starts
    data_start = 0
    for i in range(min(10, len(df))):
        row_text = ' '.join(df.iloc[i].astype(str)).lower()
        if is_header_row(row_text):
            data_start = i + 1
            continue
        if len(row_text) > 10:
            break
    
    debug_print(f"Data starts at row {data_start}")
    
    for idx in range(data_start, len(df)):
        row = df.iloc[idx]
        row_list = [clean_text(str(cell)) for cell in row]
        
        # Skip empty rows
        if not any(cell for cell in row_list if cell and cell != 'nan'):
            continue
        
        # Skip if row looks like headers
        row_text = ' '.join(row_list)
        if is_header_row(row_text):
            continue
        
        tender = Tender()
        
        # Expected columns: #, Udbudsnavn, Udbudsrolle, Projektnavn, Kontakt, Første dag, Licitationsdato
        if len(row_list) > 1:
            name = row_list[1]
            if name and name != 'nan' and not name.isdigit() and len(name) > 5:
                tender.name = name
        
        if len(row_list) > 2:
            role = row_list[2]
            if role and role != 'nan':
                tender.role = role
        
        if len(row_list) > 3:
            project = row_list[3]
            if project and project != 'nan' and len(project) > 5:
                tender.project_name = project
        
        if len(row_list) > 4:
            contact = row_list[4]
            if contact and contact != 'nan':
                tender.contact = contact
        
        # Extract dates from remaining columns
        full_text = ' | '.join(row_list)
        dates = DATE_PATTERN.findall(full_text)
        if dates:
            tender.docs_date = dates[0] if len(dates) > 0 else ""
            tender.bid_date = dates[1] if len(dates) > 1 else ""
        
        # Validate and add
        if is_valid_tender(tender):
            tenders.append(tender)
            debug_print(f"  ✓ {tender.name[:60]}... - {tender.role}")
        else:
            debug_print(f"  ✗ Filtered out: {tender.name[:40] if tender.name else '(no name)'}", "WARN")
    
    return tenders

# ========================================
# Company Info Extraction
# ========================================

def extract_company_info_from_text(text: str) -> CompanyInfo:
    """Extract company info from raw text"""
    info = CompanyInfo()
    
    # CVR number
    cvr_match = CVR_PATTERN.search(text)
    if cvr_match:
        info.cvr = cvr_match.group(1)
    
    # Contact details
    phones = extract_phones(text)
    if phones:
        info.phone = phones[0]
    
    emails = extract_emails(text)
    if emails:
        info.email = emails[0]
    
    # Website
    url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
    url_match = url_pattern.search(text)
    if url_match:
        info.website = url_match.group(0).rstrip('.,;')
    
    # Try to find company name (usually bold or near top)
    lines = text.split('\n')
    for i, line in enumerate(lines[:15]):
        line = line.strip()
        # Skip very short lines, URLs, common headers
        if len(line) < 3 or line.lower() in ['oplysninger', 'smart', 'byggefakta']:
            continue
        # Company name usually has certain patterns
        if len(line) > 5 and len(line) < 50 and ('A/S' in line or 'ApS' in line):
            info.name = line
            break
    
    return info

# ========================================
# Main Parser
# ========================================

def parse_pdf(pdf_path: str) -> ParsedDocument:
    """
    Parse Smart Byggefakta PDF using Camelot
    """
    debug_print(f"\n{'='*80}")
    debug_print(f"Parsing: {pdf_path}")
    debug_print(f"{'='*80}")
    
    doc = ParsedDocument(source_file=Path(pdf_path).name)
    
    try:
        # Extract all tables from all pages using 'stream' flavor
        debug_print("Extracting tables with Camelot (stream mode)...")
        tables = camelot.read_pdf(
            pdf_path,
            pages='all',
            flavor='stream',
            edge_tol=50,
            row_tol=10,
        )
        
        debug_print(f"Found {len(tables)} tables across all pages")
        
        # Process each table
        for i, table in enumerate(tables):
            df = table.df
            
            if df.empty:
                continue
            
            debug_print(f"\n--- Table {i+1} (Shape: {df.shape}) ---")
            
            # Classify table
            table_type = classify_table(df)
            debug_print(f"Classified as: {table_type}")
            
            # Extract data based on type
            if table_type == 'contacts':
                contacts = extract_contacts_from_df(df)
                doc.contacts.extend(contacts)
            
            elif table_type == 'projects':
                projects = extract_projects_from_df(df)
                doc.projects.extend(projects)
            
            elif table_type == 'tenders':
                tenders = extract_tenders_from_df(df)
                doc.tenders.extend(tenders)
            
            else:
                debug_print(f"Skipping unknown table type", "WARN")
        
        # Extract company info from first page
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                first_text = pdf.pages[0].extract_text() or ""
                doc.company_info = extract_company_info_from_text(first_text)
        except Exception as e:
            debug_print(f"Could not extract company info: {e}", "WARN")
    
    except Exception as e:
        debug_print(f"Failed to parse PDF: {e}", "ERROR")
        import traceback
        debug_print(traceback.format_exc(), "ERROR")
    
    # Summary
    debug_print(f"\n{'='*80}")
    debug_print(f"✓ Extraction complete:")
    debug_print(f"  Company: {doc.company_info.name or '(not found)'}")
    debug_print(f"  CVR: {doc.company_info.cvr or '(not found)'}")
    debug_print(f"  Contacts: {len(doc.contacts)}")
    debug_print(f"  Projects: {len(doc.projects)}")
    debug_print(f"  Tenders: {len(doc.tenders)}")
    debug_print(f"{'='*80}\n")
    
    return doc

# ========================================
# CLI
# ========================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Improved Smart Byggefakta PDF Scraper',
        epilog='Example: python smart_scraper_improved.py file1.pdf -o output.json'
    )
    
    parser.add_argument('files', nargs='+', help='PDF file(s) to parse')
    parser.add_argument('-o', '--output', help='Output JSON file (default: print to stdout)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON output')
    
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = not args.no_debug
    
    # Parse all files
    results = []
    for pdf_file in args.files:
        if not Path(pdf_file).exists():
            print(f"ERROR: File not found: {pdf_file}", file=sys.stderr)
            continue
        
        try:
            doc = parse_pdf(pdf_file)
            results.append(doc)
        except Exception as e:
            print(f"ERROR parsing {pdf_file}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # Convert to JSON
    output = [asdict(doc) for doc in results]
    
    indent = 2 if args.pretty else None
    json_output = json.dumps(output, ensure_ascii=False, indent=indent)
    
    # Output
    if args.output:
        Path(args.output).write_text(json_output, encoding='utf-8')
        print(f"✓ Wrote {args.output}")
    else:
        print(json_output)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
