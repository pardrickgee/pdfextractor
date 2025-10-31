#!/usr/bin/env python3
"""
Smart Byggefakta PDF Scraper V3 - FINAL COMPREHENSIVE VERSION
Complete extraction using both Camelot and pdfplumber
"""

from pathlib import Path
import sys
import json
import re
from dataclasses import dataclass, asdict, field
from typing import List, Optional
import pandas as pd

try:
    import camelot
    import pdfplumber
except ImportError as e:
    print(f"ERROR: {e}\nInstall: pip install 'camelot-py[cv]' pdfplumber pandas")
    sys.exit(1)

DEBUG = True

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
    number: str = ""
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
    fax: str = ""
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
    if pd.isna(text) or not text:
        return ""
    return str(text).strip().replace('\n', ' ')

def extract_cvr(text: str) -> str:
    match = re.search(r'CVR.*?:?\s*(\d{8})', text, re.I)
    return match.group(1) if match else ""

def extract_phones(text: str, exclude_cvr: str = "") -> List[str]:
    phones = []
    pattern = re.compile(r'(?:\+45\s*)?(\d{8}|\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2})')
    for match in pattern.finditer(text):
        phone = match.group(1).replace(' ', '').replace('-', '')
        if len(phone) == 8 and phone != exclude_cvr:
            phones.append(phone)
    return list(set(phones))

def extract_emails(text: str) -> List[str]:
    pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return list(set(pattern.findall(text)))

# ========================================
# Text-based Section Extraction
# ========================================

def extract_contacts_from_text(text: str, cvr: str) -> List[Contact]:
    """Extract contacts from KONTAKTER text section"""
    contacts = []
    
    # Find KONTAKTER section
    match = re.search(r'KONTAKTER(.+?)(?:UDBUD|Hubexo|$)', text, re.DOTALL | re.I)
    if not match:
        return contacts
    
    section = match.group(1)
    lines = section.split('\n')
    
    current_contact = None
    for line in lines:
        line = line.strip()
        if not line or line == '#' or 'Navn' in line or 'Firma' in line:
            continue
        
        # Check if this is a new contact entry (starts with number)
        if re.match(r'^\d+\s+', line):
            # Save previous contact
            if current_contact and current_contact.name:
                contacts.append(current_contact)
            
            # Start new contact
            current_contact = Contact()
            
            # Parse: "1 Jack Johansen Clever A/S 82303030 Kontaktperson.Bygherre"
            parts = line.split()
            if len(parts) >= 2:
                # Skip number, extract name (next 2-3 parts)
                name_parts = []
                company_parts = []
                in_company = False
                
                for i, part in enumerate(parts[1:], 1):
                    if 'A/S' in part or 'ApS' in part:
                        in_company = True
                    
                    if not in_company and not part.isdigit() and '@' not in part:
                        # Check if it's a role keyword
                        if not any(kw in part.lower() for kw in ['entr', 'projektleder', 'kontakt', 'bygherre']):
                            name_parts.append(part)
                    elif in_company:
                        company_parts.append(part)
                        if 'A/S' in part or 'ApS' in part:
                            break
                
                current_contact.name = ' '.join(name_parts)
                current_contact.company = ' '.join(company_parts)
                current_contact.phones = extract_phones(line, cvr)
                current_contact.emails = extract_emails(line)
                
                # Extract roles
                role_part = line.split(current_contact.company)[-1] if current_contact.company else line
                for role_kw in ['Kontaktperson', 'Bygherre', 'Projektleder', 'Entreprenør']:
                    if role_kw in role_part:
                        current_contact.roles.append(role_kw)
        
        # Check if this line has additional phone/role info for current contact
        elif current_contact and (line.isdigit() or any(kw in line for kw in ['Kontakt', 'Byg'])):
            if line.isdigit() and len(line) == 8:
                if line not in current_contact.phones:
                    current_contact.phones.append(line)
            else:
                for role in ['Kontaktperson', 'Bygherre', 'Projektleder']:
                    if role in line and role not in current_contact.roles:
                        current_contact.roles.append(role)
    
    # Add last contact
    if current_contact and current_contact.name:
        contacts.append(current_contact)
    
    debug_print(f"Extracted {len(contacts)} contacts from text")
    return contacts

def extract_tenders_from_text(text: str) -> List[Tender]:
    """Extract tenders from UDBUD text section"""
    tenders = []
    
    # Find UDBUD section
    match = re.search(r'UDBUD(.+?)(?:Hubexo|$)', text, re.DOTALL | re.I)
    if not match:
        return tenders
    
    section = match.group(1)
    lines = section.split('\n')
    
    current_tender = None
    for line in lines:
        line = line.strip()
        if not line or 'Udbudsnavn' in line:
            continue
        
        # Check if new tender entry (starts with number or "ARKIV")
        if re.match(r'^\d+\s+', line) or line.startswith('ARKIV'):
            if current_tender and current_tender.name:
                tenders.append(current_tender)
            
            current_tender = Tender()
            
            # Parse tender line
            parts = line.split()
            if len(parts) > 1:
                # Skip number if present
                start_idx = 1 if parts[0].isdigit() else 0
                current_tender.name = ' '.join(parts[start_idx:start_idx+5])  # Rough estimate
                
                # Look for role keywords
                for part in parts:
                    if 'entreprenør' in part.lower():
                        current_tender.role = part
                        break
    
    if current_tender and current_tender.name:
        tenders.append(current_tender)
    
    debug_print(f"Extracted {len(tenders)} tenders from text")
    return tenders

# ========================================
# Camelot-based Project Extraction
# ========================================

def extract_projects_from_tables(tables) -> List[Project]:
    """Extract projects from Camelot tables"""
    all_projects = []
    seen = set()
    
    for i, table in enumerate(tables):
        df = table.df
        if df.empty:
            continue
        
        # Check if this is a projects table
        sample = ' '.join(df.iloc[0].astype(str)).lower()
        if 'projekter' not in sample and 'projektnavn' not in sample:
            continue
        
        debug_print(f"Processing projects table {i+1}")
        
        # Find data start
        data_start = 1
        for j in range(min(5, len(df))):
            row_text = ' '.join(df.iloc[j].astype(str)).lower()
            if 'projektnavn' in row_text or '#' in row_text:
                data_start = j + 1
                break
        
        for idx in range(data_start, len(df)):
            row = df.iloc[idx]
            
            project = Project()
            
            # Extract fields from row
            row_list = [clean_text(str(cell)) for cell in row]
            
            # Number (first column if digit)
            if row_list and row_list[0].isdigit():
                project.number = row_list[0]
            
            # Name (first substantial text column)
            for cell in row_list[1:5]:
                if cell and len(cell) > 10 and not cell.isdigit():
                    project.name = cell
                    break
            
            if not project.name:
                continue
            
            # Other fields from row text
            row_text = ' | '.join(row_list)
            
            # Region
            for region in ['Hovedstaden', 'Sjælland', 'Syddanmark', 'Midtjylland', 'Nordjylland']:
                if region.lower() in row_text.lower():
                    project.region = region
                    break
            
            # Budget
            money_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:mio|mia)', row_text, re.I)
            if money_match:
                project.budget = money_match.group(0)
            
            # Stage
            for stage in ['Udførelsesproces', 'Udbudsproces', 'Projekteringsproces', 'Planlægningsproces']:
                if stage.lower() in row_text.lower():
                    project.stage = stage
                    break
            
            # Dates
            date_matches = re.findall(r'(Januar|Februar|Marts|April|Maj|Juni|Juli|August|September|Oktober|November|December)\s+(\d{4})', row_text, re.I)
            if date_matches:
                project.start_date = f"{date_matches[0][0]} {date_matches[0][1]}"
                if len(date_matches) > 1:
                    project.update_date = f"{date_matches[1][0]} {date_matches[1][1]}"
            
            # Roles
            for cell in row_list:
                if 'entr' in cell.lower() or 'bygherre' in cell.lower():
                    project.roles.append(cell)
            
            # Deduplicate
            key = (project.name, project.budget)
            if key not in seen and project.name:
                all_projects.append(project)
                seen.add(key)
    
    debug_print(f"Total unique projects: {len(all_projects)}")
    return all_projects

# ========================================
# Company Info Extraction
# ========================================

def extract_company_info(text: str) -> CompanyInfo:
    """Extract company info from first page"""
    info = CompanyInfo()
    
    lines = text.split('\n')
    
    for line in lines[:20]:
        line = line.strip()
        line_lower = line.lower()
        
        if 'cvr' in line_lower and 'nr' in line_lower:
            info.cvr = extract_cvr(line)
        elif line_lower.startswith('telefonnummer'):
            phones = extract_phones(line, info.cvr)
            if phones:
                info.phone = phones[0]
        elif line_lower.startswith('fax'):
            match = re.search(r'(\d{8})', line)
            if match:
                info.fax = match.group(1)
        elif 'e-mail' in line_lower:
            emails = extract_emails(line)
            if emails:
                info.email = emails[0]
        elif 'hjemmeside' in line_lower or 'website' in line_lower:
            match = re.search(r'https?://[^\s]+', line)
            if match:
                info.website = match.group(0)
    
    # Company name
    for line in lines[:15]:
        if ('A/S' in line or 'ApS' in line) and 5 < len(line) < 50:
            if 'http' not in line and 'oplysninger' not in line.lower():
                info.name = line.strip()
                break
    
    return info

# ========================================
# Main Parser V3
# ========================================

def parse_pdf_v3(pdf_path: str) -> ParsedDocument:
    """
    Final comprehensive parsing using both Camelot and pdfplumber
    """
    debug_print(f"\n{'='*80}")
    debug_print(f"Parsing (V3 - Comprehensive): {pdf_path}")
    debug_print(f"{'='*80}")
    
    doc = ParsedDocument(source_file=Path(pdf_path).name)
    
    try:
        # Extract all text from all pages using pdfplumber
        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                all_text += (page.extract_text() or "") + "\n"
        
        # Extract company info
        doc.company_info = extract_company_info(all_text)
        
        # Extract contacts from text
        doc.contacts = extract_contacts_from_text(all_text, doc.company_info.cvr)
        
        # Extract tenders from text
        doc.tenders = extract_tenders_from_text(all_text)
        
        # Extract projects using Camelot (better for large tables)
        debug_print("Extracting project tables with Camelot...")
        tables = camelot.read_pdf(
            pdf_path,
            pages='all',
            flavor='stream',
            edge_tol=50,
            row_tol=10,
        )
        
        doc.projects = extract_projects_from_tables(tables)
        
    except Exception as e:
        debug_print(f"ERROR: {e}", "ERROR")
        import traceback
        debug_print(traceback.format_exc(), "ERROR")
    
    # Summary
    debug_print(f"\n{'='*80}")
    debug_print(f"✓ FINAL EXTRACTION:")
    debug_print(f"  Company: {doc.company_info.name}")
    debug_print(f"  CVR: {doc.company_info.cvr}")
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
        description='Smart Byggefakta PDF Scraper V3 - FINAL',
    )
    
    parser.add_argument('files', nargs='+', help='PDF file(s) to parse')
    parser.add_argument('-o', '--output', help='Output JSON file')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON')
    
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = not args.no_debug
    
    results = []
    for pdf_file in args.files:
        if not Path(pdf_file).exists():
            print(f"ERROR: File not found: {pdf_file}", file=sys.stderr)
            continue
        
        try:
            doc = parse_pdf_v3(pdf_file)
            results.append(doc)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
    
    output = [asdict(doc) for doc in results]
    json_str = json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None)
    
    if args.output:
        Path(args.output).write_text(json_str, encoding='utf-8')
        print(f"✓ Wrote {args.output}")
    else:
        print(json_str)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
