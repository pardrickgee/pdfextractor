#!/usr/bin/env python3
# app_refactor.py — robust, layout-agnostic Smart Byggefakta scraper
# Usage:
#   python app_refactor.py input1.pdf input2.pdf -o output.json
#
# Requires: pdfplumber (`pip install pdfplumber`)
# Optional: pytesseract/layoutparser for advanced table detection (not required for base run)

from pathlib import Path
import sys
import json
import re
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

try:
    import pdfplumber
except Exception as e:
    pdfplumber = None
    PDFPLUMBER_IMPORT_ERROR = str(e)
else:
    PDFPLUMBER_IMPORT_ERROR = None

# -----------------------------
# Config
# -----------------------------

SECTION_PATTERNS = [
    r"\\bKONTAKTER\\b", r"\\bKONTAKT\\b",
    r"\\bPROJEKTER\\b", r"\\bPROJEKT\\b",
    r"\\bUDBUD\\b",
    r"\\bOPLYSNINGER\\b", r"\\bFAKTA\\b", r"\\bBESKRIVELSE\\b"
]

PHONE_RE = re.compile(r"(?:\\+45\\s*)?\\b\\d{2}\\s?\\d{2}\\s?\\d{2}\\s?\\d{2}\\b")
EMAIL_RE = re.compile(r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b")

# x/y tolerances in points
X_ALIGN_TOL = 8
Y_LINE_TOL = 3
ROW_GAP_TOL = 8
ALIGN_RATIO_MIN = 0.7

BIN_WIDTH = 22
MIN_BIN_COUNT = 6

# -----------------------------
# Data models
# -----------------------------

@dataclass
class Contact:
    name: str = ""
    company: str = ""
    phones: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    raw_row: Dict[int, str] = field(default_factory=dict)

@dataclass
class ProjectRow:
    title: str = ""
    roles: List[str] = field(default_factory=list)
    region: str = ""
    budget_kr: str = ""
    stage: str = ""
    raw_row: Dict[int, str] = field(default_factory=dict)

@dataclass
class TenderRow:
    title: str = ""
    role: str = ""
    contact: str = ""
    first_docs_date: str = ""
    bid_date: str = ""
    raw_row: Dict[int, str] = field(default_factory=dict)

@dataclass
class ParsedDoc:
    source_file: str
    source_company: Optional[str] = None
    contacts: List[Contact] = field(default_factory=list)
    projects: List[ProjectRow] = field(default_factory=list)
    tenders: List[TenderRow] = field(default_factory=list)

# -----------------------------
# Helpers
# -----------------------------

def _find_header(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in SECTION_PATTERNS:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(0).upper()
    return None

def _xcenter(w: Dict[str, Any]) -> float:
    return (w["x0"] + w["x1"]) / 2.0

def _infer_column_spans(words: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    if not words:
        return []
    xs = sorted(_xcenter(w) for w in words)
    if not xs:
        return []
    bins = {}
    for x in xs:
        b = int(x // BIN_WIDTH)
        bins[b] = bins.get(b, 0) + 1
    peak_bins = [b for b, c in bins.items() if c >= MIN_BIN_COUNT]
    if not peak_bins:
        top = sorted(bins.items(), key=lambda kv: kv[1], reverse=True)[:4]
        peak_bins = [b for b, _ in top]
    peak_bins = sorted(set(peak_bins))
    merged = []
    for b in peak_bins:
        if not merged:
            merged.append([b, b])
        else:
            if b <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([b, b])
    spans = []
    pad = BIN_WIDTH * 0.45
    for b0, b1 in merged:
        x_left = b0 * BIN_WIDTH - pad
        x_right = (b1 + 1) * BIN_WIDTH + pad
        spans.append((x_left, x_right))
    spans.sort(key=lambda ab: ab[0])
    return spans

def _align_to_spans(words: List[Dict[str, Any]], spans: List[Tuple[float, float]], tol: float = X_ALIGN_TOL) -> bool:
    if not spans or not words:
        return False
    total = len(words)
    in_bins = 0
    centers = [ (a+b)/2.0 for a,b in spans ]
    for w in words:
        xc = _xcenter(w)
        if any(abs(xc - c) <= max(tol, BIN_WIDTH*0.4) for c in centers):
            in_bins += 1
    ratio = in_bins / max(1, total)
    return ratio >= ALIGN_RATIO_MIN

def _group_words_into_lines(words: List[Dict[str, Any]], y_tol: float = Y_LINE_TOL) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    current = [words_sorted[0]]
    for w in words_sorted[1:]:
        if abs(w["top"] - current[-1]["top"]) <= y_tol:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda t: t["x0"]))
            current = [w]
    lines.append(sorted(current, key=lambda t: t["x0"]))
    return lines

def _assign_line_to_columns(line: List[Dict[str, Any]], spans: List[Tuple[float, float]]) -> Dict[int, str]:
    cols: Dict[int, List[str]] = {}
    if not spans:
        text = " ".join(w["text"] for w in line)
        return {0: text}
    centers = [ (a+b)/2.0 for a,b in spans ]
    for w in line:
        xc = _xcenter(w)
        idx = min(range(len(centers)), key=lambda i: abs(xc - centers[i]))
        cols.setdefault(idx, []).append(w["text"])
    return {i: " ".join(tokens) for i, tokens in cols.items()}

def _merge_lines_into_rows(lines: List[List[Dict[str, Any]]], spans: List[Tuple[float, float]]) -> List[Dict[int, str]]:
    rows: List[Dict[int, str]] = []
    current: Dict[int, str] = {}
    last_y = None
    for line in lines:
        assigned = _assign_line_to_columns(line, spans)
        assigned = {k: v.strip() for k, v in assigned.items() if v and v.strip()}
        first_col_has_text = (0 in assigned and assigned[0].strip() != "")
        y_top = min(w["top"] for w in line)
        start_new = False
        if not current:
            start_new = True
        elif first_col_has_text:
            start_new = True
        elif last_y is not None and (y_top - last_y) > ROW_GAP_TOL and assigned:
            start_new = True
        if start_new:
            if current:
                rows.append(current)
            current = dict(assigned)
        else:
            for ci, txt in assigned.items():
                if ci in current:
                    current[ci] = (current[ci] + " " + txt).strip()
                else:
                    current[ci] = txt
        last_y = y_top
    if current:
        rows.append(current)
    return rows

# -----------------------------
# Section handlers
# -----------------------------

def _extract_contacts(rows: List[Dict[int, str]]) -> List[Contact]:
    out: List[Contact] = []
    for r in rows:
        name = (r.get(0) or "").strip()
        company = (r.get(1) or "").strip()
        all_text = " | ".join(r.get(i, "") for i in sorted(r.keys()))
        phones = sorted(set(PHONE_RE.findall(all_text)))
        emails = sorted(set(EMAIL_RE.findall(all_text)))
        rightmost_idx = max(r.keys())
        rightmost_text = r.get(rightmost_idx, "")
        roles = [s.strip(" .;:") for s in re.split(r"[\\n|/•]|  {2,}", rightmost_text) if s.strip()]
        if not (name or company or phones or emails or roles):
            continue
        out.append(Contact(name=name, company=company, phones=phones, emails=emails, roles=roles, raw_row=r))
    return out

def _extract_projects(rows: List[Dict[int, str]]) -> List[ProjectRow]:
    out: List[ProjectRow] = []
    for r in rows:
        cells = [r.get(i, "") for i in sorted(r.keys())]
        text = " | ".join(cells)
        title = cells[0].strip() if cells else ""
        region = ""
        budget = ""
        stage = ""
        m = re.search(r"(\\d[\\d\\. ]+)\\s*(kr|DKK)", text, flags=re.I)
        if m:
            budget = m.group(0)
        if re.search(r"(fase|stage|status)\\s*[:\\-]?\\s*(\\w+)", text, flags=re.I):
            stage = re.sub(r".*?(fase|stage|status)\\s*[:\\-]?\\s*", "", text, flags=re.I)
        out.append(ProjectRow(title=title, region=region, budget_kr=budget, stage=stage, raw_row=r))
    return out

def _extract_tenders(rows: List[Dict[int, str]]) -> List[TenderRow]:
    out: List[TenderRow] = []
    for r in rows:
        cells = [r.get(i, "") for i in sorted(r.keys())]
        title = cells[0].strip() if cells else ""
        role = cells[1].strip() if len(cells) > 1 else ""
        contact = cells[2].strip() if len(cells) > 2 else ""
        dates_text = " ".join(cells)
        m1 = re.findall(r"\\b\\d{1,2}[./-]\\d{1,2}[./-]\\d{2,4}\\b", dates_text)
        first_docs_date = m1[0] if m1 else ""
        bid_date = m1[1] if len(m1) > 1 else ""
        out.append(TenderRow(title=title, role=role, contact=contact, first_docs_date=first_docs_date, bid_date=bid_date, raw_row=r))
    return out

# -----------------------------
# Core parser
# -----------------------------

def parse_pdf(path: str) -> ParsedDoc:
    if pdfplumber is None:
        raise RuntimeError(f"pdfplumber import failed: {PDFPLUMBER_IMPORT_ERROR}")
    parsed = ParsedDoc(source_file=Path(path).name)
    with pdfplumber.open(path) as pdf:
        current_section: Optional[str] = None
        current_spans: List[Tuple[float, float]] = []
        for page in pdf.pages:
            try:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                text = ""
            try:
                words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            except Exception:
                words = []
            header = _find_header(text)
            if header:
                current_section = header
                current_spans = _infer_column_spans(words)
            elif current_section and _align_to_spans(words, current_spans, tol=X_ALIGN_TOL):
                pass
            else:
                current_section, current_spans = None, []
            if not current_section:
                continue
            lines = _group_words_into_lines(words, y_tol=Y_LINE_TOL)
            rows = _merge_lines_into_rows(lines, current_spans)
            if "KONTAKT" in current_section:
                parsed.contacts.extend(_extract_contacts(rows))
            elif "PROJEKT" in current_section:
                parsed.projects.extend(_extract_projects(rows))
            elif "UDBUD" in current_section:
                parsed.tenders.extend(_extract_tenders(rows))
            else:
                if not parsed.source_company and text:
                    top_lines = "\\n".join(text.splitlines()[:8])
                    m = re.search(r"^(?:Firma|Virksomhed|Company)\\s*[:\\-]\\s*(.+)$", top_lines, flags=re.I|re.M)
                    if m:
                        parsed.source_company = m.group(1).strip()
    return parsed

# -----------------------------
# CLI
# -----------------------------

def main(argv: List[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Smart Byggefakta PDF scraper (layout-agnostic, multi-page).")
    parser.add_argument("inputs", nargs="+", help="One or more input PDF files")
    parser.add_argument("-o", "--output", help="Write combined JSON to this file")
    args = parser.parse_args(argv)
    all_docs: List[ParsedDoc] = []
    for path in args.inputs:
        try:
            doc = parse_pdf(path)
            all_docs.append(doc)
        except Exception as e:
            all_docs.append(ParsedDoc(source_file=Path(path).name))
            print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)
    payload = [asdict(d) for d in all_docs]
    js = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(js, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(js)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
