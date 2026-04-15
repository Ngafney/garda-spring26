import os
import glob
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# 0. Dynamic Sector Loader
# ==========================================
SECTOR_MAP_PATH = r"C:\Users\rajaa\OneDrive\Desktop\GardDat\company_sector_map.csv"
COMPANY_TO_SECTOR = {}

if os.path.exists(SECTOR_MAP_PATH):
    print("--> Loading Sector Data from CSV...")
    sector_df = pd.read_csv(SECTOR_MAP_PATH)
    for _, row in sector_df.iterrows():
        COMPANY_TO_SECTOR[str(row['company_slug']).lower()] = str(row['sector'])
else:
    print("⚠️ WARNING: company_sector_map.csv not found. All companies will be 'Unassigned'.")

# ==========================================
# 1. Constants & Lexicons (FULL VERSION)
# ==========================================
LOGGER = logging.getLogger(__name__)

WORD_RE = re.compile(r"\b[\w%./-]+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DATE_IN_STEM_RE = re.compile(r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<call_date>\d{4}-\d{2}-\d{2})_(?P<fiscal_period>[^.]+)$")

QA_HEADER_RE = re.compile(r"^\s*(questions?\s*(?:&|and)\s*answers?|question-and-?answer(?:\s+session)?|q&a)\s*:?\s*$", flags=re.IGNORECASE)
QA_TRANSITION_RE = re.compile(r"\b(we will now begin the question-and-answer session|we will now be conducting a question-and-answer session|open the line for q&a|open the call for q&a|operator[, ]+please open the call for q&a|operator[, ]+please provide instructions for those interested in asking a question|let'?s open it up for questions)\b", flags=re.IGNORECASE)

MANAGEMENT_TITLE_HINTS = ("chief", "ceo", "cfo", "coo", "president", "chair", "chairman", "founder", "investor relations", "finance", "financial officer", "treasurer", "controller", "vice president", "svp", "evp", "vp", "director", "head of", "general manager")
ANALYST_TITLE_HINTS = ("analyst", "research", "securities", "capital markets", "equity research")

BOILERPLATE_PATTERNS = (re.compile(r"\bforward-looking statements?\b", flags=re.IGNORECASE), re.compile(r"\bsafe harbor\b", flags=re.IGNORECASE), re.compile(r"\bnon-gaap\b", flags=re.IGNORECASE), re.compile(r"\breconciliation\b", flags=re.IGNORECASE), re.compile(r"\boperator instructions?\b", flags=re.IGNORECASE), re.compile(r"\binvestor relations website\b", flags=re.IGNORECASE), re.compile(r"\bsec\b", flags=re.IGNORECASE), re.compile(r"\bwebcast replay\b", flags=re.IGNORECASE), re.compile(r"\bpress release\b", flags=re.IGNORECASE))
GUIDANCE_RAISED_PATTERNS = (re.compile(r"\brais(?:e|ed|ing)\b.{0,40}\bguidance\b", flags=re.IGNORECASE), re.compile(r"\bincreas(?:e|ed|ing)\b.{0,40}\boutlook\b", flags=re.IGNORECASE), re.compile(r"\bupdat(?:e|ed|ing)\b.{0,40}\bupward\b", flags=re.IGNORECASE))
GUIDANCE_LOWERED_PATTERNS = (re.compile(r"\blower(?:ed|ing)?\b.{0,40}\bguidance\b", flags=re.IGNORECASE), re.compile(r"\breduc(?:e|ed|ing)\b.{0,40}\boutlook\b", flags=re.IGNORECASE), re.compile(r"\bcut\b.{0,40}\bguidance\b", flags=re.IGNORECASE))

@dataclass(frozen=True)
class Block:
    section: str
    speaker: str | None
    title: str | None
    text: str

@dataclass(frozen=True)
class LexiconSpec:
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()

class PhraseCounter:
    def __init__(self, phrases: Sequence[str]):
        escaped = [re.escape(phrase.lower()) for phrase in phrases if phrase]
        self.pattern = re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(escaped), flags=re.IGNORECASE) if escaped else None
    def count(self, text: str) -> int:
        return len(self.pattern.findall(text)) if text and self.pattern else 0

BASE_SENTIMENT = LexiconSpec(positive=("accelerating", "accretive", "backlog", "benefit", "beneficial", "beat", "confidence", "confident", "constructive", "disciplined", "durable", "efficiency", "efficient", "expand", "expansion", "favorable", "free cash flow", "gain share", "growth", "healthy", "improve", "improved", "improving", "margin expansion", "momentum", "opportunity", "outperform", "positive", "pricing power", "productivity", "ramp", "record", "resilient", "robust", "solid", "stable", "strength", "strong", "upside", "well positioned"), negative=("challenging", "compression", "constraint", "constraints", "cautious", "cut", "cutting", "decline", "deceleration", "destocking", "deterioration", "difficult", "disruption", "downturn", "erosion", "headwind", "impairment", "inflationary", "loss", "miss", "negative", "pressure", "recession", "restructuring", "risk", "shortage", "slowdown", "soft", "softness", "uncertain", "uncertainty", "underperform", "volatility", "weak", "weaker", "weakness"))
UNCERTAINTY_TERMS = ("uncertain", "uncertainty", "volatile", "volatility", "visibility", "limited visibility", "challenging backdrop", "challenging environment", "macro uncertainty", "not clear", "unknown", "range of outcomes", "hard to predict", "difficult to predict", "fluid environment", "monitor closely", "cautious")
RISK_TERMS = ("headwind", "headwinds", "pressure", "risk", "risks", "challenging", "volatility", "recession", "tariff", "tariffs", "geopolitical", "fx", "foreign exchange", "interest rate", "interest rates", "consumer weakness", "slowdown", "macro pressure", "uncertainty")
DEMAND_LEXICON = LexiconSpec(positive=("strong demand", "healthy demand", "robust demand", "solid demand", "better demand", "improving demand", "stable demand", "order growth", "bookings growth", "backlog growth", "good demand", "demand recovery", "volume growth", "share gains", "market share gains"), negative=("weak demand", "soft demand", "demand slowdown", "slowing demand", "lower demand", "demand pressure", "order weakness", "bookings weakness", "backlog pressure", "customer caution", "cautious customer", "destocking", "inventory correction", "volume pressure", "traffic weakness"))
PRICING_LEXICON = LexiconSpec(positive=("pricing power", "price increase", "price increases", "positive pricing", "favorable pricing", "pricing discipline", "price realization", "net price", "margin expansion", "mix benefit", "premiumization", "higher price", "pass-through", "pass through", "pricing actions"), negative=("price pressure", "pricing pressure", "promotional", "promotions", "discounting", "discounts", "margin pressure", "cost inflation", "inflationary pressure", "input cost", "commodity inflation", "mix headwind", "unfavorable mix", "deflation", "price elasticity"))
CAPEX_LEXICON = LexiconSpec(positive=("capital expenditure", "capex", "investment", "investing", "capacity expansion", "buildout", "factory expansion", "new plant", "greenfield", "brownfield", "data center", "expansion project", "ramping capacity", "automation investment", "infrastructure investment"), negative=("cut capex", "reduce capex", "lower capex", "pause investment", "delay investment", "project delay", "project delays", "cancel project", "capacity reduction"))
SUPPLY_PRESSURE_TERMS = ("supply chain", "shortage", "shortages", "constraint", "constraints", "constrained", "bottleneck", "bottlenecks", "lead time", "lead times", "backorder", "backorders", "shipping delay", "freight pressure", "logistics pressure", "inventory shortage", "component shortage")
LABOR_PRESSURE_TERMS = ("labor shortage", "tight labor market", "labor availability", "wage pressure", "hiring challenge", "staffing challenge", "recruiting challenge", "turnover", "retention challenge", "overtime", "labor inflation", "wage inflation", "headcount pressure")
AUTOMATION_TERMS = ("automation", "automate", "automated", "robotics", "ai", "artificial intelligence", "machine learning", "copilot", "productivity gains", "efficiency gains", "digitalization", "software driven", "self-service")
MACRO_RISK_TERMS = ("recession", "macro uncertainty", "geopolitical", "tariff", "tariffs", "interest rate", "interest rates", "higher rates", "consumer weakness", "europe weakness", "china weakness", "fx headwind", "foreign exchange", "currency headwind", "inflation", "deflation", "credit tightening")

COUNTERS = {
    "base_positive": PhraseCounter(BASE_SENTIMENT.positive), "base_negative": PhraseCounter(BASE_SENTIMENT.negative),
    "uncertainty": PhraseCounter(UNCERTAINTY_TERMS), "risk": PhraseCounter(RISK_TERMS),
    "demand_positive": PhraseCounter(DEMAND_LEXICON.positive), "demand_negative": PhraseCounter(DEMAND_LEXICON.negative),
    "pricing_positive": PhraseCounter(PRICING_LEXICON.positive), "pricing_negative": PhraseCounter(PRICING_LEXICON.negative),
    "capex_positive": PhraseCounter(CAPEX_LEXICON.positive), "capex_negative": PhraseCounter(CAPEX_LEXICON.negative),
    "supply_pressure": PhraseCounter(SUPPLY_PRESSURE_TERMS), "labor_pressure": PhraseCounter(LABOR_PRESSURE_TERMS),
    "automation": PhraseCounter(AUTOMATION_TERMS), "macro_risk": PhraseCounter(MACRO_RISK_TERMS),
}

# ==========================================
# 2. Text Parsing & Analytics Methods 
# ==========================================
def normalize_space(text: str) -> str: return re.sub(r"\s+", " ", text).strip()
def slug_to_name(slug: str) -> str: return re.sub(r"\s+", " ", slug.replace("-", " ").replace("_", " ").strip()).title()
def split_sentences(text: str) -> list[str]: return [normalize_space(part) for part in SENTENCE_RE.split(text) if normalize_space(part)] if text else []
def tokenize(text: str) -> list[str]: return WORD_RE.findall(text.lower())
def safe_density(count: int, denominator: int) -> float: return count / denominator if denominator > 0 else 0.0
def safe_net(pos: int, neg: int, denom: int) -> float: return (pos - neg) / denom if denom > 0 else 0.0

def looks_like_speaker_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90 or stripped.endswith(":"): return False
    if QA_HEADER_RE.match(stripped): return False
    return True

def is_speaker_triplet(lines: list[str], index: int) -> bool:
    if index + 2 >= len(lines): return False
    return looks_like_speaker_line(lines[index]) and lines[index + 1].strip() == "--" and bool(lines[index + 2].strip())

def find_qa_start(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if QA_HEADER_RE.match(line.strip()): return idx
    floor = max(10, math.floor(len(lines) * 0.2))
    for idx in range(floor, len(lines)):
        if QA_TRANSITION_RE.search(lines[idx]): return idx
    return None

def parse_blocks(text: str) -> list[Block]:
    lines = text.splitlines()
    qa_start = find_qa_start(lines)
    blocks: list[Block] = []
    section, idx = "prepared", 0
    while idx < len(lines):
        if qa_start is not None and idx == qa_start:
            section = "qa"
            if QA_HEADER_RE.match(lines[idx].strip()):
                idx += 1; continue
            qa_start = None
        line = lines[idx].strip()
        if not line:
            idx += 1; continue
        if QA_HEADER_RE.match(line):
            section = "qa"
            idx += 1; continue
        if is_speaker_triplet(lines, idx):
            speaker, title = normalize_space(lines[idx]), normalize_space(lines[idx + 2])
            idx += 3
            body: list[str] = []
            while idx < len(lines):
                if qa_start is not None and idx == qa_start: break
                if QA_HEADER_RE.match(lines[idx].strip()) or is_speaker_triplet(lines, idx): break
                if stripped := normalize_space(lines[idx]): body.append(stripped)
                idx += 1
            blocks.append(Block(section=section, speaker=speaker, title=title, text=normalize_space(" ".join(body))))
            continue
        body: list[str] = []
        while idx < len(lines):
            if qa_start is not None and idx == qa_start: break
            if QA_HEADER_RE.match(lines[idx].strip()) or is_speaker_triplet(lines, idx): break
            if stripped := normalize_space(lines[idx]): body.append(stripped)
            idx += 1
        if joined := normalize_space(" ".join(body)):
            blocks.append(Block(section=section, speaker=None, title=None, text=joined))
    return blocks

def normalize_speaker_name(name: str | None) -> str | None:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()) if name else None

def classify_title(title: str | None) -> str:
    lowered = (title or "").lower()
    if "operator" in lowered: return "operator"
    if any(hint in lowered for hint in ANALYST_TITLE_HINTS): return "analyst"
    if any(hint in lowered for hint in MANAGEMENT_TITLE_HINTS): return "management"
    return "unknown"

def classify_block_roles(blocks: list[Block]) -> list[tuple[Block, str]]:
    management_speakers = {normalize_speaker_name(b.speaker) for b in blocks if b.section == "prepared" and normalize_speaker_name(b.speaker) and normalize_speaker_name(b.speaker) != "operator" and classify_title(b.title) != "analyst"}
    classified: list[tuple[Block, str]] = []
    for block in blocks:
        speaker_key, title_role = normalize_speaker_name(block.speaker), classify_title(block.title)
        if speaker_key == "operator" or title_role == "operator": role = "operator"
        elif speaker_key and speaker_key in management_speakers: role = "management"
        elif title_role in ("management", "analyst"): role = title_role
        elif block.section == "prepared": role = "management" if speaker_key else "other"
        elif block.section == "qa": role = "analyst" if speaker_key else "other"
        else: role = "other"
        classified.append((block, role))
    return classified

def clean_signal_text(text: str) -> str:
    return normalize_space(" ".join(s for s in split_sentences(normalize_space(text)) if not any(p.search(s) for p in BOILERPLATE_PATTERNS)))

def section_texts(text: str) -> dict[str, str]:
    classified = classify_block_roles(parse_blocks(text))
    prepared_parts, qa_parts, management_qa_parts, analyst_qa_parts = [], [], [], []
    for block, role in classified:
        if role == "operator": continue
        if block.section == "prepared": prepared_parts.append(block.text)
        elif block.section == "qa":
            qa_parts.append(block.text)
            if role == "management": management_qa_parts.append(block.text)
            elif role == "analyst": analyst_qa_parts.append(block.text)
    prepared, qa = clean_signal_text(" ".join(prepared_parts)), clean_signal_text(" ".join(qa_parts))
    return {
        "overall": clean_signal_text(" ".join(part for part in (prepared, qa) if part)),
        "prepared": prepared, "qa": qa,
        "management_qa": clean_signal_text(" ".join(management_qa_parts)),
        "analyst_qa": clean_signal_text(" ".join(analyst_qa_parts)),
    }

def complexity_metrics(text: str) -> dict[str, float]:
    sentences, tokens = split_sentences(text), tokenize(text)
    alpha_tokens = [t for t in tokens if t.isalpha()]
    avg_sentence_length = len(tokens) / len(sentences) if sentences else 0.0
    long_word_share = sum(len(t) >= 8 for t in alpha_tokens) / len(alpha_tokens) if alpha_tokens else 0.0
    return {
        "sentence_count": len(sentences), "token_count": len(tokens),
        "avg_sentence_length": avg_sentence_length, "long_word_share": long_word_share,
        "complexity_score": avg_sentence_length * long_word_share,
        "numeric_token_share": sum(any(c.isdigit() for c in t) for t in tokens) / len(tokens) if tokens else 0.0,
    }

def tone_metrics(text: str, prefix: str) -> dict[str, float]:
    tc = len(tokenize(text))
    pos, neg = COUNTERS["base_positive"].count(text), COUNTERS["base_negative"].count(text)
    return {
        f"{prefix}_positive_density": safe_density(pos, tc),
        f"{prefix}_negative_density": safe_density(neg, tc),
        f"{prefix}_net_tone": safe_net(pos, neg, tc),
        f"{prefix}_token_count": tc,
    }

def topic_metrics(text: str) -> dict[str, float]: 
    tc = len(tokenize(text)) 
    return { 
        "demand_positive_density": safe_density(COUNTERS["demand_positive"].count(text), tc), 
        "demand_negative_density": safe_density(COUNTERS["demand_negative"].count(text), tc), 
        "demand_net": safe_net(COUNTERS["demand_positive"].count(text), COUNTERS["demand_negative"].count(text), tc), 
        "pricing_positive_density": safe_density(COUNTERS["pricing_positive"].count(text), tc), 
        "pricing_negative_density": safe_density(COUNTERS["pricing_negative"].count(text), tc), 
        "pricing_power_net": safe_net(COUNTERS["pricing_positive"].count(text), COUNTERS["pricing_negative"].count(text), tc), 
        "capex_positive_density": safe_density(COUNTERS["capex_positive"].count(text), tc), 
        "capex_negative_density": safe_density(COUNTERS["capex_negative"].count(text), tc), 
        "capex_net": safe_net(COUNTERS["capex_positive"].count(text), COUNTERS["capex_negative"].count(text), tc), 
        "supply_chain_pressure_density": safe_density(COUNTERS["supply_pressure"].count(text), tc), 
        "labor_pressure_density": safe_density(COUNTERS["labor_pressure"].count(text), tc), 
        "automation_density": safe_density(COUNTERS["automation"].count(text), tc), 
        "uncertainty_density": safe_density(COUNTERS["uncertainty"].count(text), tc),
        "risk_density": safe_density(COUNTERS["risk"].count(text), tc), 
        "macro_risk_density": safe_density(COUNTERS["macro_risk"].count(text), tc), 
    } 

def guidance_flags(text: str) -> dict[str, int]:
    return {
        "guidance_raised": int(any(p.search(text) for p in GUIDANCE_RAISED_PATTERNS)),
        "guidance_lowered": int(any(p.search(text) for p in GUIDANCE_LOWERED_PATTERNS)),
    }

# ==========================================
# 3. Data Engineering Logic
# ==========================================
def score_transcript(file_path: str, folder_dir: str) -> dict[str, object]:
    relative_path = os.path.relpath(file_path, folder_dir)
    path_parts = relative_path.split(os.sep)
    
    company_name = path_parts[0] if len(path_parts) > 1 else os.path.basename(file_path).replace('.txt', '')
    fiscal_year = path_parts[1] if len(path_parts) > 1 else ""
    fiscal_period = path_parts[2] if len(path_parts) > 2 else ""

    # DYNAMIC SECTOR LOOKUP!
    sector = COMPANY_TO_SECTOR.get(company_name.lower(), "Unassigned")

    try:
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f: text = f.read()

    sections = section_texts(text)
    
    overall_complexity = complexity_metrics(sections["overall"])
    overall_tone = tone_metrics(sections["overall"], "overall")
    prepared_tone = tone_metrics(sections["prepared"], "prepared")
    qa_tone = tone_metrics(sections["qa"], "qa")
    management_qa_tone = tone_metrics(sections["management_qa"], "management_qa")
    analyst_qa_tone = tone_metrics(sections["analyst_qa"], "analyst_question")
    
    topical = topic_metrics(sections["overall"])
    guidance = guidance_flags(text)

    qna_reality_gap = management_qa_tone["management_qa_net_tone"] - prepared_tone["prepared_net_tone"]
    analyst_management_gap = management_qa_tone["management_qa_net_tone"] - analyst_qa_tone["analyst_question_net_tone"]

    growth_signal = overall_tone["overall_net_tone"] + topical["demand_net"] + topical["capex_net"] - topical["macro_risk_density"]
    margin_signal = (topical["pricing_power_net"] + topical["automation_density"] - topical["labor_pressure_density"] - topical["supply_chain_pressure_density"])
    credibility_signal = analyst_management_gap + qna_reality_gap - topical["uncertainty_density"]

    return {
        "sector": sector,
        "company_name": company_name,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "qa_detected": int(bool(sections["qa"])),
        **overall_complexity, **overall_tone, **prepared_tone, **qa_tone,
        **management_qa_tone, **analyst_qa_tone, **topical, **guidance,
        "qna_reality_gap": qna_reality_gap, 
        "analyst_management_gap": analyst_management_gap,
        "growth_signal": growth_signal, 
        "margin_signal": margin_signal,
        "credibility_signal": credibility_signal,
        "composite_signal": growth_signal + margin_signal + credibility_signal,
    }

# ==========================================
# 4. Execution Logic
# ==========================================
def main():
    folder_dir = r"C:\Users\rajaa\OneDrive\Desktop\GardDat"
    search_pattern = os.path.join(folder_dir, "**", "*.txt")
    txt_files = glob.glob(search_pattern, recursive=True)
    
    print(f"--> SYSTEM FOUND {len(txt_files)} TEXT FILES <--")
    if not txt_files: return

    rows = [score_transcript(path, folder_dir) for path in tqdm(txt_files, desc="Scoring")]
    panel = pd.DataFrame(rows)
    
    # Create the Year-Quarter label
    panel['fiscal_year'] = panel['fiscal_year'].astype(str).str.strip()
    panel['fiscal_period'] = panel['fiscal_period'].astype(str).str.strip()
    panel['year_quarter'] = panel['fiscal_year'] + "-" + panel['fiscal_period']
    
    # GROUP BY SECTOR AND QUARTER
    numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    sector_quarter_df = panel.groupby(['sector', 'year_quarter'])[numeric_cols].mean().reset_index()
    sector_quarter_df = sector_quarter_df.sort_values(by=['sector', 'year_quarter'])
    
    results_df = sector_quarter_df[["sector", "year_quarter", "composite_signal", "growth_signal", "margin_signal", "overall_net_tone"]].round(4)
    
    print("\n✅ Success! Aggregated Data by SECTOR and QUARTER:")
    print(results_df.to_string(index=False))
    
    output_csv = os.path.join(folder_dir, "earnings_research_sector_quarterly.csv")
    sector_quarter_df.to_csv(output_csv, index=False)
    print(f"\n📂 Final sector data saved to: {output_csv}")

if __name__ == "__main__":
    main()