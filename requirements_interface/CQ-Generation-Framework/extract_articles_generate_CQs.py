import os
import time
import re
import json
import spacy
import datetime
import io
import openai
import requests
import tiktoken
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pandas as pd
from newspaper import Article
from serpapi import GoogleSearch
from dotenv import load_dotenv
from langdetect import detect
from urllib.parse import urlparse

# ========== Setup & Helpers ==========
MIN_TEXT_CHARS = 1500

nlp = spacy.load("en_core_web_sm")
def lemmatized_tokens(text: str, max_chars: int = 8000) -> set:
    """
    Returns a set of lemmatized lowercase tokens from the text.
    Limits processed length to avoid spacy slowdown on very long documents.
    """
    doc = nlp(text[:max_chars].lower())
    return {token.lemma_ for token in doc if token.is_alpha and not token.is_stop}

def estimate_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def load_environment_variables() -> None:
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)

load_environment_variables()

openai.api_key = os.getenv("OPENAI_API_KEY_4o")
openai.api_type = os.getenv("OPENAI_API_TYPE_4o")
openai.api_version = os.getenv("OPENAI_API_VERSION_4o")
openai.azure_endpoint = os.getenv("OPENAI_API_BASE_4o")
deployment_name = os.getenv("DEPLOYMENT_NAME_4o") or "gpt-4o"
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if openai.azure_endpoint:
    setattr(openai, "api_base", openai.azure_endpoint)
if openai.api_type:
    if openai.api_type not in ("openai", "azure"):
        raise ValueError(f"Invalid OPENAI_API_TYPE_4o value: {openai.api_type}")
    setattr(openai, "api_type", openai.api_type)
if openai.api_version:
    setattr(openai, "api_version", openai.api_version)

SYSTEM_PROMPT = "You are ChatGPT, a helpful assistant."
MAX_TOKENS_GEN = 8000   # Output size
PRESENCE_PENALTY = 0.0

@dataclass(frozen=True)
class GenConfig:
    temperature: float
    top_p: float
    freq_penalty: float
    def tag(self): return f"T{self.temperature}-P{self.top_p}-F{self.freq_penalty}"

def chat_call(content, temperature, top_p, freq_penalty, max_tokens):
    while True:
        try:
            resp = openai.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=freq_penalty,
                presence_penalty=PRESENCE_PENALTY,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("OpenAI error, retry in 5s:", e)
            time.sleep(5)

def normalize_paragraphs(text: str) -> list[str]:
    """
    Convert raw PDF/HTML text into logical paragraphs:
    - unify newlines
    - remove hyphenation at line breaks: 'deterio-\nration' -> 'deterioration'
    - join single newlines (hard wraps) into spaces
    - split on blank lines to get paragraphs
    """
    t = text.replace("\r\n", "\n")

    # de-hyphenate line-break splits
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # collapse sequences like "word\nword" into "word word" (but keep blank-line paragraph breaks)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)

    # squeeze 3+ newlines into exactly two
    t = re.sub(r"\n{3,}", "\n\n", t)

    # split paragraphs on blank lines
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    return paras

# ========== Domain config ==========
PUBLISHER_SITES = [
    "site:springer.com",
]

APPROVED_DOMAINS = [
    "springer.com",
]

def is_english(text: str, min_chars: int = 300) -> bool:
    """Return True if text is detected as English. Requires langdetect."""
    try:
        # Use only a chunk if text is huge
        sample = text if len(text) <= 2000 else text[:2000]
        if len(sample) < min_chars:
            return True  # too short to tell reliably, allow
        return detect(sample) == "en"
    except Exception:
        return True  # on failure, allow


# ========== Load domain information from JSON ==========
def load_domain_config(config_path: str = "domain-info.json") -> Dict:
    """
    Load domain configuration from JSON file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        return {}

# Load domain configuration
DOMAIN_CONFIG = load_domain_config()

# Assign variables from domain configuration
MAIN_DOMAIN_NAME = DOMAIN_CONFIG.get("MAIN_DOMAIN_NAME", "Unknown Domain")
TOPIC_TERMS = DOMAIN_CONFIG.get("TOPIC_TERMS", [])
FILTER_KEYWORDS = DOMAIN_CONFIG.get("FILTER_KEYWORDS", [])
MAIN_DOMAIN_WORDS = DOMAIN_CONFIG.get("MAIN_DOMAIN_WORDS", [])
COMPOUND_GENERAL_TERMS = DOMAIN_CONFIG.get("COMPOUND_GENERAL_TERMS", [])
ONTOLOGY_COVERAGE_AREAS = DOMAIN_CONFIG.get("ONTOLOGY_COVERAGE_AREAS", [])

all_items = []

# Iterate through all key-value pairs in the domain_info dictionary
for key, value in DOMAIN_CONFIG.items():
    # Add the key itself
    all_items.append(key)

    # Add the values
    if isinstance(value, list):
        all_items.extend(value)
    else:
        all_items.append(str(value))

scope_text = "\n".join(all_items)

# ========== Save Article Summary ==========
def save_article_summary(articles: List[Dict], token_count: int, output_path: str = "article_summary.txt"):
    """
    Save article information and token count to a text file.
    """
    summary_path = Path(output_path)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ARTICLE COLLECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total Articles Collected: {len(articles)}\n")
        f.write(f"Total Input Tokens: {token_count}\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("ARTICLES DETAILS:\n")
        f.write("-" * 30 + "\n\n")

        for i, article in enumerate(articles, 1):
            f.write(f"Article {i}:\n")
            f.write(f"  Title: {article['title']}\n")
            f.write(f"  URL: {article['url']}\n")
            f.write(f"  Text Length: {len(article['text'])} characters\n")

            # Count filter keywords in this article
            lower_text = article['text'].lower()
            keyword_count = sum(1 for k in FILTER_KEYWORDS if k in lower_text)
            f.write(f"  Filter Keywords Found: {keyword_count}\n")
            f.write("\n")

# ========== Fetching from Google Scholar via SerpAPI ==========
def build_scholar_queries() -> List[str]:
    """
    Compose several Scholar queries mixing your topic terms with publisher site limits.
    We’ll favor OA PDFs but still try HTML when needed.
    """
    queries = []
    for site in PUBLISHER_SITES:
        for topic in TOPIC_TERMS:
            q = f'("{topic}") ({COMPOUND_GENERAL_TERMS[0]} OR {COMPOUND_GENERAL_TERMS[1]}) {site}'
            queries.append(q)
    # add a broad OA-leaning query per site
    for site in PUBLISHER_SITES:
        queries.append(f'{COMPOUND_GENERAL_TERMS[0]} OR {COMPOUND_GENERAL_TERMS[1]} {site} (pdf OR "open access")')
    return queries[:31]


def scholar_search(query: str, start: int = 0) -> List[Dict]:
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": serpapi_api_key,
        "start": start,
        "num": 10,
        "hl": "en",
    }
    search = GoogleSearch(params)
    result = search.get_dict()
    return result.get("organic_results", []) or []


def get_pdf_url_from_result(res: Dict) -> Optional[str]:
    """
    SerpAPI Google Scholar often includes a 'resources' list with PDFs.
    """
    resources = res.get("resources") or []
    for r in resources:
        if r.get("file_format", "").lower() == "pdf" and r.get("link"):
            return r["link"]
    # sometimes the result link itself is a PDF
    link = res.get("link")
    if link and link.lower().endswith(".pdf"):
        return link
    return None

# ========== Content extraction ==========
def try_download(url: str, timeout: int = 25) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Return (bytes, content_type) or (None, None).
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return r.content, (r.headers.get("Content-Type") or "").lower()
    except Exception as e:
        print(f"[x] Download failed: {url} ({e})")
    return None, None

def looks_like_pdf(content: Optional[bytes], content_type: Optional[str]) -> bool:
    """
    Trust magic bytes or content-type; never just the URL suffix.
    """
    if content_type and "application/pdf" in content_type:
        return True
    if content and content[:4] == b"%PDF":
        return True
    return False

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text with pdfminer; fail gracefully so we can fall back to HTML.
    """
    from pdfminer.high_level import extract_text
    try:
        text = extract_text(io.BytesIO(pdf_bytes))
        return text or ""
    except Exception as e:
        print(f"[x] PDF parse failed: {e}")
        return ""

# Configurable flag: only allow fallback if explicitly enabled
ALLOW_HTML_FALLBACK = False

def extract_article_text_from_url(url: str) -> Tuple[str, str]:
    """
    Returns (title, text). Tries PDF extraction first.
    Accepts shorter PDFs and does not require section headings by default.
    Falls back to HTML (newspaper3k) only if ALLOW_HTML_FALLBACK is True.

    Loosened gates:
      - Length threshold defaults to 1500 chars (configurable via MIN_TEXT_CHARS).
      - 'Paper structure' tokens are optional (toggle via REQUIRE_PAPER_STRUCTURE).
    """
    # Config knobs with safe fallbacks if not defined at module level
    MIN_TEXT_CHARS = globals().get("MIN_TEXT_CHARS", 1500)

    content, ctype = try_download(url)

    # Only treat as PDF when content-type or magic bytes confirm it
    if content and looks_like_pdf(content, ctype) and len(content) > 2048:
        text = extract_pdf_text(content).strip()
        if not text:
            print(f"[!] Skipped PDF (empty after parse): {url}")
            return "", ""

        lower = text.lower()

        # Looser length gate
        if len(text) < MIN_TEXT_CHARS:
            print(f"[!] Skipped PDF (too short: {len(text)} chars): {url}")
            return "", ""

        # Passed gates
        return "", text

    # If not a valid/usable PDF
    if not ALLOW_HTML_FALLBACK:
        print(f"[!] Skipped non-PDF or invalid PDF: {url}")
        return "", ""

    # Optional HTML fallback (disabled unless ALLOW_HTML_FALLBACK = True)
    try:
        art = Article(url)
        art.download()
        art.parse()
        title = (art.title or "").strip()
        text = art.text or ""
        if len(text) >= MIN_TEXT_CHARS:
            print(f"[~] Using fallback HTML: {url}")
            return title, text
        else:
            print(f"[!] Skipped HTML (too short: {len(text)} chars): {url}")
    except Exception as e:
        print(f"[x] Newspaper parse failed: {url} ({e})")

    return "", ""

# ========== Pipeline: fetch at most 30 full, related articles ==========

def fetch_fulltext_articles(required_count: int = 30) -> List[Dict]:
    """
    Fetch articles until all TOPIC_TERMS are covered, or up to 30 articles max.
    Returns a list of dicts: { 'title', 'url', 'text' }
    """
    collected: List[Dict] = []
    seen_urls = set()
    covered_terms = set()
    queries = build_scholar_queries()
    current_year = datetime.datetime.now().year

    for qi, q in enumerate(queries):
        print(f"[Q{qi + 1}/{len(queries)}] {q}")
        for start in (0, 10, 20, 30):  # Four pages per query
            results = scholar_search(q, start=start)
            if not results:
                continue

            for r in results:
                # --- YEAR FILTER ---
                year = None

                # Try dict with explicit year
                pub_info = r.get("publication_info") or {}
                if isinstance(pub_info, dict):
                    year = pub_info.get("year")
                    if not year and "summary" in pub_info:
                        m = re.search(r"\b(19|20)\d{2}\b", pub_info["summary"])
                        if m:
                            year = int(m.group(0))
                elif isinstance(pub_info, str):
                    m = re.search(r"\b(19|20)\d{2}\b", pub_info)
                    if m:
                        year = int(m.group(0))

                # Fallback: look in title/snippet
                if not year:
                    for field in [r.get("title", ""), r.get("snippet", "")]:
                        m = re.search(r"\b(19|20)\d{2}\b", field)
                        if m:
                            year = int(m.group(0))
                            break

                # Enforce cutoff
                if year and year < current_year - 15:
                    print(f"[i] Skipped old article ({year}): {r.get('title')}")
                    continue
                # --------------------------------

                if len(collected) >= 30:
                    print("[!] Max limit of 30 articles reached.")
                    print(f"[i] Final coverage: {len(covered_terms)} / {len(TOPIC_TERMS)} topic terms")
                    return collected

                pdf_url = get_pdf_url_from_result(r)
                target_url = pdf_url or r.get("link")
                if not target_url or target_url in seen_urls:
                    continue

                domain = urlparse(target_url).netloc.lower()
                if not any(domain.endswith(allowed) for allowed in APPROVED_DOMAINS):
                    continue
                if "scopus.com" in target_url:
                    continue

                seen_urls.add(target_url)
                title, text = extract_article_text_from_url(target_url)

                if not title and pdf_url and r.get("link") and r["link"] != target_url:
                    try:
                        a2 = Article(r["link"])
                        a2.download(); a2.parse()
                        title = (a2.title or "").strip()
                    except Exception:
                        pass

                if text and len(text) > 1500:
                    lower = text.lower()
                    if any(term in lower for term in MAIN_DOMAIN_WORDS):
                        # Require at least 10 filter keywords in the article
                        keyword_count = sum(1 for k in FILTER_KEYWORDS if k in lower)
                        if keyword_count >= 7:
                            if is_english(text):
                                article_lemmas = lemmatized_tokens(text)
                                matched_terms = []
                                for term in TOPIC_TERMS:
                                    term_lemmas = lemmatized_tokens(term)
                                    if term_lemmas and all(t in article_lemmas for t in term_lemmas):
                                        matched_terms.append(term)
                                if len(matched_terms) >= 3:
                                    covered_terms.update(matched_terms)
                                    title = title or (r.get("title") or "Untitled Article")
                                    collected.append({"title": title, "url": target_url, "text": text})
                                    print(f"[+] Article added: {title} ({target_url}) - Terms matched: {matched_terms}")

                                    if len(covered_terms) == len(TOPIC_TERMS):
                                        print("All topic terms covered.")
                                        return collected
                            else:
                                print(f"[i] Skipped non-English: {target_url}")

    print(f"[i] Final coverage: {len(covered_terms)} / {len(TOPIC_TERMS)} topic terms")
    return collected


# ========== Snippet filtering & CQ prompt ==========
def filter_snippets(text: str, keywords: list[str]) -> list[str]:
    """
    Return paragraphs that contain any of the keywords.
    Uses normalized paragraphs instead of raw line splits.
    """
    paras = normalize_paragraphs(text)
    out = []
    kws = [k.lower() for k in keywords]
    for p in paras:
        p_low = p.lower()
        if len(p) < 100:  # keep your minimum-length guard
            continue
        if any(k in p_low for k in kws):
            out.append(p)
    return out

PROMPT_CQ = """
You are a **Senior Ontology Engineer** with deep domain expertise in **{MAIN_DOMAIN_NAME}**.

Your task is to generate Competency Questions (CQs) that will guide ontology development.
You combine domain knowledge with ontological thinking to create questions that are:
- Practically relevant to domain practitioners
- Technically implementable in an ontology
- Aligned with ontology engineering best practices

Below are **real excerpts** from articles related to **{MAIN_DOMAIN_NAME}**:
{payload}

Based on these articles, generate competency questions (CQs) suitable for ONTOLOGY DESIGN in the following domain:
{scope_text}

Focus on questions that help define:
- Classes and hierarchies
- Properties and relationships  
- Data structures and taxonomies
- Domain scope and boundaries

Note: Avoid "how" and "why" questions. Instead ask questions with "What", "Which", "What [entities]..., "What are...", "What is...", "What types", "What categories", "What properties", and "What relationships".

I'll give you examples of BAD CQs and GOOD CQs for ontology development. Generate CQs following the GOOD pattern:

BAD: "How does sleep quality affect cognitive function?"
GOOD: "What types of sleep quality measurements are collected?"

BAD: "How do construction practices contribute to risk?"  
GOOD: "What categories of construction practices affect building risk?"

BAD: "How do assessments correlate with diagnoses?"
GOOD: "What relationships exist between assessments and diagnoses?"

BAD: "How does building age influence vulnerability?" 
GOOD: "What properties describe building vulnerability factors?"

GOOD: "What criteria define mild vs. moderate cognitive impairment?"
GOOD: "What components constitute a comprehensive neuropsychological assessment?"


Now your task is to:
1. Generate at least 80 Competency Questions (CQs) from the provided content only.  
    - **Do not use your general knowledge**. Only base your competency questions on the given text snippets.   

2. The CQs must align with the following **Ontology Coverage Areas**:  
     **{ONTOLOGY_COVERAGE_AREAS}** 
     CRITICAL: Ensure CQs diversity. Generate questions covering all coverage areas.
      
3. Each CQ must be:  
   - **Domain-relevant**: CQs must be in the domain.   
   - **Clear and precise**: CQs should be clear and understandable for all stakeholders.  
   - **simple and ontology-driven**: Aiming to define the scope and structure of the knowledge base rather than answer complex or analytical questions.
   - **Atomic**: An "atomic" Competency Question is one that expresses a single, indivisible information requirement, free of compound concepts that can be logically split.
        
4. Output format:  
   - One CQ per line, numbered sequentially  
   - Avoid duplication of meaning (no semantic redundancies)  

"""

def run_experiment(out_path=None):
    if out_path is None:
        output_dir = Path(__file__).resolve().parent / "output"
        out_path = output_dir / f"llm_input_springer_{datetime.datetime.now():%Y%m%d_%H%M%S}.xlsx"
    print("Fetching full-text scholarly articles from target publishers...")
    articles = fetch_fulltext_articles(required_count=30)

    if not articles:
        print("No articles collected. Exiting.")
        return

    article_data = []
    snippet_table = {}

    for idx, a in enumerate(articles):
        print(f"\nArticle {idx + 1}: {a['title']}\nURL: {a['url']}")
        text = a["text"]
        title = a["title"]
        snippets = filter_snippets(text, FILTER_KEYWORDS)
        article_data.append({"title": title, "snippets": snippets})
        snippet_table[title] = snippets

    # Build payload (round-robin across articles for fair coverage)
    max_per_article = 3  # take up to 3 snippets per article before looping
    BUDGET = 50000
    payload = ""

    max_len = max((len(a["snippets"]) for a in article_data), default=0)
    stop = False
    for i in range(max_len):
        for a in article_data:
            if i < len(a["snippets"]) and i < max_per_article:
                snippet = a["snippets"][i]
                if estimate_tokens(payload + "\n\n" + snippet) > BUDGET:
                    stop = True
                    break
                payload += "\n\n" + snippet
        if stop:
            break

    token_count = estimate_tokens(payload)
    print("Estimated input token count:", token_count)
    summary_path = output_dir / f"articles_summary_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    save_article_summary(articles, token_count, summary_path)

    # Generate competency questions
    config = GenConfig(temperature=0.3, top_p=1.0, freq_penalty=0.2)
    print("\nGenerating competency questions...")
    coverage_text = "\n- " + "\n- ".join(ONTOLOGY_COVERAGE_AREAS)
    resp = chat_call(PROMPT_CQ.format(payload=payload, scope_text=scope_text, MAIN_DOMAIN_NAME=MAIN_DOMAIN_NAME, ONTOLOGY_COVERAGE_AREAS=coverage_text),
                     temperature=config.temperature,
                     top_p=config.top_p,
                     freq_penalty=config.freq_penalty,
                     max_tokens=MAX_TOKENS_GEN)

    # Parse CQs
    rows = []
    for line in resp.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            rows.append({"Config": config.tag(), "Source": "scholar+publishers", "CQ": line})

    df_cq = pd.DataFrame(rows)

    # Prepare snippet DataFrame with 1 column per article title
    max_rows = max((len(a["snippets"]) for a in article_data), default=0)
    snippet_df = pd.DataFrame()
    for a in article_data:
        padded = a["snippets"] + [""] * (max_rows - len(a["snippets"]))
        snippet_df[a["title"]] = padded

    out_path = Path(out_path)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_cq.to_excel(writer, sheet_name="CQs", index=False)
        snippet_df.to_excel(writer, sheet_name="Snippets", index=False)

    print(f"\nResults saved to: {out_path.resolve()}")

if __name__ == "__main__":
    run_experiment()
