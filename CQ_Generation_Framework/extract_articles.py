import re
import json
import spacy
import datetime
import io
import requests
import tiktoken
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pandas as pd
from newspaper import Article
from serpapi import GoogleSearch
from langdetect import detect
from urllib.parse import urlparse
from utils import load_environment_variables, initialize_clients

load_environment_variables()
deployment_name, serpapi_api_key = initialize_clients()

# ========== Setup & Helpers ==========
MIN_TEXT_CHARS = 1500
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

nlp = spacy.load("en_core_web_sm")

def lemmatized_tokens(text: str, max_chars: int = 8000) -> set:
    doc = nlp(text[:max_chars].lower())
    return {token.lemma_ for token in doc if token.is_alpha and not token.is_stop}

def estimate_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def normalize_paragraphs(text: str) -> list[str]:
    t = text.replace("\r\n", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    return paras

# ========== Domain config ==========
PUBLISHER_SITES = ["site:springer.com"]
APPROVED_DOMAINS = ["springer.com"]

def is_english(text: str, min_chars: int = 300) -> bool:
    try:
        sample = text if len(text) <= 2000 else text[:2000]
        if len(sample) < min_chars:
            return True
        return detect(sample) == "en"
    except Exception:
        return True

# ========== Load domain information from JSON ==========
def load_domain_config(
        config_path: str = "json_input/domain-info.json") -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        return {}

DOMAIN_CONFIG = load_domain_config()

MAIN_DOMAIN_NAME = DOMAIN_CONFIG.get("MAIN_DOMAIN_NAME", "Unknown Domain")
TOPIC_TERMS = DOMAIN_CONFIG.get("TOPIC_TERMS", [])
FILTER_KEYWORDS = DOMAIN_CONFIG.get("FILTER_KEYWORDS", [])
MAIN_DOMAIN_WORDS = DOMAIN_CONFIG.get("MAIN_DOMAIN_WORDS", [])
COMPOUND_GENERAL_TERMS = DOMAIN_CONFIG.get("COMPOUND_GENERAL_TERMS", [])
ONTOLOGY_COVERAGE_AREAS = DOMAIN_CONFIG.get("ONTOLOGY_COVERAGE_AREAS", [])

all_items = []
for key, value in DOMAIN_CONFIG.items():
    all_items.append(key)
    if isinstance(value, list):
        all_items.extend(value)
    else:
        all_items.append(str(value))
scope_text = "\n".join(all_items)

# ========== Save Article Summary ==========
def save_article_summary(
        articles: List[Dict],
        token_count: int,
        output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("ARTICLE COLLECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Articles Collected: {len(articles)}\n")
        f.write(f"Total Input Tokens: {token_count}\n")
        f.write(
            f"Generated on: "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("ARTICLES DETAILS:\n")
        f.write("-" * 30 + "\n\n")
        for i, article in enumerate(articles, 1):
            f.write(f"Article {i}:\n")
            f.write(f"  Title: {article['title']}\n")
            f.write(f"  URL: {article['url']}\n")
            f.write(f"  Text Length: {len(article['text'])} characters\n")
            lower_text = article['text'].lower()
            keyword_count = sum(
                1 for k in FILTER_KEYWORDS if k in lower_text)
            f.write(f"  Filter Keywords Found: {keyword_count}\n\n")

# ========== Fetching from Google Scholar via SerpAPI ==========
def build_scholar_queries() -> List[str]:
    queries = []
    for site in PUBLISHER_SITES:
        for topic in TOPIC_TERMS:
            q = (f'("{topic}") '
                 f'({COMPOUND_GENERAL_TERMS[0]} OR '
                 f'{COMPOUND_GENERAL_TERMS[1]}) {site}')
            queries.append(q)
    for site in PUBLISHER_SITES:
        queries.append(
            f'{COMPOUND_GENERAL_TERMS[0]} OR '
            f'{COMPOUND_GENERAL_TERMS[1]} {site} (pdf OR "open access")'
        )
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
    resources = res.get("resources") or []
    for r in resources:
        if r.get("file_format", "").lower() == "pdf" and r.get("link"):
            return r["link"]
    link = res.get("link")
    if link and link.lower().endswith(".pdf"):
        return link
    return None

# ========== Content extraction ==========
def try_download(
        url: str,
        timeout: int = 25) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return r.content, (r.headers.get("Content-Type") or "").lower()
    except Exception as e:
        print(f"[x] Download failed: {url} ({e})")
    return None, None

def looks_like_pdf(
        content: Optional[bytes],
        content_type: Optional[str]) -> bool:
    if content_type and "application/pdf" in content_type:
        return True
    if content and content[:4] == b"%PDF":
        return True
    return False

def extract_pdf_text(pdf_bytes: bytes) -> str:
    from pdfminer.high_level import extract_text
    try:
        text = extract_text(io.BytesIO(pdf_bytes))
        return text or ""
    except Exception as e:
        print(f"[x] PDF parse failed: {e}")
        return ""

ALLOW_HTML_FALLBACK = False

def extract_article_text_from_url(url: str) -> Tuple[str, str]:
    content, ctype = try_download(url)
    if content and looks_like_pdf(content, ctype) and len(content) > 2048:
        text = extract_pdf_text(content).strip()
        if not text:
            print(f"[!] Skipped PDF (empty after parse): {url}")
            return "", ""
        if len(text) < MIN_TEXT_CHARS:
            print(
                f"[!] Skipped PDF (too short: {len(text)} chars): {url}")
            return "", ""
        return "", text
    if not ALLOW_HTML_FALLBACK:
        print(f"[!] Skipped non-PDF or invalid PDF: {url}")
        return "", ""
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
            print(
                f"[!] Skipped HTML (too short: {len(text)} chars): {url}")
    except Exception as e:
        print(f"[x] Newspaper parse failed: {url} ({e})")
    return "", ""

# ========== Pipeline: fetch articles ==========
def fetch_fulltext_articles(required_count: int = 30) -> List[Dict]:
    collected: List[Dict] = []
    seen_urls = set()
    covered_terms = set()
    queries = build_scholar_queries()
    current_year = datetime.datetime.now().year

    for qi, q in enumerate(queries):
        print(f"[Q{qi + 1}/{len(queries)}] {q}")
        for start in (0, 10, 20, 30):
            results = scholar_search(q, start=start)
            if not results:
                continue
            for r in results:
                year = None
                pub_info = r.get("publication_info") or {}
                if isinstance(pub_info, dict):
                    year = pub_info.get("year")
                    if not year and "summary" in pub_info:
                        m = re.search(
                            r"\b(19|20)\d{2}\b", pub_info["summary"])
                        if m:
                            year = int(m.group(0))
                elif isinstance(pub_info, str):
                    m = re.search(r"\b(19|20)\d{2}\b", pub_info)
                    if m:
                        year = int(m.group(0))
                if not year:
                    for field in [
                            r.get("title", ""), r.get("snippet", "")]:
                        m = re.search(r"\b(19|20)\d{2}\b", field)
                        if m:
                            year = int(m.group(0))
                            break
                if year and year < current_year - 15:
                    print(
                        f"[i] Skipped old article ({year}): "
                        f"{r.get('title')}")
                    continue

                if len(collected) >= 30:
                    print("[!] Max limit of 30 articles reached.")
                    print(
                        f"[i] Final coverage: {len(covered_terms)} / "
                        f"{len(TOPIC_TERMS)} topic terms")
                    return collected

                pdf_url = get_pdf_url_from_result(r)
                target_url = pdf_url or r.get("link")
                if not target_url or target_url in seen_urls:
                    continue

                domain = urlparse(target_url).netloc.lower()
                if not any(
                        domain.endswith(a) for a in APPROVED_DOMAINS):
                    continue
                if "scopus.com" in target_url:
                    continue

                seen_urls.add(target_url)
                title, text = extract_article_text_from_url(target_url)

                if not title and pdf_url and r.get("link") \
                        and r["link"] != target_url:
                    try:
                        a2 = Article(r["link"])
                        a2.download()
                        a2.parse()
                        title = (a2.title or "").strip()
                    except Exception:
                        pass

                if text and len(text) > 1500:
                    lower = text.lower()
                    if any(term in lower for term in MAIN_DOMAIN_WORDS):
                        keyword_count = sum(
                            1 for k in FILTER_KEYWORDS if k in lower)
                        if keyword_count >= 7:
                            if is_english(text):
                                article_lemmas = lemmatized_tokens(text)
                                matched_terms = []
                                for term in TOPIC_TERMS:
                                    term_lemmas = lemmatized_tokens(term)
                                    if term_lemmas and all(
                                            t in article_lemmas
                                            for t in term_lemmas):
                                        matched_terms.append(term)
                                if len(matched_terms) >= 3:
                                    covered_terms.update(matched_terms)
                                    title = title or (
                                        r.get("title") or
                                        "Untitled Article")
                                    collected.append({
                                        "title": title,
                                        "url": target_url,
                                        "text": text
                                    })
                                    print(
                                        f"[+] Article added: {title} "
                                        f"({target_url}) - "
                                        f"Terms matched: {matched_terms}"
                                    )
                                    if len(covered_terms) == \
                                            len(TOPIC_TERMS):
                                        print("All topic terms covered.")
                                        return collected
                            else:
                                print(
                                    f"[i] Skipped non-English: "
                                    f"{target_url}")

    print(
        f"[i] Final coverage: {len(covered_terms)} / "
        f"{len(TOPIC_TERMS)} topic terms")
    return collected

# ========== Snippet filtering ==========
def filter_snippets(text: str, keywords: list[str]) -> list[str]:
    paras = normalize_paragraphs(text)
    out = []
    kws = [k.lower() for k in keywords]
    for p in paras:
        p_low = p.lower()
        if len(p) < 100:
            continue
        if any(k in p_low for k in kws):
            out.append(p)
    return out

# ========== Main ==========
def run_extraction() -> Path:
    """
    Fetch articles, extract snippets, save to Excel.
    Returns path to saved Excel file.
    """
    print("Fetching full-text scholarly articles...")
    articles = fetch_fulltext_articles(required_count=30)

    if not articles:
        print("No articles collected. Exiting.")
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"llm_input_springer_{timestamp}.xlsx"
    summary_path = OUTPUT_DIR / f"articles_summary_{timestamp}.txt"

    article_data = []
    snippet_table = {}
    for idx, a in enumerate(articles):
        print(f"\nArticle {idx + 1}: {a['title']}\nURL: {a['url']}")
        snippets = filter_snippets(a["text"], FILTER_KEYWORDS)
        article_data.append({"title": a["title"], "snippets": snippets})
        snippet_table[a["title"]] = snippets

    # Build payload for token count estimation
    max_per_article = 3
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
    print(f"Estimated input token count: {token_count}")
    save_article_summary(articles, token_count, summary_path)

    # Save snippets to Excel
    max_rows = max(
        (len(a["snippets"]) for a in article_data), default=0)
    snippet_df = pd.DataFrame()
    for a in article_data:
        padded = a["snippets"] + [""] * (max_rows - len(a["snippets"]))
        snippet_df[a["title"]] = padded

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        snippet_df.to_excel(
            writer, sheet_name="Snippets", index=False)

    print(f"\n[INFO] Snippets saved to: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    run_extraction()