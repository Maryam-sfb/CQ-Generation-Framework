import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
import spacy
import numpy as np
from datetime import datetime

# ========== Load ENV ==========
def load_environment_variables() -> None:
    """
    Load environment variables from the .env file.
    """
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)


load_environment_variables()

openai.api_key = os.getenv("OPENAI_API_KEY_4o")
openai.api_type = os.getenv("OPENAI_API_TYPE_4o")
openai.api_version = os.getenv("OPENAI_API_VERSION_4o")
openai.azure_endpoint = os.getenv("OPENAI_API_BASE_4o")
deployment_name = os.getenv("DEPLOYMENT_NAME_4o") or "gpt-4o"
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

print("[DEBUG] OpenAI ENV loaded:")
print("KEY:", bool(openai.api_key))
print("BASE:", openai.azure_endpoint)
print("DEPLOYMENT:", deployment_name)

openai.api_key = openai.api_key

if openai.azure_endpoint:
    setattr(openai, "api_base", openai.azure_endpoint)  # avoid mypy attr error

if openai.api_type:
    if openai.api_type not in ("openai", "azure"):
        raise ValueError(f"Invalid OPENAI_API_TYPE_4o value: {openai.api_type}")
    setattr(openai, "api_type", openai.api_type)

if openai.api_version:
    setattr(openai, "api_version", openai.api_version)

# Initialize local models
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Load spaCy model for linguistic complexity
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Configurable weights
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", 0.7))
SYNTACTIC_WEIGHT = float(os.getenv("SYNTACTIC_WEIGHT", 0.3))
REDUNDANCY_THRESHOLD = float(os.getenv("REDUNDANCY_THRESHOLD", 0.75))
RELEVANCE_WEIGHT = float(os.getenv("RELEVANCE_WEIGHT", 0.7))  # 70% weight to relevance
COMPLEXITY_WEIGHT = float(os.getenv("COMPLEXITY_WEIGHT", 0.3))  # 30% weight to (inverted) complexity
FINAL_THRESHOLD = float(os.getenv("FINAL_THRESHOLD", 0.5))

# ========== Load Domain Info ==========
domain_info_file = Path(__file__).resolve().parent / "revised_DFC_domain_info.json"
with open(domain_info_file, "r", encoding="utf-8") as f:
    domain_info = json.load(f)

MAIN_DOMAIN_NAME = domain_info.get("MAIN_DOMAIN_NAME", "Unknown Domain")

all_items = []

# Iterate through all key-value pairs in the domain_info dictionary
for key, value in domain_info.items():
    # Add the key itself
    all_items.append(key)

    # Add the values
    if isinstance(value, list):
        all_items.extend(value)
    else:
        all_items.append(str(value))

scope_text = "\n".join(all_items)

# ========== Normalize Relevance Score ==========
def normalize_relevance(relevance: int) -> float:
    """Map relevance (1–4) to a custom 0–1 scale with more weight on 3 and 4."""
    if relevance == 1:
        return 0.0
    elif relevance == 2:
        return 0.5
    elif relevance == 3:
        return 0.75
    elif relevance == 4:
        return 1.0
    else:
        return 0.0   # safe fallback

# ========== Linguistic Complexity Analysis ==========
def analyze_linguistic_complexity(text: str) -> dict:
    """
    Analyze linguistic complexity using spaCy
    """
    doc = nlp(text)

    # Count various linguistic features
    noun_phrases = len(list(doc.noun_chunks))
    verbs = len([token for token in doc if token.pos_ == "VERB"])
    prepositions = len([token for token in doc if token.pos_ == "ADP"])
    conjunctions = len([token for token in doc if token.pos_ in ["CCONJ", "SCONJ"]])
    modifiers = len([token for token in doc if token.pos_ in ["ADJ", "ADV"]])

    # Question type analysis
    question_words = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}
    first_word = doc[0].text.lower() if len(doc) > 0 else ""

    if first_word in question_words:
        question_type = "WH-question"
    elif first_word == "is" or first_word == "are" or first_word == "do" or first_word == "does":
        question_type = "Yes/No"
    elif "how many" in text.lower() or "how much" in text.lower():
        question_type = "Quantitative"
    else:
        question_type = "Other"

    # Calculate complexity score (weights can be adjusted as needed)
    complexity_score = (
            noun_phrases * 0.3 +
            verbs * 0.2 +
            prepositions * 0.15 +
            conjunctions * 0.15 +
            modifiers * 0.2
    )

    # Normalize complexity score to 0-1 range (adjust scaling factor as needed)
    normalized_complexity = min(complexity_score / 5.0, 1.0)

    return {
        "noun_phrases": noun_phrases,
        "verbs": verbs,
        "prepositions": prepositions,
        "conjunctions": conjunctions,
        "modifiers": modifiers,
        "question_type": question_type,
        "complexity_score": round(normalized_complexity, 3)
    }


# ========== Relevance check ==========
def get_relevance_score(cq: str) -> int:
    prompt = f"""
    You are an experienced expert in the field of **{MAIN_DOMAIN_NAME}**. Below is a transcript describing the ontology scope, keywords and coverage areas:

{scope_text}

Your task is to determine whether a given competency question (CQ) is relevant to the domain of this ontology. 

Rate the following competency question (CQ) on a 4-point Likert scale for domain relevance:
(4) Explicitly matches requirements of the ontology.
(3) Implicit but clearly inferable requirement necessary for ontology goals.
(2) Only tangentially related — loosely connected but not necessary for ontology goals.
(1) Irrelevant — not expressed, not inferable, and not useful for this ontology.

CQ: "{cq}"

Answer only with a single number: 1, 2, 3, or 4.

Be strict: If the question is not clearly useful for the ontology scope, score it 1. Do not hesitate to give a score of 1 when in doubt.

"""
    try:
        response = openai.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful ontology engineering assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5,
        )
        score = int(response.choices[0].message.content.strip())
        return score
    except Exception as e:
        print(f"LLM relevance error for CQ '{cq}':", e)
        return 1

# ========== Redundancy Functions ==========
def get_embedding(text: str) -> np.ndarray:
    """Return a numpy embedding vector (safe fallback to zeros of correct dimension)."""
    try:
        # SentenceTransformer.encode usually returns a numpy array already
        emb = embedding_model.encode(text, convert_to_numpy=True)
        return np.array(emb, dtype=float)
    except Exception as e:
        print(f"Local embedding error for text (fallback to zeros): {e}")
        # use model API to get correct dimension (works for SentenceTransformer)
        try:
            dim = embedding_model.get_sentence_embedding_dimension()
        except Exception:
            dim = 768  # conservative fallback if model introspection fails
        return np.zeros((dim,), dtype=float)


def tokenize(text: str) -> set:
    return set(re.findall(r"\b\w+\b", text.lower()))


def syntactic_similarity(q1: str, q2: str) -> float:
    tokens1, tokens2 = tokenize(q1), tokenize(q2)
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def remove_redundant_questions(questions: list, threshold: float = 0.75, prefer_relevance: bool = False) -> tuple:
    """
    Robust redundancy removal:
      - compute full semantic similarity matrix (cosine)
      - compute syntactic similarity matrix
      - combine by weights and build adjacency where combined >= threshold
      - find connected components (clusters of similar questions)
      - in each component, pick one representative to keep (shortest by default,
        or highest relevance if prefer_relevance=True)
    Returns: (sorted list of kept indices, redundancy_info list)
    """
    n = len(questions)
    if n == 0:
        return [], []

    # 1) embeddings matrix (n x dim)
    embeddings = np.vstack([get_embedding(q) for q in questions])  # shape (n, dim)
    # Defensive: if some embeddings are zero vectors, cosine_similarity can behave badly; clip later
    try:
        semantic_matrix = cosine_similarity(embeddings)
        # Clip negatives to 0 (we want similarity in [0,1] for combining)
        semantic_matrix = np.clip(semantic_matrix, 0.0, 1.0)
    except Exception as e:
        print("Semantic similarity matrix build error:", e)
        semantic_matrix = np.zeros((n, n), dtype=float)

    # 2) syntactic similarity matrix
    tokens = [tokenize(q) for q in questions]
    syntactic_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            if not tokens[i] or not tokens[j]:
                s = 0.0
            else:
                s = len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j])
            syntactic_matrix[i, j] = s
            syntactic_matrix[j, i] = s

    # 3) combined similarity
    combined = SEMANTIC_WEIGHT * semantic_matrix + SYNTACTIC_WEIGHT * syntactic_matrix

    # 4) adjacency graph: edges where combined >= threshold (ignore self-edges)
    adj = combined >= threshold
    np.fill_diagonal(adj, False)

    # 5) find connected components (clusters)
    visited = set()
    components = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        comp = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            # neighbors where adj[node] is True
            neighbors = np.where(adj[node])[0].tolist()
            for nb in neighbors:
                if nb not in visited:
                    stack.append(nb)
        components.append(sorted(comp))

    # 6) choose representative for each component
    to_keep = set()
    redundancy_info = []
    for comp in components:
        if len(comp) == 1:
            to_keep.add(comp[0])
            continue

        # Decide which index to keep
        if prefer_relevance:
            # compute relevance for members and pick highest score (tie -> shortest)
            relevances = []
            for idx in comp:
                try:
                    r = get_relevance_score(questions[idx])
                except Exception:
                    r = 1
                relevances.append((r, idx))
            # pick max relevance, break ties by shorter text
            relevances.sort(key=lambda t: (-t[0], len(questions[t[1]])))
            keep_idx = relevances[0][1]
        else:
            # simple heuristic: keep the shortest question (by character length).
            keep_idx = min(comp, key=lambda idx: len(questions[idx]))

        removed = sorted([idx for idx in comp if idx != keep_idx])
        to_keep.add(keep_idx)

        redundancy_info.append({
            "component_members": comp,
            "kept_index": keep_idx,
            "removed_indices": removed,
            "kept_question": questions[keep_idx],
            "removed_questions": [questions[r] for r in removed]
        })

    return sorted(to_keep), redundancy_info


# ========== Main Filtering ==========
def cluster_questions(questions: list) -> pd.DataFrame:
    prompt = f"""
    You are an expert in the field of **{MAIN_DOMAIN_NAME}**.

    Follow these steps to cluster the competency questions:

    Step 1: Read through all {len(questions)} questions and identify recurring themes or topics.
    Step 2: Determine 5-7 main themes that best organize these questions.
    Step 3: Assign each question to the most appropriate theme based on its focus.
    Step 4: Create meaningful cluster titles that capture each theme.
    Step 5: Verify that ALL {len(questions)} questions are included (none omitted).

    Return your result as JSON:
    [{{"Cluster": "title", "Questions": ["Q1", "Q2"]}}]

    CQs:
    {json.dumps(questions, indent=2, ensure_ascii=False)}
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system",
                     "content": "You are an ontology engineering assistant skilled in thematic clustering. You MUST include every question in your output."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=5000,
            )
            raw_output = response.choices[0].message.content.strip()

            # Try to extract JSON if wrapped in markdown or has extra text
            match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if match:
                raw_output = match.group(0)

            clusters = json.loads(raw_output)

            # VALIDATION: Check if all questions are included
            clustered_questions = []
            for cluster in clusters:
                clustered_questions.extend(cluster["Questions"])

            questions_set = set(questions)
            clustered_set = set(clustered_questions)
            missing = questions_set - clustered_set

            if missing:
                print(f"WARNING: {len(missing)} question(s) missing (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue  # Retry
                else:
                    # Add missing questions to Uncategorized cluster
                    print(f"Adding {len(missing)} missing questions to 'Uncategorized' cluster")
                    clusters.append({
                        "Cluster": "Uncategorized",
                        "Questions": list(missing)
                    })

            # Build DataFrame
            rows = []
            for cluster in clusters:
                for q in cluster["Questions"]:
                    rows.append({
                        "Cluster": cluster["Cluster"],
                        "Question": q
                    })

            df = pd.DataFrame(rows)
            print(f"✓ Clustered {len(df)} questions (expected {len(questions)})")
            return df

        except Exception as e:
            print(f"Clustering error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                # Fallback: single cluster with all questions
                return pd.DataFrame([{"Cluster": "All Questions", "Question": q} for q in questions])
            time.sleep(1)


def filter_cqs(input_file: str, output_file: str, renumber_sequential: bool = False):
    start_time = time.time()
    df = pd.read_excel(input_file)

    # Keep a dataframe of rows that actually have an Abstracted CQ, but preserve original row index in column 'orig_row'
    non_null_df = df[df["Abstracted CQ"].notna()].reset_index().rename(columns={"index": "orig_row"})
    original_questions = non_null_df["Abstracted CQ"].tolist()

    print(f"Starting with {len(original_questions)} questions (non-null)")

    # Step 1: Remove redundant questions (indices are relative to original_questions / non_null_df)
    print("Removing redundant questions...")
    kept_positions, redundancy_info = remove_redundant_questions(original_questions, REDUNDANCY_THRESHOLD)
    # kept_positions are integer positions into non_null_df (0..N-1)

    # Build a DataFrame of the non-redundant rows WITH their original row numbers preserved
    non_redundant_df = non_null_df.iloc[kept_positions].reset_index(drop=True)

    non_redundant_questions = non_redundant_df["Abstracted CQ"].tolist()
    print(f"After redundancy removal: {len(non_redundant_questions)} questions kept")

    # Step 2: Calculate relevance scores for non-redundant questions
    print("Calculating relevance scores...")
    relevance_scores = []
    for i, cq in enumerate(non_redundant_questions):
        print(f"  Processing question {i + 1}/{len(non_redundant_questions)}")
        relevance = get_relevance_score(cq)
        relevance_scores.append(relevance)
        time.sleep(0.5)

    # Step 3: Calculate linguistic complexity for non-redundant questions
    print("Calculating linguistic complexity...")
    complexity_scores = []
    for cq in non_redundant_questions:
        complexity = analyze_linguistic_complexity(cq)
        complexity_scores.append(complexity["complexity_score"])

    # Step 4: Create results using the preserved original row number (orig_row)
    results = []
    for i, row in non_redundant_df.iterrows():
        orig_row_num = int(row["orig_row"])  # original df row index (0-based)
        cq = row["Abstracted CQ"]
        relevance = relevance_scores[i]
        complexity_score = complexity_scores[i]

        normalized_relevance = normalize_relevance(relevance)
        final_score = (RELEVANCE_WEIGHT * normalized_relevance) + (COMPLEXITY_WEIGHT * (1 - complexity_score))

        should_keep = "Yes" if final_score >= FINAL_THRESHOLD else "No"

        # Option A: keep original spreadsheet row number (1-based for human readability)
        if not renumber_sequential:
            question_number = orig_row_num + 1
        else:
            # Option B: renumber sequentially in the filtered sheet
            question_number = i + 1

        results.append({
            "Question #": question_number,
            "CQ": cq,
            "Relevance": relevance,
            "Complexity_Score": round(complexity_score, 3),
            "Final Score": round(final_score, 2),
            "Keep?": should_keep
        })

    # Step 5: Build DataFrame and sort
    filtered_questions_df = pd.DataFrame(results)
    filtered_questions_df = filtered_questions_df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)

    # Step 6: Cluster only the final kept questions
    final_kept_cqs = filtered_questions_df[filtered_questions_df["Keep?"] == "Yes"]["CQ"].tolist()
    cluster_df = cluster_questions(final_kept_cqs)

    # Step 7: Map redundancy_info indices back to original df row numbers (for human-readable Excel)
    mapped_redundancy = []
    for comp in redundancy_info:
        # comp fields expected: component_members, kept_index, removed_indices, kept_question, removed_questions
        mapped = {
            "component_members_orig_rows": [int(non_null_df.iloc[idx]["orig_row"]) + 1 for idx in comp.get("component_members", [])],
            "kept_orig_row": int(non_null_df.iloc[comp["kept_index"]]["orig_row"]) + 1 if "kept_index" in comp else None,
            "removed_orig_rows": [int(non_null_df.iloc[idx]["orig_row"]) + 1 for idx in comp.get("removed_indices", [])],
            "kept_question": comp.get("kept_question"),
            "removed_questions": comp.get("removed_questions", [])
        }
        mapped_redundancy.append(mapped)
    redundancy_df = pd.DataFrame(mapped_redundancy)

    # Step 8: Save results to Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        filtered_questions_df.to_excel(writer, sheet_name="Filtered_CQs", index=False)
        cluster_df.to_excel(writer, sheet_name="Clusters", index=False)
        if not redundancy_df.empty:
            redundancy_df.to_excel(writer, sheet_name="Redundancy_Details", index=False)

    elapsed_time = time.time() - start_time
    print(f"\nResults saved to {output_file}")
    print(f"Total runtime: {elapsed_time:.2f} seconds")
    print(f"Original questions (non-null): {len(original_questions)}")
    print(f"After redundancy removal: {len(non_redundant_questions)}")
    print(f"Final kept questions: {len(final_kept_cqs)}")


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "output"
    refined_files = list(output_dir.glob("refined_cqs_springer_*.xlsx"))

    if not refined_files:
        raise FileNotFoundError("No matching refined_cqs_springer_*.xlsx files found in output/")

    # Extract timestamp from filenames and pick the latest
    def extract_timestamp(f: Path) -> datetime:
        # filename pattern: refined_cqs_springer_YYYYMMDD_HHMMSS.xlsx
        ts_str = f.stem.replace("refined_cqs_springer_", "")
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")

    latest_file = max(refined_files, key=extract_timestamp)
    input_file = latest_file

    # Create output filename with same timestamp
    timestamp_str = latest_file.stem.replace("refined_cqs_springer_", "")
    output_file = output_dir / f"joint_filtered_cqs_{timestamp_str}.xlsx"

    print(f"Using latest refined file: {input_file.name}")
    print(f"Output file will be: {output_file.name}")

    filter_cqs(str(input_file), str(output_file))
