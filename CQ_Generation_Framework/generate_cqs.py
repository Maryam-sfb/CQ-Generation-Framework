import datetime
import openai
import time
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from utils import load_environment_variables, initialize_clients
from extract_articles import (
    OUTPUT_DIR,
    MAIN_DOMAIN_NAME,
    ONTOLOGY_COVERAGE_AREAS,
    scope_text,
    estimate_tokens
)

load_environment_variables()
deployment_name, serpapi_api_key = initialize_clients()

# ========== Config ==========
SYSTEM_PROMPT = "You are ChatGPT, a helpful assistant."
MAX_TOKENS_GEN = 8000
PRESENCE_PENALTY = 0.0


@dataclass(frozen=True)
class GenConfig:
    temperature: float
    top_p: float
    freq_penalty: float

    def tag(self):
        return f"T{self.temperature}-P{self.top_p}-F{self.freq_penalty}"


# ========== LLM call ==========
def chat_call(
        content: str,
        temperature: float,
        top_p: float,
        freq_penalty: float,
        max_tokens: int) -> str:
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


# ========== Prompt ==========
PROMPT_CQ = """
You are a **Senior Ontology Engineer** with deep domain expertise 
in **{MAIN_DOMAIN_NAME}**.

Your task is to generate Competency Questions (CQs) that will guide 
ontology development. You combine domain knowledge with ontological 
thinking to create questions that are:
- Practically relevant to domain practitioners
- Technically implementable in an ontology
- Aligned with ontology engineering best practices

Below are **real excerpts** from articles related to 
**{MAIN_DOMAIN_NAME}**:
{payload}

Based on these articles, generate competency questions (CQs) suitable 
for ONTOLOGY DESIGN in the following domain:
{scope_text}

Focus on questions that help define:
- Classes and hierarchies
- Properties and relationships  
- Data structures and taxonomies
- Domain scope and boundaries

Note: Avoid "how" and "why" questions. Instead ask questions with 
"What", "Which", "What [entities]..., "What are...", "What is...", 
"What types", "What categories", "What properties", and 
"What relationships".

I'll give you examples of BAD CQs and GOOD CQs for ontology 
development. Generate CQs following the GOOD pattern:

BAD: "How does sleep quality affect cognitive function?"
GOOD: "What types of sleep quality measurements are collected?"

BAD: "How do construction practices contribute to risk?"  
GOOD: "What categories of construction practices affect building risk?"

BAD: "How do assessments correlate with diagnoses?"
GOOD: "What relationships exist between assessments and diagnoses?"

BAD: "How does building age influence vulnerability?" 
GOOD: "What properties describe building vulnerability factors?"

GOOD: "What criteria define mild vs. moderate cognitive impairment?"
GOOD: "What components constitute a comprehensive neuropsychological 
assessment?"

Now your task is to:
1. Generate at least 80 Competency Questions (CQs) from the provided 
   content only.  
   - **Do not use your general knowledge**. Only base your competency 
     questions on the given text snippets.   

2. The CQs must align with the following **Ontology Coverage Areas**:  
   **{ONTOLOGY_COVERAGE_AREAS}** 
   CRITICAL: Ensure CQs diversity. Generate questions covering all 
   coverage areas.

3. Each CQ must be:  
   - **Domain-relevant**: CQs must be in the domain.   
   - **Clear and precise**: CQs should be clear and understandable 
     for all stakeholders.  
   - **Simple and ontology-driven**: Aiming to define the scope and 
     structure of the knowledge base rather than answer complex or 
     analytical questions.
   - **Atomic**: An "atomic" Competency Question expresses a single, 
     indivisible information requirement, free of compound concepts 
     that can be logically split.

4. Output format:  
   - One CQ per line, numbered sequentially  
   - Avoid duplication of meaning (no semantic redundancies)  
"""


# ========== Load snippets ==========
def load_latest_snippets() -> tuple[pd.DataFrame, Path]:
    """
    Load the most recent llm_input_springer Excel file
    from the output folder.
    """
    files = sorted(OUTPUT_DIR.glob("llm_input_springer_*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"No llm_input_springer_*.xlsx files found in "
            f"{OUTPUT_DIR}. Please run extract_articles.py first."
        )
    latest = files[-1]
    print(f"[INFO] Loading snippets from: {latest.name}")
    df = pd.read_excel(latest, sheet_name="Snippets")
    return df, latest


# ========== Main ==========
def run_generation(snippet_path: Path = None) -> None:
    """
    Load snippets from Excel and generate CQs.
    Saves CQs sheet to the same Excel file as snippets.
    """
    # Load snippets
    if snippet_path:
        print(f"[INFO] Loading snippets from: {snippet_path.name}")
        snippet_df = pd.read_excel(snippet_path, sheet_name="Snippets")
        out_path = snippet_path
    else:
        snippet_df, out_path = load_latest_snippets()

    # Rebuild payload
    article_data = []
    for col in snippet_df.columns:
        snippets = [
            s for s in snippet_df[col].dropna().tolist()
            if str(s).strip()
        ]
        article_data.append({"title": col, "snippets": snippets})

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

    # Generate CQs
    config = GenConfig(temperature=0.3, top_p=1.0, freq_penalty=0.2)
    print("\nGenerating competency questions...")
    coverage_text = "\n- " + "\n- ".join(ONTOLOGY_COVERAGE_AREAS)

    resp = chat_call(
        PROMPT_CQ.format(
            payload=payload,
            scope_text=scope_text,
            MAIN_DOMAIN_NAME=MAIN_DOMAIN_NAME,
            ONTOLOGY_COVERAGE_AREAS=coverage_text
        ),
        temperature=config.temperature,
        top_p=config.top_p,
        freq_penalty=config.freq_penalty,
        max_tokens=MAX_TOKENS_GEN
    )

    # Parse CQs
    rows = []
    for line in resp.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            rows.append({
                "Config": config.tag(),
                "Source": "scholar+publishers",
                "CQ": line
            })

    df_cq = pd.DataFrame(rows)

    # Save to the Excel file
    with pd.ExcelWriter(
            out_path, engine="xlsxwriter") as writer:
        df_cq.to_excel(writer, sheet_name="CQs", index=False)
        snippet_df.to_excel(writer, sheet_name="Snippets", index=False)

    print(f"\nResults saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_generation()