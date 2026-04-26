import os
import openai
import json
import pandas as pd
from dotenv import load_dotenv
from time import sleep
from pathlib import Path
from datetime import datetime
from typing import Dict

# ========== Load Environment Variables ==========
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


# ========== Load domain information from JSON ==========
def load_domain_config(config_path: str = "json_input/domain-info.json") -> Dict:
    """
    Load domain configuration from JSON file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        return {}

domain_config = load_domain_config("json_input/domain-info.json")
MAIN_DOMAIN_NAME = domain_config.get("MAIN_DOMAIN_NAME", "unknown")  # fallback value
print(f"[DEBUG] Domain loaded: {MAIN_DOMAIN_NAME}")

# ========== Prompt Template ==========
PROMPT_TEMPLATE = """
You are an expert in the field of **{MAIN_DOMAIN_NAME}**.

Your task is to **replace real-world entities** (like the name of technologies, materials, disasters, locations, organizations, diseases) in the following question in the **{MAIN_DOMAIN_NAME}**, with a **generalized concept**.
For example the competency question: What is the effect of earthquake load on masonry walls? should be changed to: What is the effect of the named hazard load on the named building element?

Please make sure:
1. The real-world entity is removed after replacing it with generalized concept.
2. You don't need to abstract all questions. Some questions don't need named entity abstraction. 
3. Only abstract real-world entity names which are not general names for the domain. For example 'building', 'construction', 'infrastructure' and 'cognitive' are not entity names and can not be abstracted.
---

Below are more examples.

Example 1:
input: What challenges exist in using computer vision for damage assessment of timber structures?
output: What challenges exist in using the named technology for damage assessment of the named structures?

Example 2:
input: What repair methods are used for concrete buildings affected by flooding?
output: What repair methods are used for this type of buildings affected by the named hazard?

Example 3:
input: What characteristics define slightly damaged buildings?
output: What characteristics define slightly damaged buildings?

Example 4: 
input: What is the significance of deep learning in structural health monitoring?
output: What is the significance of the named technology in structural health monitoring?

Example 5:
input: What are the potential risks associated with relying solely on automated damage detection methods?
output: What are the potential risks associated with relying solely on the named damage detection methods?

Example 6:
input: How does volcanic ash fall affect building integrity?
output: How does the named hazard affect building integrity?

Example 7:
input: What are the common methods for early detection of Alzheimer’s disease?
output: What are the common methods for early detection of the named disease?
---

Now abstract the following:

Original: {QUESTION}
Abstracted:
"""

# ========== OpenAI Abstraction Call ==========
def abstract_question(question: str, domain: str) -> str:
    prompt = PROMPT_TEMPLATE.format(MAIN_DOMAIN_NAME=domain, QUESTION=question)
    try:
        response = openai.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
            frequency_penalty=0,
            presence_penalty=0,
        )
        result = response.choices[0].message.content.strip()
        return result.split("Abstracted:")[-1].strip() if "Abstracted:" in result else result
    except Exception as e:
        print("OpenAI error (abstraction):", e)
        return question  # fallback

# ========== Main Process ==========

# Find latest Excel file in output folder based on timestamp in filename
output_dir = Path(__file__).resolve().parent / "output"
excel_files = list(output_dir.glob("llm_input_springer_*.xlsx"))

if not excel_files:
    raise FileNotFoundError("No matching Excel files found in output/")

# Extract datetime from filenames and pick the latest
def extract_timestamp(f: Path) -> datetime:
    # filename pattern: llm_input_springer_YYYYMMDD_HHMMSS.xlsx
    ts_str = f.stem.replace("llm_input_springer_", "")
    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")

latest_file = max(excel_files, key=extract_timestamp)
input_file = latest_file

timestamp_str = latest_file.stem.replace("llm_input_springer_", "")
output_file = output_dir / f"refined_cqs_springer_{timestamp_str}.xlsx"

print(f"Using latest input file: {input_file.name}")
print(f"Output file will be: {output_file.name}")

df = pd.read_excel(input_file, sheet_name="CQs")
questions = df["CQ"].tolist()

abstracted_questions = []

print("Processing...")

for i, q in enumerate(questions, 1):
    print(f"Abstracting Q{i}/{len(questions)}")
    abstracted = abstract_question(q, MAIN_DOMAIN_NAME)
    abstracted_questions.append(abstracted)
    sleep(1)  # avoid hitting API rate limits

# Save results (only abstraction, no atomicity)
df["Abstracted CQ"] = abstracted_questions
df.to_excel(output_file, index=False)
print(f"\nDone. Results saved to: {output_file}")


