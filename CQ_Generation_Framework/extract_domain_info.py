import os
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import openai

# ========== Load Environment Variables ==========
def load_environment_variables() -> None:
    """
    Load environment variables from the .env file.
    """
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)

load_environment_variables()

openai.api_key = os.getenv("OPENAI_API_KEY_5")
openai.api_type = os.getenv("OPENAI_API_TYPE_5")
openai.api_version = os.getenv("OPENAI_API_VERSION_5")
openai.azure_endpoint = os.getenv("OPENAI_API_BASE_5")
deployment_name = os.getenv("DEPLOYMENT_NAME_5") or "gpt-5"
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

# ========== Load Expert Scope ==========
with open("json_input/scope-expert.json", "r", encoding="utf-8") as f:
    scope_json = json.load(f)

scope_text = "\n".join(item["response"] for item in scope_json)

# ========== LLM Helper ==========
def call_llm(prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
    try:
        response = openai.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful ontology engineering assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return ""

# ========== Prompt ==========
prompt = f"""
You are a **Senior Ontology Engineer** specializing in requirements analysis for knowledge-based systems. 
You are given domain expert responses to four questions regarding domain, end users, purpose, and use-cases of the ontology to be developed.

{scope_text}

From this, extract and return a JSON object with the following keys:

1. "MAIN_DOMAIN_NAME" : a short title for domain in 5 to 10 words, representing the domain core area.  
2. "TOPIC_TERMS": a list of 20 to 30 domain-relevant compound topic terms (two or three-word phrases) that can be used for searching articles in this domain.
3. "FILTER_KEYWORDS": a list of 30 to 50 filter keywords directly relevant to the domain (single or short terms) that can be used for filtering articles.
4. "MAIN_DOMAIN_WORDS": a list of 3 to 5 core (single) words directly relevant to the domain.
5. "COMPOUND_GENERAL_TERMS": 2 core compound terms that are directly relevant to the domain and used to query articles in the domain.
6. "ONTOLOGY_COVERAGE_AREAS": a list of 5 to 10 high-level ontology coverage areas.

An example for the Domain of "Building damage and construction defect analysis" is as below:

Input:
[{
  "question": "Describe the knowledge domain or subject area in two to three sentences.",
  "response": "The knowledge domain or subject area includes description and documentation of damages to buildings, including corresponding explanations and determination of causal effects. Recommendations for remediation and prevention should also be included. The basis for the ontology should consist of real-world damage cases."
},
{
"question": "Who is the intended audience for the created knowledge system and who will work with it in the future?",
"response": "The intended users of the created knowledge system are experts in construction who deal with damage analysis, remediation, repair, and prevention in the context of the building industry. Architects and surveyors/experts in the field of construction. Additionally, lawyers (attorneys and judges) will use the data if a damage case is subject to legal proceedings. Beyond that, the data serves as a resource and learning material for anyone looking to further educate their expertise in this field."
},
  {
    "question": "Name specific problems in the subject area that are to be addressed by the knowledge system.",
    "response": "The database is the central point of contact for specific real-world examples and the current state of the art for the development, prevention and remediation of damage and defects in building construction, both for the assessment of damage patterns in buildings and for the new construction and implementation of necessary measures regarding buildings."
  },
  {
    "question": "Describe some use cases of the knowledge system in short sentences.",
    "response": "The use-cases of the knowledge system are as below: The knowledge system should include specific documents and sources related to explicit queries and present relevant content in a structured and understandable manner. If a defect appears in a building, a damage expert should be able to search the database for relevant content referencing the current state of the art in the field and applicable norms. Ideally, they will find a description of a similar damage scenario from past reports and guidance on how to prevent or remedy such defects."
  }
]

Output:
{
  "MAIN_DOMAIN_NAME": "Building Damage and Construction Defects",
  "TOPIC_TERMS": [
    "Building Damage",
    "Remediation Techniques",
    "Construction Defects",
    "Legal Proceedings",
    "Damage Prevention",
    "Expert Recommendations",
    "Causal Effects",
    "Damage Documentation",
    "Building Assessment",
    "Repair Strategies",
    "Construction Norms",
    "Damage Patterns",
    "Foundation Problems",
    "Structural Fatigue"
  ],
  "FILTER_KEYWORDS": [
    "building",
    "remediation",
    "prevention",
    "construction",
    "defect",
    "damage",
    "analysis",
    "legal",
    "assessment",
    "pattern",
    "norms",
    "repair",
    "documentation",
    "source",
    "guidance"
  ],
  "MAIN_DOMAIN_WORDS": [
    "building",
    "construction",
    "damage"
  ],
  "COMPOUND_GENERAL_TERMS": [
    "Building Maintenance",
    "Construction Defects"
  ],
  "ONTOLOGY_COVERAGE_AREAS": [
    "Damage Analysis",
    "Remediation Strategies",
    "Legal Implications",
    "Construction Standards",
    "Damage Prevention"
  ]
}

Return ONLY valid JSON in the following format:
{{
  "MAIN_DOMAIN_NAME": [...],
  "TOPIC_TERMS": [...],
  "FILTER_KEYWORDS": [...],
  "MAIN_DOMAIN_WORDS": [...],
  "COMPOUND_GENERAL_TERMS": [...],
  "ONTOLOGY_COVERAGE_AREAS": [...]
}}
"""

# ========== Run Extraction ==========
result_text = call_llm(prompt)
try:
    domain_info = json.loads(result_text)
except json.JSONDecodeError:
    print("Model did not return valid JSON, here’s the raw output:")
    print(result_text)
    domain_info = {}

# ========== Save Output ==========
# Save JSON
with open("json_input/domain-info.json", "w", encoding="utf-8") as f:
    json.dump(domain_info, f, indent=2)

# Convert to DataFrame with all values in one sheet
max_len = max(len(v) if isinstance(v, list) else 1 for v in domain_info.values())
data = {}
for key, values in domain_info.items():
    if isinstance(values, list):
        padded = values + [""] * (max_len - len(values))
    else:
        padded = [values] + [""] * (max_len - 1)
    data[key] = padded

df = pd.DataFrame(data)
df.to_excel("domain_info.xlsx", sheet_name="DomainInfo", index=False)

print("\nExtracted domain info saved to domain-info.json and domain_info.xlsx")
