import pandas as pd
from openai import AzureOpenAI
import json
import os
from time import sleep
from pathlib import Path
from dotenv import load_dotenv
from scipy import stats


# Load your environment variables
def load_environment_variables() -> None:
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)


load_environment_variables()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY_4o"),
    api_version=os.getenv("OPENAI_API_VERSION_4o"),
    azure_endpoint=os.getenv("OPENAI_API_BASE_4o")
)

deployment_name = os.getenv("DEPLOYMENT_NAME_4o") or "gpt-4o"

# ============================================================================
# LOAD DOMAIN INFORMATION FROM JSON
# ============================================================================
print("=" * 70)
print("LOADING DOMAIN INFORMATION")
print("=" * 70)

with open('domain_info_DFC.json', 'r') as f:
    domain_info = json.load(f)

print(f"\nMain Domain: {domain_info['MAIN_DOMAIN_NAME']}")
print(f"Topic Terms: {len(domain_info['TOPIC_TERMS'])} terms")
print(f"Coverage Areas: {len(domain_info['ONTOLOGY_COVERAGE_AREAS'])} areas")

# Format domain info for the prompt
coverage_areas_str = "\n   - ".join(domain_info['ONTOLOGY_COVERAGE_AREAS'])
topic_terms_str = ", ".join(domain_info['TOPIC_TERMS'][:10]) + "..."  # First 10 for brevity

print("\n" + "=" * 70)
print("DOMAIN CONTEXT LOADED")
print("=" * 70)

# ============================================================================
# LOAD AND CLEAN CQs
# ============================================================================
expert_df = pd.read_excel('DFC_CQs.xlsx')
generated_df = pd.read_excel('LLM_generated_DFC_CQs.xlsx')

print("\nExpert columns:", expert_df.columns.tolist())
print("Generated columns:", generated_df.columns.tolist())

print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)


def extract_and_clean_cqs(df, source_name):
    cq_column = 'CQ'
    print(f"Raw data shape for {source_name}: {df.shape}")

    questions = df[cq_column].dropna().tolist()
    print(f"After dropping NaN for {source_name}: {len(questions)} questions")

    common_headers = ['question', 'cq', 'competency question', 'competency questions', 'cqs']
    cleaned_questions = []

    for q in questions:
        q_str = str(q).strip()
        if (q_str.lower() in common_headers or q_str == '' or q_str.isdigit()):
            continue
        cleaned_questions.append(q_str)

    print(f"After removing headers/empty for {source_name}: {len(cleaned_questions)} questions")

    unique_questions = []
    seen_questions = set()

    for q in cleaned_questions:
        normalized_q = q.lower().strip().rstrip('?')
        if normalized_q not in seen_questions:
            seen_questions.add(normalized_q)
            unique_questions.append(q)
        else:
            print(f"Removing duplicate from {source_name}: '{q}'")

    print(f"After removing duplicates for {source_name}: {len(unique_questions)} questions")
    return unique_questions


expert_questions = extract_and_clean_cqs(expert_df, 'expert')
generated_questions = extract_and_clean_cqs(generated_df, 'generated')

expert_cqs = pd.DataFrame({
    'cq_id': [f"expert_{i + 1}" for i in range(len(expert_questions))],
    'question': expert_questions,
    'source': 'expert'
})

generated_cqs = pd.DataFrame({
    'cq_id': [f"generated_{i + 1}" for i in range(len(generated_questions))],
    'question': generated_questions,
    'source': 'generated'
})

print(f"\nFinal counts:")
print(f"Expert CQs: {len(expert_cqs)}")
print(f"Generated CQs: {len(generated_cqs)}")
print(f"Total CQs to evaluate: {len(expert_cqs) + len(generated_cqs)}")

all_cqs = pd.concat([expert_cqs, generated_cqs], ignore_index=True)
shuffled_cqs = all_cqs.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nShuffled CQs: {len(shuffled_cqs)}")


# ============================================================================
# EVALUATION FUNCTION WITH DOMAIN CONTEXT
# ============================================================================
def evaluate_cq(question, cq_id, domain_info):
    """
    Evaluate a CQ using domain-specific context from JSON file.
    """
    prompt = f"""You are an expert ontology engineer evaluating Competency Questions for a PEMFC degradation ontology.

ONTOLOGY DOMAIN INFORMATION:
Main Domain: {domain_info['MAIN_DOMAIN_NAME']}

Coverage Areas (the ontology should address these):
   - {coverage_areas_str}

Key Topic Terms: {topic_terms_str}

QUESTION TO EVALUATE: {question}

Evaluate on these FOUR criteria (1-5 scale):

1. DOMAIN RELEVANCE (1-5)
   - 5: Directly addresses the main domain and coverage areas listed above
   - 4: Strongly related to the domain and topic terms
   - 3: Generally about PEMFCs but not specifically aligned with coverage areas
   - 2: Tangentially related to the domain
   - 1: Not relevant to the stated domain or coverage areas

2. ATOMICITY (1-5)
   - 5: Single, focused question addressing one concept
   - 4: Mostly focused but with minor secondary aspect
   - 3: Two related questions combined
   - 2: Multiple distinct questions combined
   - 1: Complex compound question with 3+ separate concerns

3. CLARITY (1-5)
   - 5: Unambiguous, specific terminology, clear scope
   - 4: Mostly clear with minor ambiguity
   - 3: Some vague terms or unclear scope
   - 2: Multiple ambiguous terms or unclear intent
   - 1: Very vague, ambiguous, or confusing

4. IMPORTANCE (1-5)
   - 5: Addresses a critical competency from the coverage areas (essential)
   - 4: Addresses an important aspect of the domain
   - 3: Addresses a useful but not essential aspect
   - 2: Addresses a peripheral aspect
   - 1: Does not address any important competency for this ontology

Consider the specific coverage areas when rating importance:
{coverage_areas_str}

Return ONLY valid JSON with no additional text: {{"relevance": score, "atomicity": score, "clarity": score, "importance": score}}"""

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error evaluating {cq_id}: {e}")
        return None


# ============================================================================
# CHECK DOMAIN TERM USAGE IN CQs
# ============================================================================
def check_domain_term_usage(question, domain_info):
    """
    Check which domain terms appear in the CQ.
    Returns count of domain terms found.
    """
    question_lower = question.lower()

    # Check filter keywords (most important)
    filter_keywords_found = sum(1 for kw in domain_info['FILTER_KEYWORDS']
                                if kw.lower() in question_lower)

    # Check topic terms
    topic_terms_found = sum(1 for term in domain_info['TOPIC_TERMS']
                            if term.lower() in question_lower)

    return {
        'filter_keywords': filter_keywords_found,
        'topic_terms': topic_terms_found,
        'total': filter_keywords_found + topic_terms_found
    }


# ============================================================================
# EVALUATE ALL CQs
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATING CQs WITH DOMAIN CONTEXT")
print("=" * 70)

results = []
for idx, row in shuffled_cqs.iterrows():
    print(f"Evaluating {idx + 1}/{len(shuffled_cqs)}: {row['cq_id']}")

    # Evaluate with domain context
    evaluation = evaluate_cq(row['question'], row['cq_id'], domain_info)

    # Check domain term usage
    term_usage = check_domain_term_usage(row['question'], domain_info)

    results.append({
        'cq_id': row['cq_id'],
        'question': row['question'],
        'source': row['source'],
        'evaluation_raw': evaluation,
        'domain_terms_count': term_usage['total'],
        'filter_keywords_count': term_usage['filter_keywords'],
        'topic_terms_count': term_usage['topic_terms']
    })

    sleep(1)

results_df = pd.DataFrame(results)


# ============================================================================
# PARSE EVALUATION RESULTS
# ============================================================================
def parse_scores(eval_str):
    if not eval_str:
        return {'relevance': None, 'atomicity': None, 'clarity': None, 'importance': None}
    try:
        eval_str = eval_str.replace('```json', '').replace('```', '').strip()
        scores = json.loads(eval_str)
        return scores
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic response: {eval_str}")
        return {'relevance': None, 'atomicity': None, 'clarity': None, 'importance': None}


scores_df = results_df['evaluation_raw'].apply(parse_scores).apply(pd.Series)
results_df = pd.concat([results_df, scores_df], axis=1)

# ============================================================================
# COMPARE GROUPS
# ============================================================================
comparison = results_df.groupby('source').agg({
    'relevance': ['mean', 'std', 'count'],
    'atomicity': ['mean', 'std', 'count'],
    'clarity': ['mean', 'std', 'count'],
    'importance': ['mean', 'std', 'count'],
    'domain_terms_count': ['mean', 'std']
}).round(3)

print("\n" + "=" * 70)
print("RESULTS - Quality Comparison (Domain-Informed Evaluation)")
print("=" * 70)
print(comparison)

# ============================================================================
# DOMAIN TERM USAGE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("DOMAIN TERMINOLOGY USAGE")
print("=" * 70)

for source in ['expert', 'generated']:
    source_data = results_df[results_df['source'] == source]

    print(f"\n{source.upper()}:")
    print(f"  Avg domain terms per CQ: {source_data['domain_terms_count'].mean():.2f}")
    print(f"  Avg filter keywords per CQ: {source_data['filter_keywords_count'].mean():.2f}")
    print(f"  Avg topic terms per CQ: {source_data['topic_terms_count'].mean():.2f}")
    print(f"  CQs with 0 domain terms: {(source_data['domain_terms_count'] == 0).sum()}")
    print(f"  CQs with 3+ domain terms: {(source_data['domain_terms_count'] >= 3).sum()}")

# ============================================================================
# STATISTICAL SIGNIFICANCE
# ============================================================================
print("\n" + "=" * 70)
print("Statistical Significance (Mann-Whitney U Test with Effect Sizes)")
print("=" * 70)

statistical_results = []

for metric in ['relevance', 'atomicity', 'clarity', 'importance', 'domain_terms_count']:
    expert_scores = results_df[results_df['source'] == 'expert'][metric].dropna()
    generated_scores = results_df[results_df['source'] == 'generated'][metric].dropna()

    if len(expert_scores) > 0 and len(generated_scores) > 0:
        stat, p_value = stats.mannwhitneyu(expert_scores, generated_scores)
        significance = 'significant' if p_value < 0.05 else 'not significant'

        # Calculate Cohen's d
        n1, n2 = len(expert_scores), len(generated_scores)
        var1, var2 = expert_scores.var(), generated_scores.var()
        pooled_std = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        if pooled_std > 0:
            cohens_d = (generated_scores.mean() - expert_scores.mean()) / (pooled_std ** 0.5)
        else:
            cohens_d = 0

        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        print(f"{metric.capitalize()}: p={p_value:.4f} ({significance}), d={cohens_d:.2f} ({effect_interpretation})")

        statistical_results.append({
            'metric': metric.capitalize(),
            'expert_mean': round(expert_scores.mean(), 2),
            'generated_mean': round(generated_scores.mean(), 2),
            'p_value': round(p_value, 4),
            'cohens_d': round(cohens_d, 2),
            'effect_size': effect_interpretation,
            'significance': significance
        })

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

summary_data = []
for source in ['expert', 'generated']:
    source_data = results_df[results_df['source'] == source]
    print(f"\n{source.upper()}:")
    for metric in ['relevance', 'atomicity', 'clarity', 'importance']:
        scores = source_data[metric].dropna()
        if len(scores) > 0:
            mean_val = scores.mean()
            std_val = scores.std()
            print(f"  {metric.capitalize()}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
            summary_data.append({
                'source': source.upper(),
                'metric': metric.capitalize(),
                'mean': round(mean_val, 2),
                'std': round(std_val, 2),
                'count': len(scores)
            })

# ============================================================================
# ALIGNMENT WITH COVERAGE AREAS
# ============================================================================
print("\n" + "=" * 70)
print("ALIGNMENT WITH ONTOLOGY COVERAGE AREAS")
print("=" * 70)


def check_coverage_area_alignment(question, coverage_areas):
    """Check which coverage areas the CQ addresses."""
    question_lower = question.lower()
    matched_areas = []

    for area in coverage_areas:
        # Extract key terms from coverage area
        area_terms = area.lower().split()
        # Check if any significant terms appear in question
        if any(term in question_lower for term in area_terms if len(term) > 4):
            matched_areas.append(area)

    return matched_areas


results_df['matched_coverage_areas'] = results_df['question'].apply(
    lambda q: check_coverage_area_alignment(q, domain_info['ONTOLOGY_COVERAGE_AREAS'])
)

for source in ['expert', 'generated']:
    source_data = results_df[results_df['source'] == source]

    print(f"\n{source.upper()}:")

    # Count how many CQs match each coverage area
    for area in domain_info['ONTOLOGY_COVERAGE_AREAS']:
        count = sum(1 for areas in source_data['matched_coverage_areas'] if area in areas)
        pct = (count / len(source_data)) * 100
        print(f"  {area[:50]}: {count} CQs ({pct:.1f}%)")

    # CQs matching no coverage areas
    no_match = sum(1 for areas in source_data['matched_coverage_areas'] if len(areas) == 0)
    print(f"  [No coverage area match]: {no_match} CQs ({(no_match / len(source_data)) * 100:.1f}%)")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

with pd.ExcelWriter('RQ2_domain_informed_blind_evaluation.xlsx') as writer:
    # Sheet 1: Detailed evaluations
    results_df.to_excel(writer, sheet_name='Detailed_Evaluations', index=False)

    # Sheet 2: Quality comparison
    comparison_df = comparison.copy()
    comparison_df.columns = ['_'.join(col).strip() for col in comparison_df.columns.values]
    comparison_df.to_excel(writer, sheet_name='Quality_Comparison')

    # Sheet 3: Statistical results
    stats_df = pd.DataFrame(statistical_results)
    stats_df.to_excel(writer, sheet_name='Statistical_Significance', index=False)

    # Sheet 4: Summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

    # Sheet 5: Domain info used
    domain_info_df = pd.DataFrame([
        {'Field': k, 'Value': str(v)[:500]}  # Truncate long values
        for k, v in domain_info.items()
    ])
    domain_info_df.to_excel(writer, sheet_name='Domain_Info_Used', index=False)

    # Sheet 6: Coverage area alignment
    coverage_alignment = []
    for area in domain_info['ONTOLOGY_COVERAGE_AREAS']:
        expert_count = sum(1 for areas in results_df[results_df['source'] == 'expert']['matched_coverage_areas']
                           if area in areas)
        generated_count = sum(1 for areas in results_df[results_df['source'] == 'generated']['matched_coverage_areas']
                              if area in areas)
        coverage_alignment.append({
            'Coverage Area': area,
            'Expert CQs': expert_count,
            'Generated CQs': generated_count
        })
    coverage_df = pd.DataFrame(coverage_alignment)
    coverage_df.to_excel(writer, sheet_name='Coverage_Area_Alignment', index=False)

print("\nResults saved to: RQ2_domain_informed_blind_evaluation.xlsx")
print("\nFile contains 6 sheets:")
print("1. Detailed_Evaluations - Individual CQ scores with domain term counts")
print("2. Quality_Comparison - Group comparison table")
print("3. Statistical_Significance - p-values, effect sizes, means")
print("4. Summary_Statistics - Mean and SD for all metrics")
print("5. Domain_Info_Used - The domain information used for evaluation")
print("6. Coverage_Area_Alignment - How CQs align with coverage areas")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)