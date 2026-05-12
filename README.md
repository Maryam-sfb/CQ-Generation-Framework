# CQ-Generation-Framework
An LLM-powered framework for Competency Question generation, refinement, filtering and clustering.

Competency Questions play a central role in ontology engineering by defining the functional requirements that an ontology should
satisfy. However, eliciting high-quality competency questions typically requires extensive collaboration between ontology engineers and domain experts, making the process time-consuming and difficult to scale. 

This project presents an LLM-powered framework for competency question generation. The framework integrates structured domain expert input with the retrieval of scholarly literature from Springer.com and leverages large language models to generate, refine, filter, and cluster the questions. The proposed framework reduces the dependency on continuous expert involvement while maintaining high question quality.

The framework has been developed using Python 3.13. It uses Azure OpenAI services, and the gpt-4o-mini model was selected for its optimal balance of performance and cost effectiveness in processing large volumes of text through multiple pipeline stages. Also,
GPT-5 was utilized for Domain information extraction for higher precision. All API calls are configured with a low temperature setting to ensure deterministic and reproducible outputs across the ontology engineering pipeline: temperature=0.3 for the generative CQ creation step, and temperature=0.1 for more deterministic tasks like domain information extraction, CQ refinement, filtering,
and clustering. 

Several Python libraries have been used to develop the pipeline, including spaCy for text processing, langdetect for language identification of scholarly articles, sentence-transformers and scikit-learn for calculating the semantic similarity of CQs, SciPy for data analysis, and other common libraries such as pandas and NumPy.

## Usage
1. Install the required dependencies: ```pip install -r requirements.txt```
2. Copy `.env-sample` to a new file named `.env` in the `CQ_Generation_Framework` folder and fill in the required keys.
3. Open `json_input/scope-expert.json` and replace the existing answers with your own domain scoping responses to the four questions.
4. Run the pipeline: `python run_pipeline.py`

> **Note:** The pipeline requires valid API keys for Azure OpenAI (GPT-4o-mini) and SerpAPI.

## Framework Execution Guide

The framework is implemented as a series of modular Python scripts 
that form an automated sequential pipeline. All outputs are saved 
in the `output/` folder and passed automatically as inputs to 
subsequent stages based on the most recent timestamp.

### Pipeline Scripts

1. **`extract_domain_info.py`**
   - Purpose: Extracts structured domain information from expert input
   - Input: `json_input/scope-expert.json`
   - Output: `json_input/domain-info.json`, `output/domain_info.xlsx`

2. **`extract_articles.py`**
   - Purpose: Retrieves and filters scholarly articles, extracts snippets
   - Input: `json_input/domain-info.json`
   - Output: `output/llm_input_springer_[datetime].xlsx`,
             `output/articles_summary_[datetime].txt`

3. **`generate_cqs.py`**
   - Purpose: Generates Competency Questions from article snippets
   - Input: `output/llm_input_springer_[datetime].xlsx` (latest)
   - Output: Appends `CQs` sheet to same Excel file

4. **`refinement.py`**
   - Purpose: Abstracts named entities in CQs to enhance reusability
   - Input: `output/llm_input_springer_[datetime].xlsx` (latest)
   - Output: `output/refined_cqs_springer_[datetime].xlsx`

5. **`joint_filtering.py`**
   - Purpose: Removes redundancy, scores relevance, filters by 
     linguistic complexity
   - Input: `output/refined_cqs_springer_[datetime].xlsx` (latest)
   - Output: `output/joint_filtered_cqs_[datetime].xlsx`
  
> **Note:** The current implementation is using gpt-4o-mini across all steps, including extract_domain_info.py. However, in our research, we used GPT-5 for domain information extraction to achieve higher precision. If you plan to use a model other than GPT-4o-mini for this step (which is recommended), make sure to update the relevant files accordingly.
