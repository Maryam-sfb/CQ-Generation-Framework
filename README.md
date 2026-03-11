# CQ-Generation-Framework
An LLM-powered framework for Competency Question generation, refinement, filtering and clustering.

Competency Questions play a central role in ontology engineering by defining the functional requirements that an ontology should
satisfy. However, eliciting high-quality competency questions typically requires extensive collaboration between ontology engineers and domain experts, making the process time-consuming and difficult to scale. 

This project presents an LLM-powered framework for competency question generation. The framework integrates structured domain expert input with the retrieval of scholarly literature from Springer.com and leverages large language models to generate, refine, filter, and cluster the questions. The proposed framework reduces the dependency on continuous expert involvement while maintaining high question quality.

The framework has been developed using Python 3.13. It uses Azure OpenAI services, and the gpt-4o-mini model was selected for its optimal balance of performance and cost effectiveness in processing large volumes of text through multiple pipeline stages. Also,
GPT-5 was utilized for Domain information extraction for higher precision. All API calls are configured with a low temperature setting to ensure deterministic and reproducible outputs across the ontology engineering pipeline: temperature=0.3 for the generative CQ creation step, and temperature=0.1 for more deterministic tasks like domain information extraction, CQ refinement, filtering,
and clustering. 

Several Python libraries have been used to develop the pipeline, including spaCy for text processing, langdetect for language identification of scholarly articles, sentence-transformers and scikit-learn for calculating the semantic similarity of CQs, SciPy for data analysis, and other common libraries such as pandas and NumPy.
