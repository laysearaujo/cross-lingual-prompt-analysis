# Cross-Lingual Prompt Analysis: A Study on Prompt Quality and Language Impact

The research investigates two key variables that influence the performance of Large Language Models (LLMs): prompt quality (from vague to specific) and language (Brazilian Portuguese vs. English).

## 📌 Research Problem & Hypotheses
The effectiveness of LLMs is highly sensitive to the quality of the input prompt. This challenge is compounded in non-English contexts, where model performance can differ. This study aims to systematically quantify the individual and combined impact of prompt specificity and language choice.

Our research is guided by the following hypotheses:

* H₁: Prompts with a higher degree of specificity and structure lead to a measurable improvement in LLM performance in both Portuguese and English.

* H₂: The performance gain obtained from prompt optimization differs significantly between Portuguese and English.

## 🔬 Proposed Methodology
This project will follow a rigorous experimental design to test the hypotheses. The main steps include:

1. **Task & Prompt Creation:** A dataset of questions across multiple domains (e.g., General Knowledge, Technical, Creative) will be created. For each question, four levels of prompt specificity will be constructed (Minimal, Contextual, Detailed, Structured).

2. **Multilingual Setup:** All prompts will be created in English and then professionally translated and reviewed for Brazilian Portuguese to create a parallel corpus.

3. **Experiment Execution:** Prompts will be run against a series of LLMs (e.g., Gemini, GPT-4o, Llama-3), with multiple samples generated for each condition.

4. **Evaluation:** A dual-evaluation strategy will be employed, using an automated LLM-as-a-Judge for large-scale scoring and Human Validation on a subset of the data.

## 📈 Status
This project is currently in the **planning and experimental** design phase. The repository will be updated as the scripts and data are developed.
