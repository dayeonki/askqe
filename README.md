<div align="center">

 # AskQE: Question Answering as Automatic Evaluation <br> for Machine Translation

<img width="1036" height="284" alt="Screenshot 2026-04-03 at 1 30 44 PM" src="https://github.com/user-attachments/assets/4829a417-17b2-41b9-b822-73e13a3bf704" />


<a href=https://dayeonki.github.io/>Dayeon Ki</a><sup>1</sup>, <a href=https://www.cs.jhu.edu/~kevinduh/>Kevin Duh</a><sup>2</sup>, <a href=https://www.cs.umd.edu/~marine/>Marine Carpuat<a><sup>1</sup> <br>
<sup>1</sup>University of Maryland, <sup>2</sup>Johns Hopkins University
<br>

This repository contains the code and dataset for our ACL Findings 2025 paper <br> **AskQE: Question Answering as Automatic Evaluation for Machine Translation**.

<p>
  <a href="https://aclanthology.org/2025.findings-acl.899/" target="_blank" style="text-decoration:none">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat&logo=arxiv" alt="arXiv">
  </a>
</p>

</div>

---

## 👾 TL;DR
How can a monolingual English speaker determine whether an automatic translation in French is good enough to be shared? We introduce **AskQE**, a question generation and answering framework designed to detect critical MT errors and provide actionable feedback, helping users decide whether to accept or reject MT outputs even without the knowledge of the target language.


## 📰 News
- **`2025-05-15`** Our paper is accepted to **ACL Findings 2025**! See you in Vienna!
- **`2025-05-07`** Our paper is featured in <a href=https://slator.com/how-question-answering-can-transform-ai-translation-evaluation/>Slator</a>!


## ✏️ Content
- [🗺️ Overview](#overview)
- [🚀 Quick Start](#quick_start)
  - [Data Preparation](#data-preparation)
  - [Question Generation](#question-generation)
  - [Question Answering](#question-answering)
  - [Backtranslation](#backtranslation)
  - [Evaluation](#evaluation)
  - [Analysis](#analysis)
- [🤲 Citation](#citation)
- [📧 Contact](#contact)

---

<a id="overview"></a>
## 🗺️ Overview

How can we identify critical translation errors and provide actionable feedback tohelp monolingual source speakers decide whether to accept or reject MT in high-stake contexts? We propose **AskQE**, a question generation and answering framework based on the idea that a translation is unreliable if key questions about the source text yield different answers when derived from the source or the backtranslated MT. We first generate a list of questions conditioned on the source (**QG**), generate answers for each question based on the source or the backtranslated MT output (**QA**), and compute the answer overlap.


### Results

<div align="center">
<img width="905" height="312" alt="Screenshot 2026-04-03 at 1 46 05 PM" src="https://github.com/user-attachments/assets/52a93c10-f8c7-47bc-9cbd-30257a4b968d" />
</div>


<a id="quick_start"></a>
## 🚀 Quick Start

### Data Preparation
For our main experiments, we use [TICO-19](https://tico-19.github.io/) dataset as our testbed. To align with practical settings, we select language pairs that are high in demand in the United States healthcare system: English-Spanish (en-es), English-French (en-fr), English-Hindi (en-hi), English-Tagalog (en-tl), and English-Chinese (en-zh).

- Data for each language pair is in: `data/processed/{language_pair}.jsonl`
- Translated source sentences are in: `data/google_translate/{language_pair}-gt.jsonl`
- xCOMET scores between the source and target are in: `data/xcomet/{language_pair}-xcomet.jsonl`


#### (1) ContraTICO

Since the original TICO-19 dataset only provides English source and reference translations in the target language, we construct a dataset with synthetic errors, **ContraTICO**, to assess the impact of AskQE design in a controlled setting. We prompt GPT4o to perturb the reference translation with specific error, which results in a perturbed translation. We define eight linguistic perturbations, categorized by their level of severity into either Minor or Critical, based on the potential implications of the translation error in practice.

- **Minor:** These errors do not lead to loss of meaning but introduce small inaccuracies or stylistic inconsistencies that might marginally affect clarity.
  - Spelling: Misspell one to two words.
  - Word Order: Reorder words in the sentence.
  - Synonym: Replace one word to its synonym.
  - Intensifier: Modify the intensity of an adjective or an adverb (e.g., small to very small).
  - Expansion (No Impact): Expand a word or phrase by adding contextually implied details without introducing new meaning.
- **Critical:** These errors significantly changes the original meaning and usually appear in a highly visible or important part of the content.
  - Expansion (Impact): Expand a word or phrase by introducing new meaning.
  - Omission: Omit a word or phrase.
  - Alteration: Alter a word or phrase by changing its original meaning.

We show example for each perturbation as follows:
<p align="center">
<img width="900" alt="Screenshot 2025-03-21 at 3 48 30 PM" src="https://github.com/user-attachments/assets/648ed2e2-4316-43ea-80b8-13d2653751da" />
</p>

- To perturb, run `python contratico/gpt_perturb.py` by setting the `OPENAI_API_KEY` to your personal OpenAI API key.
- Results per language pair and perturbation are in `contratico/{language_pair}/{perturbation}.jsonl`.

#### (2) BioMQM

BIOMQM is a biomedical domain MT dataset with error annotations by professional translators based on the multidimensional quality metrics (MQM). We extend analysis with the ContraTICO dataset with the BIOMQM dataset with more naturally occurring translation errors and additional language pairs.

- **Data:** `biomqm/dev_with_backtranslation.jsonl`
- **AskQE code:** `biomqm/askqe`
- **Baseline code** (QE metrics): `biomqm/baselines`
- **Human simulation:** Both code and results can be found in `biomqm/human_simulation`


### Question Generation
Given a source sentence, we generate a set of questions that can be answered based on the sentence. Before generating questions, we extract information from the source on what to ask questions about and incorporate it as additional context in the prompt. Specifically, to ensure comprehensive coverage of the information from the source, we implement a two-step natural language inference (NLI) pipeline (as shown in the main figure):
1. **Fact extraction:** we prompt GPT-4o to extract atomic facts that can be inferred from the source sentence
2. **Entailment classification:** we use an off-theshelf NLI classifier to assess the binary entailment relationship (entailed or contradictory) between each extracted fact (as the hypothesis) and the source sentence (as the premise)

We discard facts labeled as contradictory, potentially indicating that they cannot be reliably inferred from the source. We then prompt an LLM to generate questions given the source sentence and the filtered set of entailed atomic facts.
We also test different variants: vanilla and semantic. Results for each variant can be found in `QG/{model}/{variant}_{model}.jsonl`.

To run question generation for each model,

```bash
python -u QG/code/{$LLM}.py \
  --output_path $PATH_TO_OUTPUT_FILE \
  --prompt $QG_VARIANT \
```

Arguments for the QG code are as follows:
  - `$LLM`: Model name to use for QG.
  - `--output_path`: Save path of output file (after question generation).
  - `--prompt`: QG variant (whether vanilla / semantic / atomic).


### Question Answering
We generate answers for each question using two different contexts: source sentence and the backtranslated MT output. Results for each variant can be found in `QA/{model}/{language}-{variant}-{perturbation}.jsonl`.

To run question answering for each model,

```bash
python -u QA/code/{$LLM}.py \
  --output_path $PATH_TO_OUTPUT_FILE \
  --sentence_type $QA_VARIANT \
```

Arguments for the QA code are as follows:
  - `$LLM`: Model name to use for QA.
  - `--output_path`: Save path of output file (after question generation).
  - `--sentence_type`: QA variant (whether source or backtranslated MT output).


### Backtranslation
For backtranslation, we use the Google Translate API. We can run the backtranslation code as `python backtranslation/backtranslate.py` for all language pairs. Results for backtranslated MT output are in `backtranslation/{language_pair}/bt-{perturbation}.jsonl`.

### Evaluation
#### (1) Baselines (QE Metrics)
We further compare our method against three established QE metrics:
- **xCOMET-QE** (evaluate the source and MT output): `python evaluation/xcomet-qe/xcomet.py`
- **MetricX** (evaluate the source and MT output)
- **BT-score** (evaluate the similarity between the source and backtranslated MT output using BERTScore as MT metric): `python evaluation/bt-score/run_bt.py`

#### (2) Main evaluation
For running overall evaluation on AskQE outputs (the results are in each directory):
- **SBERT:** `python evaluation/sbert/sbert.py`
- **String comparison metrics** (F1, EM, BLEU, chrF): `python evaluation/string-comparison/string_comparison.py`

#### (3) Desiderata evaluation
We define five quality desiderata that measures the correctness, diversity, readability, and answerability of the questions.
- **Empty:** `python evaluation/desiderata/i_avg_questions.py`
- **Duplicate:** `python evaluation/desiderata/i_duplicate.py`
- **Diversity:** `python evaluation/desiderata/i_diversity.py`
- **Answerability:** `python evaluation/desiderata/q_answerability.py`
- **Readability:** `python evaluation/desiderata/q_readability.py`


### Analysis
We provide code used for analysis in `analysis/`. To run question categorization, run `python analysis/question_categorization.py` by setting the `OPENAI_API_KEY` to your personal OpenAI API key.


---

<a id="citation"></a>
## 🤲 Citation
If you find our work useful in your research, please consider citing our work:
```
@inproceedings{ki-etal-2025-askqe,
    title = "{A}sk{QE}: Question Answering as Automatic Evaluation for Machine Translation",
    author = "Ki, Dayeon  and
      Duh, Kevin  and
      Carpuat, Marine",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.899/",
    doi = "10.18653/v1/2025.findings-acl.899",
    pages = "17478--17515",
    ISBN = "979-8-89176-256-5",
}
```

<a id="contact"></a>
## 📧 Contact
For questions, issues, or collaborations, please reach out to [dayeonki@umd.edu](mailto:dayeonki@umd.edu).
