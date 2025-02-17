import json
import torch
import itertools
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
import bert_score


pipelines = ["atomic", "semantic", "vanilla"]
models = ["gemma-9b", "gemma-27b", "llama-8b", "llama-70b", "yi-9b"]


sbert_model = SentenceTransformer("all-mpnet-base-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model.to(device)


for pipeline in pipelines:
    for model in models:
        jsonl_file = f"../QG/{model}/{pipeline}_{model}.jsonl"
        output_file = f"diversity/{pipeline}_{model}.jsonl"

        if not os.path.exists(jsonl_file):
            print(f"Skipping missing file: {jsonl_file}")
            continue

        print(f"\nProcessing File: {jsonl_file}")

        total_entries = 0
        diversity_scores = []
        processed_data = []

        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line)
                    total_entries += 1
                    questions = data.get("questions", [])

                    if isinstance(questions, str):
                        try:
                            questions = json.loads(questions)
                            if not isinstance(questions, list):
                                print(f"Skipping due to invalid question format.")
                                continue
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"Skipping due to invalid question format: {e}")
                            continue

                    if len(questions) < 2:
                        continue

                    question_pairs = list(itertools.combinations(questions, 2))
                    embeddings = sbert_model.encode(questions, convert_to_tensor=True)
                    cosine_similarities = [
                        util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                        for i, j in itertools.combinations(range(len(questions)), 2)
                    ]

                    P, R, F1 = bert_score.score(questions, questions, lang="en", rescale_with_baseline=True)
                    bert_similarities = [
                        F1[i].item() for i, j in itertools.combinations(range(len(questions)), 2)
                    ]

                    avg_cosine_sim = np.mean(cosine_similarities) if cosine_similarities else 0
                    avg_bert_sim = np.mean(bert_similarities) if bert_similarities else 0
                    diversity_scores.append((avg_cosine_sim, avg_bert_sim))

                    data["cosine_similarity"] = avg_cosine_sim
                    data["bert_similarity"] = avg_bert_sim
                    processed_data.append(data)

                except json.JSONDecodeError as e:
                    print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                    continue

        if diversity_scores:
            avg_sbert_diversity = np.mean([s[0] for s in diversity_scores])
            avg_bert_diversity = np.mean([s[1] for s in diversity_scores])
        else:
            avg_sbert_diversity = 0
            avg_bert_diversity = 0

        print(f"Overall Average Cosine Similarity (SBERT): {avg_sbert_diversity:.4f}")
        print(f"Overall Average BERTScore Similarity: {avg_bert_diversity:.4f}")

        avg_score_entry = {
            "overall_avg_cosine_similarity": avg_sbert_diversity,
            "overall_avg_bert_similarity": avg_bert_diversity
        }
        processed_data.append(avg_score_entry)

        with open(output_file, "w", encoding="utf-8") as out_f:
            for entry in processed_data:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Saved results to: {output_file}")
