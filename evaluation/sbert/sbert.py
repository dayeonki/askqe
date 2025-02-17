import json
import nltk
import argparse
import csv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


nltk.download("punkt")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()


languages = ["es", "fr", "hi", "tl", "zh"]
pipelines = ["vanilla", "semantic", "atomic"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact",
                 "intensifier", "expansion_impact", "omission", "alteration"]


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


with open(args.output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["language", "perturbation", "pipeline", "cosine_similarity", "num_comparison"])

    for language in languages:
        for pipeline in pipelines:
            for perturbation in perturbations:
                print("Language: ", language)
                print("Pipeline: ", pipeline)
                print("Perturbation: ", perturbation)

                predicted_file = f"../../QA/{args.model}/{language}-{pipeline}-{perturbation}.jsonl"
                reference_file = f"../../QA/{args.model}/en-{pipeline}.jsonl"

                total_cosine_similarity = 0
                num_comparisons = 0

                try:
                    with open(predicted_file, "r", encoding="utf-8") as pred_file, open(reference_file, "r", encoding="utf-8") as ref_file:
                        for pred_line, ref_line in zip(pred_file, ref_file):
                            try:
                                pred_data = json.loads(pred_line)
                                ref_data = json.loads(ref_line)

                                predicted_answers = pred_data.get("answers", [])
                                reference_answers = ref_data.get("answers", [])

                                if isinstance(predicted_answers, str):
                                    try:
                                        predicted_answers = json.loads(predicted_answers)
                                    except json.JSONDecodeError:
                                        continue

                                if isinstance(reference_answers, str):
                                    try:
                                        reference_answers = json.loads(reference_answers)
                                    except json.JSONDecodeError:
                                        continue

                                if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list):
                                    continue
                                if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers):
                                    continue
                                for pred, ref in zip(predicted_answers, reference_answers):
                                    if not isinstance(pred, str) or not isinstance(ref, str):
                                        continue
                                    if pred.strip() == "" or ref.strip() == "":
                                        continue

                                    encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
                                    encoded_ref = tokenizer(ref, padding=True, truncation=True, return_tensors='pt')

                                    with torch.no_grad():
                                        pred_output = model(**encoded_pred)
                                        ref_output = model(**encoded_ref)

                                    pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
                                    pred_embeds = F.normalize(pred_embed, p=2, dim=1)

                                    ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
                                    ref_embeds = F.normalize(ref_embed, p=2, dim=1)

                                    cos_sim = F.cosine_similarity(pred_embeds, ref_embeds, dim=1).mean().item()
                                    total_cosine_similarity += cos_sim
                                    num_comparisons += 1

                            except json.JSONDecodeError as e:
                                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                                continue

                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    continue

                if num_comparisons > 0:
                    avg_cosine_similarity = total_cosine_similarity / num_comparisons

                    print("-" * 80)
                    print("Average Scores:")
                    print(f"Num comparisons: {num_comparisons}")
                    print(f"Cosine Similarity: {avg_cosine_similarity:.3f}")
                    print("=" * 80)

                    with open(args.output_file, mode="a", newline="", encoding="utf-8") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([language, perturbation, pipeline, avg_cosine_similarity, num_comparisons])

                else:
                    print("No valid comparisons found in the JSONL files.")
