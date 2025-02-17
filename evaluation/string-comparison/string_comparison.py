import json
import nltk
from utils import compare_answers

nltk.download("punkt")


languages = ["es", "fr", "hi", "tl", "zh"]
pipelines = ["vanilla", "semantic", "atomic"]
perturbations = ["alteration", "expansion_impact", "expansion_noimpact", "intensifier", "omission", "spelling", "synonym", "word_order"]


for language in languages:
    for pipeline in pipelines:
        for perturbation in perturbations:
            predicted_file = f"../../QA/llama-70b/{language}-{pipeline}-{perturbation}.jsonl"
            reference_file = f"../../QA/llama-70b/en-{pipeline}.jsonl"

            results_list = []
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

                            row_scores = []
                            for pred, ref in zip(predicted_answers, reference_answers):
                                f1, EM, chrf, bleu = compare_answers(pred, ref)
                                row_scores.append({
                                    "f1": f1,
                                    "em": EM,
                                    "chrf": chrf,
                                    "bleu": bleu
                                })

                            # Save per-row result
                            row_data = {
                                "id": pred_data.get("id", "unknown"),
                                "en": pred_data.get("en", "unknown"),
                                "scores": row_scores
                            }
                            results_list.append(row_data)

                        except json.JSONDecodeError as e:
                            print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                            continue

            except FileNotFoundError as e:
                print(f"File not found: {e}")

            jsonl_output_file = f"en-{language}/{perturbation}.jsonl"
            with open(jsonl_output_file, "w", encoding="utf-8") as jsonl_file:
                for row in results_list:
                    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
