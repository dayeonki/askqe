import json
import os
import torch
from bert_score import score


languages = ["es", "fr", "hi", "tl", "zh"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact", 
                 "intensifier", "expansion_impact", "omission", "alteration"]

base_path = "../backtranslation"
device = "cuda" if torch.cuda.is_available() else "cpu"


for language in languages:
    for perturbation in perturbations:
        jsonl_file = f"{base_path}/en-{language}/bt-{perturbation}.jsonl"
        output_file = f"en-{language}/bt-{perturbation}_bertscore.jsonl"

        print(f"\nProcessing File: {jsonl_file}")

        if not os.path.exists(jsonl_file):
            print(f"File not found: {jsonl_file}")
            continue

        src_sentences = []
        mt_sentences = []
        raw_data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if "en" in item and f"pert_{language}" in item: 
                    src_sentences.append(item[f"en"])
                    mt_sentences.append(item[f"bt_pert_{language}"])
                    raw_data.append(item)

        if not src_sentences:
            print(f"No valid data in {jsonl_file}")
            continue

        P, R, F1 = score(mt_sentences, src_sentences, lang="en", 
                         model_type="microsoft/deberta-xlarge-mnli", 
                         device=device,
                         batch_size=2)

        average_score = F1.mean().item() if len(F1) > 0 else 0

        with open(output_file, "w", encoding="utf-8") as out_f:
            for item, f1_score in zip(raw_data, F1):
                item["bertscore_f1"] = f1_score.item()
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Average BERTScore (F1): {average_score:.4f}")
