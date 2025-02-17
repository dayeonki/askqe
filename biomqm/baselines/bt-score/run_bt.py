import json
import torch
from bert_score import score


device = "cuda" if torch.cuda.is_available() else "cpu"
jsonl_file = f"../dev_with_backtranslation.jsonl"
output_file = f"dev_btscore.jsonl"


src_sentences = []
mt_sentences = []
raw_data = []
with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        if "src" in item and "bt_tgt" in item:
            src_sentences.append(item[f"src"])
            mt_sentences.append(item[f"bt_tgt"])
            raw_data.append(item)

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
