import json


gold_file = "human_ratings.jsonl"
pred_file = "btscore.jsonl"

def load_jsonl_list(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

gold_data = load_jsonl_list(gold_file)
pred_data = load_jsonl_list(pred_file)

if len(gold_data) != len(pred_data):
    print(f"Warning: Different number of entries in files! Gold: {len(gold_data)}, Pred: {len(pred_data)}")

correct = 0
total = min(len(gold_data), len(pred_data))

for i in range(total):
    if gold_data[i].get("decision") == pred_data[i].get("decision"):
        correct += 1

accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Decision Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
