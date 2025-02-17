import json


input_jsonl_1 = "qa_llama-70b.jsonl"
input_jsonl_2 = "qa_llama-70b-bt.jsonl"
output_jsonl = "askqe_intuitive_n5.jsonl"

MAX_ALLOWED_MISMATCHES = 5  # If more than this, we reject

data_entries = []

with open(input_jsonl_1, "r", encoding="utf-8") as f1, open(input_jsonl_2, "r", encoding="utf-8") as f2:
    for line1, line2 in zip(f1, f2):
        pred_data = json.loads(line2)
        ref_data = json.loads(line1)

        predicted_answers = pred_data.get("answers", [])
        reference_answers = ref_data.get("answers", [])

        if isinstance(predicted_answers, str):
            try:
                predicted_answers = json.loads(predicted_answers)
            except json.JSONDecodeError:
                predicted_answers = []

        if isinstance(reference_answers, str):
            try:
                reference_answers = json.loads(reference_answers)
            except json.JSONDecodeError:
                reference_answers = []

        num_mismatches = sum(1 for a1, a2 in zip(reference_answers, predicted_answers) if a1 != a2)

        pred_data["num_mismatches"] = num_mismatches
        pred_data["decision"] = "reject" if num_mismatches > MAX_ALLOWED_MISMATCHES else "accept"

        data_entries.append(pred_data)


with open(output_jsonl, "w", encoding="utf-8") as f:
    for entry in data_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Processing complete. Results saved to {output_jsonl}.")
