import json


pipelines = ["atomic", "semantic", "vanilla"]
models = ["gemma-9b", "gemma-27b", "llama-8b", "llama-70b", "yi-9b"]

for pipeline in pipelines:
    for model_name in models:
        jsonl_file = f"../QG/{model_name}/{pipeline}_{model_name}.jsonl"
        print("File: ", jsonl_file)

        total_entries = 0
        total_questions = 0
        duplicate_questions_count = 0

        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                total_entries += 1
                questions = data.get("questions", [])

                if isinstance(questions, str):
                    try:
                        questions = json.loads(questions)
                        if not isinstance(questions, list):  
                            continue
                    except (json.JSONDecodeError, ValueError):
                        continue

                unique_questions = set(questions)  
                if len(unique_questions) < len(questions):
                    duplicate_questions_count += 1
                total_questions += len(questions)

        avg_questions = total_questions / total_entries if total_entries > 0 else 0
        print(f"Average Number of Questions: {avg_questions:.2f}")
