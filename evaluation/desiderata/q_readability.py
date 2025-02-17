import json
import textstat
import numpy as np


pipelines = ["atomic", "semantic", "vanilla"]
models = ["gemma-9b", "gemma-27b", "llama-8b", "llama-70b", "yi-9b"]

for pipeline in pipelines:
    for model_name in models:
        jsonl_file = f"../QG/{model_name}/{pipeline}_{model_name}.jsonl"
        print("File: ", jsonl_file)

        total_entries = 0
        readability_scores = []

        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
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
     
                if len(questions) == 0:
                    continue

                total_entries += 1
                instance_scores = []

                for question in questions:
                    score = textstat.flesch_reading_ease(question)  # Flesch Reading Ease Score
                    instance_scores.append(score)

                avg_instance_score = np.mean(instance_scores)
                readability_scores.append(avg_instance_score)


        def classify_readability(score):
            if score >= 90:
                return "Very Easy (5th grade)"
            elif score >= 80:
                return "Easy (6th grade)"
            elif score >= 70:
                return "Fairly Easy (7th grade)"
            elif score >= 60:
                return "Standard (8th-9th grade)"
            elif score >= 50:
                return "Fairly Difficult (10th-12th grade)"
            elif score >= 30:
                return "Difficult (College)"
            else:
                return "Very Difficult (Graduate level)"

        if readability_scores:
            avg_readability = np.mean(readability_scores)
            print(f"Average Readability Score (Flesch-Kincaid): {avg_readability:.2f}")
        else:
            print("No valid questions found in dataset.")
        print("Division: ", classify_readability(avg_readability))
