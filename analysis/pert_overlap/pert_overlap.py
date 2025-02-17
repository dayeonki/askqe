import os
import json
import pandas as pd


languages = ["es", "fr", "hi", "tl", "zh"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact", 
                 "intensifier", "expansion_impact", "omission", "alteration"]

for language in languages:
    directory = f"../contratico/en-{language}"

    jsonl_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    jsonl_files_sorted = sorted(jsonl_files, key=lambda x: perturbations.index(x.replace(".jsonl", "")) if x.replace(".jsonl", "") in perturbations else len(perturbations))
    overlap_matrix = pd.DataFrame(index=jsonl_files_sorted, columns=jsonl_files_sorted).fillna(0)

    data_dict = {}
    for file in jsonl_files_sorted:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data_dict[file] = {json.loads(line)['id']: json.loads(line).get(f'pert_{language}', '') for line in f}

    for file1 in jsonl_files_sorted:
        for file2 in jsonl_files_sorted:
            if file1 <= file2:
                common_ids = set(data_dict[file1].keys()) & set(data_dict[file2].keys())
                overlap_score = sum(1 for id_ in common_ids if data_dict[file1][id_] == data_dict[file2][id_])
                overlap_matrix.at[file1, file2] = overlap_score
                overlap_matrix.at[file2, file1] = overlap_score

    overlap_matrix.to_csv(f"en-{language}_overlap.csv")

    print(f"Overlap matrix for en-{language}:")
    print(overlap_matrix)
