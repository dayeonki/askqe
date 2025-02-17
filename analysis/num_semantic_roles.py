import json


def count_verbs(semantic_roles_str):
    try:
        semantic_roles = eval(semantic_roles_str)        
        verb_count = 0
        for key in semantic_roles.keys():
            if key.startswith("Verb"):
                verb_count += 1
        return verb_count
    except:
        return 0

jsonl_file = "../QG/code/qg_variants.jsonl"
verb_counts = []

with open(jsonl_file, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line.strip())
        semantic_roles_str = data.get("semantic_roles", "{}")
        verb_counts.append(count_verbs(semantic_roles_str))

average_verbs = sum(verb_counts) / len(verb_counts) if verb_counts else 0
print(f"Average number of verbs per sentence: {average_verbs:.2f}")
