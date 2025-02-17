import json


jsonl_file = "../QG/code/qg_variants.jsonl"
atomic_fact_lengths = []

with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        if "atomic_facts" in data and data["atomic_facts"].strip():  
            atomic_facts = eval(data["atomic_facts"])
            atomic_fact_lengths.append(len(atomic_facts))
        else:
            continue
    print(atomic_fact_lengths)

average_length = sum(atomic_fact_lengths) / len(atomic_fact_lengths) if atomic_fact_lengths else 0
print(f"Average length of atomic facts: {average_length:.2f}")
