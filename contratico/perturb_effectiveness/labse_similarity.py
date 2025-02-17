from sentence_transformers import SentenceTransformer, util
import json


model = SentenceTransformer('sentence-transformers/LaBSE')


languages = ["es", "fr", "hi", "tl", "zh"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact", 
                 "intensifier", "expansion_impact", "omission", "alteration"]

for language in languages:
    for perturbation in perturbations:

        jsonl_file = f'en-{language}/{perturbation}.jsonl'
        similarity_scores = []
        print("File: ", jsonl_file)

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                en_embedding = model.encode(data['en'], convert_to_tensor=True)
                pert_es_embedding = model.encode(data[f'pert_{language}'], convert_to_tensor=True)
                similarity_score = util.pytorch_cos_sim(en_embedding, pert_es_embedding).item()
                similarity_scores.append(similarity_score)

        print(f"Embedding Similarity: {sum(similarity_scores) / len(similarity_scores)}")
