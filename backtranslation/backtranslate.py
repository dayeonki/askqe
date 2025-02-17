import json
from deep_translator import GoogleTranslator


languages = ["es", "fr", "hi", "tl", "zh-CN"]

for language in languages:
    translator = GoogleTranslator(source='', target='en')
    perturbations = ["alteration", "expansion_impact", "expansion_noimpact", "intensifier", "omission", "spelling", "synonym", "word_order"]

    for perturbation in perturbations:
        input_file = f"../contratico/en-{language}/{perturbation}.jsonl"
        output_file = f"en-{language}/bt-{perturbation}.jsonl"

        updated_jsonl = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                pert_key = f"pert_{language}"
                if pert_key in data:
                    print("Perturbed translation: ", data[pert_key])
                    try:
                        translated_text = translator.translate(data[pert_key])
                        print("Backtranslation: ", translated_text)
                        data[f"bt_{pert_key}"] = translated_text 
                    except Exception as e:
                        print(f"Translation failed for: {data[pert_key]} with error: {e}")
                        data[f"bt_{pert_key}"] = ""
                updated_jsonl.append(data)

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in updated_jsonl:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
