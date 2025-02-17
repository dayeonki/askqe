import json
from deep_translator import GoogleTranslator


languages = ["es", "fr", "hi", "tl", "zh-CN"]


for language in languages:
    translator = GoogleTranslator(source='en', target=language)

    input_file = f"en-{language}.jsonl"
    output_file = f"en-{language}-gt.jsonl"

    updated_jsonl = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if "en" in data:  # Check if pert_es exists
                print("Source: ", data["en"])
                try:
                    translated_text = translator.translate(data["en"])
                    print("Translation: ", translated_text)
                    data["gt_en"] = translated_text  # Add translated English text
                except Exception as e:
                    print(f"Translation failed for: {data['pert_es']} with error: {e}")
                    data["gt_en"] = ""
            updated_jsonl.append(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in updated_jsonl:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
