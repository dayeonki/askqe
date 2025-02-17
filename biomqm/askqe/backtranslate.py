import json
from deep_translator import GoogleTranslator


input_file = f"askqe_atomic_facts.jsonl"
output_file = f"askqe_atomic_facts_bt.jsonl"


updated_jsonl = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        tgt_lang = data["lang_tgt"]
        translator = GoogleTranslator(source=tgt_lang, target='en')
        if "tgt" in data:
            print("Translation: ", data["tgt"])
            try:
                translated_text = translator.translate(data["tgt"])
                print("Backtranslation: ", translated_text)
                data["bt_tgt"] = translated_text
            except Exception as e:
                print(f"Translation failed for: {data['tgt']} with error: {e}")
                data["bt_tgt"] = ""
        updated_jsonl.append(data)

with open(output_file, 'w', encoding='utf-8') as f:
    for entry in updated_jsonl:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
