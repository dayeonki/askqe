import openai
import json
from prompt import prompts


OPENAI_API_KEY = ""

LANGUAGE_MAP = {
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "tl": "Tagalog",
    "zh": "Chinese"
}

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def call_chatgpt_turbo(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


languages = ["es", "fr", "hi", "tl", "zh"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact", 
                 "intensifier", "expansion_impact", "omission", "alteration"]


for language in languages:
    for perturbation in perturbations:
        print("Perturbation: ", perturbation)
        
        input_file = f"../data/processed/en-{language}.jsonl"
        output_file = f"en-{language}/{perturbation}.jsonl"

        with open(input_file, "r", encoding="utf-8") as file, open(output_file, "w", encoding="utf-8") as out_file:
            for line in file:
                data = json.loads(line)
                if f"{language}" in data:
                    target_lang = LANGUAGE_MAP.get(language, language)
                    sentence = data[f"{language}"]
                    prompt = prompts[f"{perturbation}_{language}"].replace("{{original}}", sentence).replace("{{target_lang}}", target_lang)
                    print(prompt)
                    response = call_chatgpt_turbo(prompt)
                    print("> ", response)
                    print("=" * 80)
                    
                    data["perturbation"] = perturbation
                    data[f"pert_{language}"] = response
                    out_file.write(json.dumps(data, ensure_ascii=False) + "\n")
