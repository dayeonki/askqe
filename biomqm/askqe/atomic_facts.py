import openai
import json
from prompt import atomic_fact_prompt


OPENAI_API_KEY = ""

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def call_chatgpt_turbo(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


input_file = f"../dev_with_backtranslation.jsonl"
output_file = f"askwe_atomic_facts.jsonl"

with open(input_file, "r", encoding="utf-8") as file, open(output_file, "w", encoding="utf-8") as out_file:
    for line in file:
        data = json.loads(line)
        if "src" in data:
            sentence = data["src"]
            prompt = atomic_fact_prompt.replace("{{sentence}}", sentence)
            print(prompt)
            response = call_chatgpt_turbo(prompt)
            print("> ", response)
            print("=" * 80)
            
            data[f"atomic_facts"] = response
            out_file.write(json.dumps(data, ensure_ascii=False) + "\n")
