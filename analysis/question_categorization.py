import openai
import json
from prompt import question_categorize


OPENAI_API_KEY = ""


client = openai.OpenAI(api_key=OPENAI_API_KEY)
def call_chatgpt_turbo(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


input_file = "../QG/llama-70b/atomic_llama-70b.jsonl"
output_file = f"question-type.jsonl"

with open(input_file, "r", encoding="utf-8") as file, open(output_file, "w", encoding="utf-8") as out_file:
    for line in file:
        data = json.loads(line)
        questions = data.get("questions", [])
        answers = []

        if isinstance(questions, str):
            try:
                questions = json.loads(questions)
                if not isinstance(questions, list):  
                    continue
            except (json.JSONDecodeError, ValueError):
                continue
        
        for question in questions:
            prompt = question_categorize.replace("{{question}}", question)
            print(prompt)
            response = call_chatgpt_turbo(prompt)
            print("> ", response)
            print("=" * 80)
            answers.append(response)
        
        data["question_type"] = answers
        out_file.write(json.dumps(data, ensure_ascii=False) + "\n")
