import torch
import json
import os
import argparse
from prompt import qa_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir="",
        device_map="auto",
    ).to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--sentence_type", type=str)

    args = parser.parse_args()

    processed_sentences = set()

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as output_file:
            for line in output_file:
                data = json.loads(line.strip())
                processed_sentences.add(data["id"])

    # =========================================== Load Dataset ===========================================
    pipeline_types = ["vanilla", "atomic", "semantic"]

    for pipeline_type in pipeline_types:
        with open(f"../QG/llama-8b/{pipeline_type}_llama-8b.jsonl", 'r') as f_in, open(f"{args.output_path}-{pipeline_type}.jsonl", 'a') as f_out:
            for line in f_in:
                data = json.loads(line)

                sentence = data.get(args.sentence_type, None)
                questions = data.get("questions", None)

                if sentence and questions:
                    prompt_template = qa_prompt
                    prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{questions}}", questions)

                    print(prompt)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    input_ids = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(device)
                    terminators = [
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=1024,
                            eos_token_id=terminators,
                        )
                    response = outputs[0][input_ids.shape[-1]:]
                    generated_answers = tokenizer.decode(response, skip_special_tokens=True)

                    if generated_answers:
                        generated_answers = generated_answers.strip('"\'')
                    
                    print(f"> {generated_answers}")
                    print("\n======================================================\n")

                    data['answers'] = generated_answers
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                else:
                    pass


if __name__ == "__main__":
    main()