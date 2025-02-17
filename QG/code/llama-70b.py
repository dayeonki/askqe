from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import os
from prompt import prompts


model_id = "meta-llama/Llama-3.1-70B-Instruct"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir="",
        device_map="auto",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    processed_sentences = set()

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as output_file:
            for line in output_file:
                data = json.loads(line.strip())
                processed_sentences.add(data["id"])

    # =========================================== Load Dataset ===========================================
    with open("input.jsonl", 'r') as f_in, open(args.output_path, 'a') as f_out:
        for line in f_in:
            data = json.loads(line)
            sentence = data.get('en', None)
            print(sentence)
            if sentence:
                prompt_template = prompts[args.prompt]

                # Default to 'vanilla' prompt format if semantic_roles or atomic_facts are missing/empty
                if args.prompt == "semantic":
                    semantic = data.get('semantic_roles', None)
                    if semantic:
                        prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{semantic_roles}}", semantic)
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                elif args.prompt == "atomic":
                    atomics = data.get('atomic_facts', None)
                    if atomics:
                        prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{atomic_facts}}", str(atomics))
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                else:  # Default case (e.g., vanilla)
                    prompt = prompt_template.replace("{{sentence}}", sentence)
                
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
                generated_questions = tokenizer.decode(response, skip_special_tokens=True)

                if generated_questions:
                    generated_questions = generated_questions.strip('"\'')
                
                print(f"> {generated_questions}")
                print("\n======================================================\n")

                data['questions'] = generated_questions
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                pass


if __name__ == "__main__":
    main()
