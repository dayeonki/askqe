import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
from prompt import prompts


model_id = "01-ai/Yi-1.5-9B-Chat"


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, cache_dir="")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        cache_dir="",
        torch_dtype='auto'
    ).to(device).eval()

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
                    {"role": "user", "content": prompt}
                ]

                input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
                output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id, max_length=1024)
                generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                answer_start = "Questions:"
                if answer_start in generated_text:
                    try:
                        generation = generated_text.split(answer_start)[-1].strip().split()[0]
                    except:
                        generation = generated_text
                else:
                    generation = generated_text
                
                if "\n" in generation:
                    final_generation = generation.split("\n")[0]
                else:
                    final_generation = generation

                print("> ", final_generation)
                print("\n======================================================\n")

                data['questions'] = final_generation
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
