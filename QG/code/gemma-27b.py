from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import prompts
import torch
import json
import argparse


model_id = "google/gemma-2-27b-it"

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Using bfloat16 for lower memory usage
        cache_dir="",
        device_map="auto",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    # =========================================== Load Dataset ===========================================    
    with open("qg_variants.jsonl", 'r') as f_in, open(args.output_path, 'w') as f_out:
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
                
                input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    outputs = model.generate(
                        **input_ids,
                        max_new_tokens=1024,
                        num_beams=1,
                    )
                
                generated_questions = tokenizer.decode(outputs[0], skip_special_tokens=True)

                answer_start = "Questions: "
                if answer_start in generated_questions:
                    generation = generated_questions.split(answer_start)[-1].strip()
                    generation = generation.split("<")[0].strip()
                else:
                    generation = generated_questions

                print(f"{prompt}")
                print(f"> {generation}")
                print("\n======================================================\n")

                data['questions'] = str(generation)
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
