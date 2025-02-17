import torch
import json
import os
import argparse
from prompt import qa_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    parser.add_argument("--sentence_type", type=str)
    args = parser.parse_args()

    # =========================================== Load Dataset ===========================================    
    pipeline_types = ["vanilla", "atomic", "semantic"]

    for pipeline_type in pipeline_types:
        with open(f"../QG/yi-9b/{pipeline_type}_yi-9b.jsonl", 'r') as f_in, open(f"{args.output_path}-{pipeline_type}.jsonl", 'a') as f_out:
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

                    data['answers'] = str(final_generation)
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()