import torch
import json
import argparse
from prompt import qa_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "google/gemma-2-9b-it"

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir="",
        device_map="auto",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--sentence_type", type=str)
    args = parser.parse_args()

    # =========================================== Load Dataset ===========================================    
    pipeline_types = ["vanilla", "atomic", "semantic"]

    for pipeline_type in pipeline_types:
        with open(f"../QG/gemma-9b/{pipeline_type}_gemma-9b.jsonl", 'r') as f_in, open(f"{args.output_path}-{pipeline_type}.jsonl", 'a') as f_out:
            for line in f_in:
                data = json.loads(line)

                sentence = data.get(args.sentence_type, None)
                questions = data.get("questions", None)

                if sentence and questions:
                    prompt_template = qa_prompt
                    prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{questions}}", questions)
                    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

                    with torch.no_grad():
                        outputs = model.generate(
                            **input_ids,
                            max_new_tokens=1024,
                            num_beams=1,
                        )
                    
                    generated_questions = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    answer_start = "Answers: "
                    if answer_start in generated_questions:
                        generation = generated_questions.split(answer_start)[-1].strip()
                        generation = generation.split("<")[0].strip()
                    else:
                        generation = generated_questions

                    print(f"{prompt}")
                    print(f"> {generation}")
                    print("\n======================================================\n")

                    data['answers'] = str(generation)
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()