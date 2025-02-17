from comet import download_model, load_from_checkpoint
import json


def main():
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)

    languages = ["es", "fr", "hi", "tl", "zh"]
    perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact",
                    "intensifier", "expansion_impact", "omission", "alteration"]
    
    for language in languages:
        for perturbation in perturbations:
            print("Language: ", language)
            print("Perturbation: ", perturbation)

            reference_file = f"../QA/llama-70b/en-atomic.jsonl"
            prediction_file = f"../QA/llama-70b/{language}-atomic-{perturbation}.jsonl"
            output_file = f"en-{language}/{perturbation}.jsonl"

            try:
                with open(prediction_file, "r", encoding="utf-8") as pred_file, open(reference_file, "r", encoding="utf-8") as ref_file, open(output_file, "w", encoding="utf-8") as output_path:
                    for pred_line, ref_line in zip(pred_file, ref_file):
                        datas = []
                        try:
                            pred_data = json.loads(pred_line)
                            ref_data = json.loads(ref_line)

                            predicted_answers = pred_data.get("answers", [])
                            reference_answers = ref_data.get("answers", [])

                            if isinstance(predicted_answers, str):
                                try:
                                    predicted_answers = json.loads(predicted_answers)
                                except json.JSONDecodeError:
                                    continue

                            if isinstance(reference_answers, str):
                                try:
                                    reference_answers = json.loads(reference_answers)
                                except json.JSONDecodeError:
                                    continue

                            if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list):
                                continue
                            if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers):
                                continue

                            for pred, ref in zip(predicted_answers, reference_answers):
                                if not isinstance(pred, str) or not isinstance(ref, str):
                                    continue
                                if pred.strip() == "" or ref.strip() == "":
                                    continue

                                datas.append({"src": ref, "mt": pred})
                                model_output = model.predict(datas, batch_size=1, gpus=1)

                                pred_data["xcomet_annotation"] = {
                                    "segment_score": round(model_output.scores[0], 3),
                                    "error_spans": model_output.metadata.get("error_spans", [])[0]
                                }

                                output_path.write(json.dumps(pred_data, ensure_ascii=False) + '\n')

                                print(f"Source: {pred_data['en']}")
                                print(f"Segment-level Score: {pred_data['xcomet_annotation']['segment_score']}")
                                print(f"Error Spans: {pred_data['xcomet_annotation']['error_spans']}")
                                print("=" * 80)

                        except json.JSONDecodeError as e:
                                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                                continue

            except FileNotFoundError as e:
                print(f"File not found: {e}")
                continue


if __name__ == "__main__":
    main()
