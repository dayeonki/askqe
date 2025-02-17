from comet import download_model, load_from_checkpoint
import json


def main():
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)

    languages = ["es", "fr", "hi", "tl", "zh-CN"]
    for language in languages:

        with open(f"en-{language}-gt.jsonl", "r", encoding="utf-8") as f, open(f"en-{language}-xcomet.jsonl", "w", encoding="utf-8") as output_file:
            for line in f:
                entry = json.loads(line.strip())
                pert_key = "gt_en"

                if "en" in entry and pert_key in entry:
                    data = [{
                        "src": entry["en"],
                        "mt": entry["gt_en"]
                    }]

                    model_output = model.predict(data, batch_size=1, gpus=1)
                    entry["xcomet_annotation"] = {
                        "segment_score": round(model_output.scores[0], 3),
                        "error_spans": model_output.metadata.get("error_spans", [])[0]
                    }

                    output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    print(f"Source: {entry['en']}")
                    print(f"Perturbed MT: {entry['gt_en']}")
                    print(f"Segment-level Score: {entry['xcomet_annotation']['segment_score']}")
                    print(f"Error Spans: {entry['xcomet_annotation']['error_spans']}")
                    print("=" * 80)


if __name__ == "__main__":
    main()
