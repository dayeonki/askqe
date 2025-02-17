import json
import argparse
from comet import download_model, load_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)

    with open(f"{args.input_path}", "r", encoding="utf-8") as f, open(f"{args.output_path}", "w", encoding="utf-8") as output_file:
        for line in f:
            entry = json.loads(line.strip())

            if "src" in entry and "tgt" in entry:
                data = [{
                    "src": entry["src"],
                    "mt": entry["tgt"]
                }]

                model_output = model.predict(data, batch_size=1, gpus=1)

                entry["xcomet_annotation"] = {
                    "segment_score": round(model_output.scores[0], 3),
                    "error_spans": model_output.metadata.get("error_spans", [])[0]
                }

                output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                output_file.flush()

                print(f"Source: {entry['src']}")
                print(f"MT: {entry['tgt']}")
                print(f"Segment-level Score: {entry['xcomet_annotation']['segment_score']}")
                print(f"Error Spans: {entry['xcomet_annotation']['error_spans']}")
                print("=" * 80)


if __name__ == "__main__":
    main()
