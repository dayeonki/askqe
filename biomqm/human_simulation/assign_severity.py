import json

# Define severity levels
severity_levels = {"Critical": 4, "Major": 3, "Minor": 2, "Neutral": 1, "No error": 0}

def get_highest_severity(xcomet_annotations):
    """Retrieve the highest severity from error spans across all xcomet_annotation entries."""
    max_severity = "No error"

    if not xcomet_annotations:
        return max_severity

    for annotation in xcomet_annotations:
        for error in annotation:
            error_severity = error.get("severity", "No error")
            if severity_levels[error_severity] > severity_levels[max_severity]:
                max_severity = error_severity  # Update max severity

    return max_severity


def process_jsonl(input_file, output_file):
    """Process JSONL file and assign highest severity."""
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            data["severity"] = get_highest_severity(data.get("errors_tgt", []))  # Ensure it’s a list
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


# Example usage
input_jsonl = "dev_xcomet_aligned.jsonl"  # Replace with actual file path
output_jsonl = "dev_xcomet_aligned_high.jsonl"  # Replace with actual file path
process_jsonl(input_jsonl, output_jsonl)
