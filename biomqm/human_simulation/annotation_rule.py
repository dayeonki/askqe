import json


input_file = "raw/human_ratings.jsonl"
output_file = "classified/human_ratings.jsonl"


with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line.strip())
        
        if data.get("severity") in ["critical", "major"]:
            data["decision"] = "reject"
        else:
            data["decision"] = "accept"
        
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write("\n")
