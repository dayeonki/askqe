import json
import collections
import pandas as pd


input_file = "dev_with_backtranslation.jsonl"

lang_src_count = collections.Counter()
tgt_lang_count = collections.Counter()
error_category_count = collections.Counter()
error_subcategory_count = collections.Counter()
error_severity_count = collections.Counter()
error_combined_count = collections.Counter()
langpair_severity_count = collections.Counter()


with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        lang_src = data["lang_src"]
        lang_tgt = data["lang_tgt"]

        lang_src_count[lang_src] += 1
        tgt_lang_count[lang_tgt] += 1

        if "errors_tgt" in data and isinstance(data["errors_tgt"], list):
            if not data["errors_tgt"]:
                error_category_count["No Error"] += 1
                error_severity_count["No Error"] += 1
                langpair_key = f"{lang_src}-{lang_tgt} - No Error"
                langpair_severity_count[langpair_key] += 1
            else:
                for error in data["errors_tgt"]:
                    category = error["error_category"]
                    subcategory = error["error_subcategory"]
                    severity = error["severity"]

                    error_category_count[category] += 1
                    error_subcategory_count[subcategory] += 1
                    error_severity_count[severity] += 1

                    combined_key = f"{category} - {subcategory}"
                    error_combined_count[combined_key] += 1

                    langpair_key = f"{lang_src}-{lang_tgt} - {severity}"
                    langpair_severity_count[langpair_key] += 1


df_lang = pd.DataFrame({"Language Source": list(lang_src_count.keys()), "Count": list(lang_src_count.values())})
df_tgt = pd.DataFrame({"Target Language": list(tgt_lang_count.keys()), "Count": list(tgt_lang_count.values())})
df_category = pd.DataFrame({"Error Category": list(error_category_count.keys()), "Count": list(error_category_count.values())})
df_subcategory = pd.DataFrame({"Error Subcategory": list(error_subcategory_count.keys()), "Count": list(error_subcategory_count.values())})
df_severity = pd.DataFrame({"Error Severity": list(error_severity_count.keys()), "Count": list(error_severity_count.values())})
df_combined = pd.DataFrame({"Error Category-Subcategory": list(error_combined_count.keys()), "Count": list(error_combined_count.values())})
df_langpair_severity = pd.DataFrame({"Language Pair - Error Severity": list(langpair_severity_count.keys()), "Count": list(langpair_severity_count.values())})

print(df_lang)
print(df_tgt)
print(df_category)
print(df_subcategory)
print(df_severity)
print(df_combined)
print(df_langpair_severity)
