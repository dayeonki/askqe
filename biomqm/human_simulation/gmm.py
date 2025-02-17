import json
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


input_jsonl = "askqe_sbert.jsonl"
output_jsonl = "classified/askqe_sbert.jsonl"

scores = []
data_entries = []

with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        if "avg_cos_sim" in entry:
            scores.append(entry["avg_cos_sim"])
            data_entries.append(entry)
        else:
            scores.append(0.0)
            data_entries.append(entry)

scores = np.array(scores).reshape(-1, 1)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(scores)

probabilities = gmm.predict_proba(scores)

mean_low = np.min(gmm.means_)  # Mean of the lower-quality translations
mean_high = np.max(gmm.means_)  # Mean of the higher-quality translations
threshold = (mean_low + mean_high) / 2
print(f"Threshold for rejection: {threshold:.4f}")


for i, entry in enumerate(data_entries):
    entry["p_reject"] = probabilities[i, np.argmin(gmm.means_)]  # Probability of rejection
    if "avg_xcomet" in entry:
        entry["decision"] = "reject" if entry["avg_xcomet"] < threshold else "accept"
    else:
        entry["decision"] = "accept"

with open(output_jsonl, "w", encoding="utf-8") as f:
    for entry in data_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


plt.scatter(scores, [entry["p_reject"] for entry in data_entries], color='pink', label="P(Reject | AskQE)")
plt.axvline(threshold, color='black', linestyle="--", label=f"Threshold: {threshold:.4f}")
plt.xlabel("AskQE")
plt.ylabel("Probability of Rejection")
plt.title("GMM Rejection Probability (AskQE)")
plt.legend()
plt.savefig("gmm/gmm_askqe.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()