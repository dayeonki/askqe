# Based on https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py

from collections import Counter
import string
import re
import sacrebleu
from typing import List, Union


def split_list(mylist: List, chunk_size: Union[int]):
    return [mylist[offs:offs + chunk_size] for offs in range(0, len(mylist), chunk_size)]


def normalize_answer(s: Union[str]):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_articles(remove_punc(s)))



def f1_score(prediction: Union[str], ground_truth: Union[str], normalize=False):
    """Compute word-level F1 score"""
    if normalize:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
    else:
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: Union[str], ground_truth: Union[str], normalize=False):
    """Compute word-level EM score"""
    if normalize:
        return normalize_answer(prediction) == normalize_answer(ground_truth)
    return prediction == ground_truth


def chrf_score(prediction: Union[str], golden_truth: Union[str], normalize=False):
    """Compute sentence-level chrf score"""
    if normalize:
        return sacrebleu.sentence_chrf(normalize_answer(prediction),
                                       [normalize_answer(golden_truth)]).score
    else:
        return sacrebleu.sentence_chrf(prediction,
                                       [golden_truth]).score


def bleu_score(prediction: Union[str], golden_truth: Union[str], normalize=False):
    """Compute sentence-level bleu score"""
    if normalize:
        return sacrebleu.sentence_bleu(normalize_answer(prediction),
                                       [normalize_answer(golden_truth)]).score
    else:
        return sacrebleu.sentence_bleu(prediction,
                                       [golden_truth]).score


def compare_answers(prediction: Union[str], golden_truth: Union[str], normalize=True):
    return (
        f1_score(prediction, golden_truth, normalize),
        exact_match_score(prediction, golden_truth, normalize),
        chrf_score(prediction, golden_truth, normalize),
        bleu_score(prediction, golden_truth, normalize)
    )