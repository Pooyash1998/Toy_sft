import re
import string
import sys
import unicodedata
from collections import Counter

import datasets

PUNCT = {
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)
WHITESPACE_LANGS = ["en", "es", "hi", "vi", "de", "ar"]
MIXED_SEGMENTATION_LANGS = ["zh"]


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r"[一-龥]", char) or char in PUNCT:
            if temp_str != "":
                segs_out.extend(whitespace_tokenize(temp_str))
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char
    if temp_str != "":
        segs_out.extend(whitespace_tokenize(temp_str))
    return segs_out


def normalize_answer(s, lang):
    def remove_articles(text, lang):
        if lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", text)
        elif lang == "de":
            return re.sub(r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b", " ", text)
        elif lang == "es":
            return re.sub(r"\b(un|una|unos|unas|el|la|los|las)\b", " ", text)
        return text

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception(f"Unknown Language {lang}")
        return " ".join(t for t in tokens if t.strip())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    return white_space_fix(remove_articles(remove_punc(s.lower()), lang), lang)


def f1_score(prediction, ground_truth, lang):
    p_tokens = normalize_answer(prediction, lang).split()
    g_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth, lang):
    return normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang):
    return max(metric_fn(prediction, gt, lang) for gt in ground_truths)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process(doc):
        return {
            "context": doc["context"],
            "question": doc["question"],
            "answers": doc["answers"]["text"],
        }
    return dataset.map(_process)


def process_results_en(doc, results):
    prediction = results[0].strip()
    ground_truths = doc["answers"]
    return {
        "exact_match": metric_max_over_ground_truths(exact_match_score, prediction, ground_truths, "en"),
        "f1": metric_max_over_ground_truths(f1_score, prediction, ground_truths, "en"),
    }