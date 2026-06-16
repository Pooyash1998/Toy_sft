import json
import os
import urllib.request
import zipfile


def download(url, dest):
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def prepare_mlqa():
    os.makedirs("data/mlqa", exist_ok=True)
    zip_path = "data/mlqa/MLQA_V1.zip"
    download("https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip", zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for split, keyword in [("test", "test-context-en-question-en"),
                                ("validation", "dev-context-en-question-en")]:
            match = next((n for n in names if keyword in n), None)
            if match is None:
                raise FileNotFoundError(f"Could not find {keyword} in zip. Files: {names[:20]}")
            with zf.open(match) as f:
                data = json.load(f)

            records = []
            for article in data["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        records.append({
                            "context": context,
                            "question": qa["question"],
                            "answers": {
                                "text": [a["text"] for a in qa["answers"]],
                                "answer_start": [a["answer_start"] for a in qa["answers"]],
                            },
                        })
            out = f"data/mlqa/{split}_en_en.jsonl"
            with open(out, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"MLQA {split}: {len(records)} records → {out}")


def prepare_mathqa():
    os.makedirs("data/mathqa", exist_ok=True)
    zip_path = "data/mathqa/MathQA.zip"
    download("https://math-qa.github.io/math-QA/data/MathQA.zip", zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for split, filename in [("train", "train.json"), ("test", "test.json"), ("validation", "dev.json")]:
            match = next((n for n in names if n.endswith(filename)), None)
            if match is None:
                raise FileNotFoundError(f"Could not find {filename} in zip. Files: {names[:20]}")
            with zf.open(match) as f:
                records = json.load(f)
            out = f"data/mathqa/{split}.json"
            with open(out, "w") as f:
                json.dump(records, f)
            print(f"MathQA {split}: {len(records)} records → {out}")


if __name__ == "__main__":
    prepare_mlqa()
    prepare_mathqa()