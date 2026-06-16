from datasets import load_dataset, DatasetDict

DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
SAVE_PATH = "./data/ultrafeedback_binarized"
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 500

def prepare_data():
    raw_dataset = load_dataset(DATASET_ID)
    print("Raw dataset loaded:", list(raw_dataset.keys()))
    train = raw_dataset["train_sft"].select(range(TRAIN_SAMPLES))
    test = raw_dataset["test_sft"].select(range(TEST_SAMPLES))
    print("Columns:", train.column_names)
    print("\n sampel row:")
    for i in train[0]["messages"]:
        print(f" [{i['role']}]: {i['content'][:100]}")
    dataset = DatasetDict({"train": train, "test": test})
    dataset.save_to_disk(SAVE_PATH)
    print(f"\nDataset saved to {SAVE_PATH}")

if __name__ == "__main__":
    prepare_data()

