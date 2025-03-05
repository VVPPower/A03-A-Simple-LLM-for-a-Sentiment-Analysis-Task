# ==================================
# CISC 691 - NextGenAI
# Sample Fine tuning code for sentiment analysis
# ==================================

# standard python libraries
import logging
import json
import random
from datetime import datetime
from collections import Counter
from pathlib import Path

# HuggingFace libraries
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# sklearn, py-torch libraries
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch

# --------------------------------------
#  Set up python logging to console and a log file
# --------------------------------------
def configure_logging(loglevel: str = "DEBUG"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"fine_tune_{current_time}.log"
    numeric_level = getattr(logging, loglevel.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_filename),  # File output
        ]
    )

LOG_LEVEL = "DEBUG"
configure_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

# --------------------------------------
#  constants
# --------------------------------------
# location of intermediate results and final model
RESULTS_DIR = Path(r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03").resolve()

# model and dataset names to download from HuggingFace
BASE_MODEL_NAME = "distilbert-base-uncased"
SENTIMENT_DATASET_NAME = "imdb"

# --------------------------------------
#  Optimization tweaks: refer to torch documentation
# --------------------------------------
# Use MPS (Apple Silicon GPU) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --------------------------------------
#  Defines a class for storing the sentiment datasets used for training
# --------------------------------------
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val, dtype=torch.long) for key, val in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].to(device) for key, val in self.encodings.items()}  # Move data to device
        item["labels"] = self.labels[idx].to(device)
        return item

# --------------------------------------
#  Load the base model: downloads from HF, reduces its size, then stores locally
# --------------------------------------
def load_base_model(model_name=BASE_MODEL_NAME):
    logger.info(f"Loading model...{model_name}")
    # Load the original model and tokenizer
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Reduce model size by creating a new configuration
    config = DistilBertConfig(
        n_layers=3,                 # Default is 6 → Reduce to 3 layers
        hidden_size=384,            # Default is 768 → Reduce to 384
        intermediate_size=768,      # Default is 3072 → Reduce to 768
        num_labels=2                # Binary classification (Positive/Negative)
    )

    # Load the reduced model and move it to the appropriate device
    base_model = DistilBertForSequenceClassification(config)
    base_model.to(device)

    # Save the reduced model and tokenizer locally
    base_model_dir = RESULTS_DIR / "results2" / "base_model"
    base_model_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(str(base_model_dir))
    tokenizer.save_pretrained(str(base_model_dir))

    logger.info("Base model loaded and saved.")
    return base_model, tokenizer

# --------------------------------------
#  Load the sentiment dataset: downloads from HF, extracts a subset then stores locally
# --------------------------------------
def load_sentiment_dataset(tokenizer, dataset_name=SENTIMENT_DATASET_NAME):
    logger.info(f"Loading dataset...{dataset_name}")
    dataset = load_dataset(dataset_name)

    # Convert dataset to list to filter by label
    train_data = list(dataset["train"])
    test_data = list(dataset["test"])

    # Separate positive and negative samples
    positive_train = [ex for ex in train_data if ex["label"] == 1]
    negative_train = [ex for ex in train_data if ex["label"] == 0]

    positive_test = [ex for ex in test_data if ex["label"] == 1]
    negative_test = [ex for ex in test_data if ex["label"] == 0]

    # Select equal numbers of positive and negative examples
    subset_train = random.sample(positive_train, 1000) + random.sample(negative_train, 1000)
    subset_test = random.sample(positive_test, 250) + random.sample(negative_test, 250)

    # Shuffle to mix classes
    random.shuffle(subset_train)
    random.shuffle(subset_test)

    # Check new label distribution
    print("New Train Distribution:", Counter([ex["label"] for ex in subset_train]))
    print("New Test Distribution:", Counter([ex["label"] for ex in subset_test]))

    # Convert back to Dataset format
    subset_train = Dataset.from_list(subset_train)
    subset_test = Dataset.from_list(subset_test)

    # Tokenization function
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    # Tokenize datasets
    tokenized_train = subset_train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test = subset_test.map(tokenize_function, batched=True, remove_columns=["text"])

    logger.info("Saving sentiment dataset...")
    # Save train and test datasets to separate directories
    train_dir = RESULTS_DIR / "results2" / "imdb_train_subset"
    test_dir = RESULTS_DIR / "results2" / "imdb_test_subset"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    tokenized_train.save_to_disk(str(train_dir))
    tokenized_test.save_to_disk(str(test_dir))
    logger.info("Dataset saved.")

# --------------------------------------
#  Run the training step
# --------------------------------------
def train_model(base_model, tokenizer, train_dataset, test_dataset):
    logger.info("Starting training...")

    # Prepare training dataset
    train_encodings = {key: train_dataset[key] for key in train_dataset.features if key != "label"}
    train_labels = train_dataset["label"]
    train_data = SentimentDataset(train_encodings, train_labels)

    # Prepare test dataset
    test_encodings = {key: test_dataset[key] for key in test_dataset.features if key != "label"}
    test_labels = test_dataset["label"]
    test_data = SentimentDataset(test_encodings, test_labels)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR.as_posix(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=6,
        weight_decay=0.01,
        load_best_model_at_end=True,
        bf16=True,
    )

    # Define metrics computation function
    def compute_metrics(pred):
        predictions = torch.argmax(torch.tensor(pred.predictions), dim=-1).cpu().numpy()
        labels = pred.label_ids
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc, "f1": f1}

    # Create Trainer instance
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )
    logger.info("Running training... This may take a while.")
    #trainer.train()
    trainer.train(resume_from_checkpoint=True) # Resume training from checkpoint(As my laptop got interrupted)



    logs = trainer.state.log_history
    print("Training Logs:", logs)

    logger.info("Running evaluation...")
    results = trainer.evaluate(metric_key_prefix="eval")
    print("Evaluation Results:", results)

    # Save the fine-tuned model and tokenizer
    fine_tuned_dir = RESULTS_DIR / "fine_tuned_model"
    fine_tuned_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving fine-tuned model...")
    trainer.save_model(str(fine_tuned_dir))
    tokenizer.save_pretrained(str(fine_tuned_dir))

    logger.info("Training and saving complete.")

# --------------------------------------
# Function to predict sentiment for a single text input
# --------------------------------------
def predict_sentiment(text, model, tokenizer):
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_label = torch.argmax(logits, dim=1).item()
    return "Positive" if pred_label == 1 else "Negative"

# --------------------------------------
# New function: Evaluate sample sentences from a JSON file
# --------------------------------------
def evaluate_sample_sentences(file_path, model, tokenizer):
    logger.info(f"Evaluating sample sentences from file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    for sample in samples:
        sentence = sample["sentence"]
        language = sample.get("language", "en")
        ground_truth = sample.get("sentiment", None)
        predicted = predict_sentiment(sentence, model, tokenizer)
        print("-" * 50)
        print(f"Sentence:             {sentence}")
        print(f"Language:             {language}")
        print(f"Predicted Sentiment:  {predicted}")
        print(f"Ground Truth:         {ground_truth}")
    print("-" * 50)

# --------------------------------------
# Main function for flow
# --------------------------------------
def main():
    # Load base model and tokenizer
    model, tokenizer = load_base_model()

    # Define dataset paths
    train_path = RESULTS_DIR / "results2" / "imdb_train_subset"
    test_path = RESULTS_DIR / "results2" / "imdb_test_subset"

    # If datasets do not exist, create them
    if not train_path.exists() or not test_path.exists():
        load_sentiment_dataset(tokenizer)

    # Load datasets from disk
    train_dataset = load_from_disk(str(train_path))
    test_dataset = load_from_disk(str(test_path))
    
    # Run training
    train_model(model, tokenizer, train_dataset, test_dataset)
    
    # Predict sentiment for sample inputs
    test_texts = ["The movie was great!", "I did not enjoy the movie."]
    for text in test_texts:
        sentiment = predict_sentiment(text, model, tokenizer)
        print(f"Text: {text} | Sentiment: {sentiment}")

    # Evaluate additional sample sentences from the JSON file
    sample_file_path = r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\results2\sample_sentences_2.json"
    evaluate_sample_sentences(sample_file_path, model, tokenizer)

if __name__ == "__main__":
    main()
