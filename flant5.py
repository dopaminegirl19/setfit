import copy
from pathlib import Path

import data_loading as dl
import evaluate
import mlflow
import nltk
import numpy as np
import utils
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

nltk.download("punkt")

# Load model and tokenizer
model_id = "google/flan-t5-small"
model_backup = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Metric
metric = evaluate.load("rouge")

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir="output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False,  # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=1,
    # logging & evaluation strategies
    logging_dir=f"output/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Set up mlflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow_exp_id = mlflow.create_experiment(
    "flant5",
    artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
    tags={"version": "v1"},
)

## Set up experiment parameters
n_shots_range = [1, 5, 10, 25, 50, 75, 100]

for data_loading_fn, data_name in zip(
    [dl.load_yelp, dl.load_spam, dl.load_subjects], ["yelp_ratings", "spam", "subjects"]
):

    # Load data
    num_classes, dataset = data_loading_fn()
    data = dataset.train_test_split(test_size=0.1)

    for n_shots in tqdm(n_shots_range):
        with mlflow.start_run():

            # make a copy of the model
            model = copy.deepcopy(model_backup)

            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
            )
            # Select training data
            train_data = data["train"].shuffle(seed=42).select(range(n_shots * num_classes))

            # The maximum total input sequence length after tokenization.
            # Sequences longer than this will be truncated, sequences shorter will be padded.
            tokenized_inputs = concatenate_datasets([train_data, data["test"]]).map(
                lambda x: tokenizer(x["text"], truncation=True),
                batched=True,
                remove_columns=list(data["train"].features.keys()),
            )
            max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    
            # The maximum total sequence length for target text after tokenization.
            # Sequences longer than this will be truncated, sequences shorter will be padded."
            tokenized_targets = concatenate_datasets([train_data, data["test"]]).map(
                lambda x: tokenizer(x["label_text"], truncation=True),
                batched=True,
                remove_columns=list(data["train"].features.keys()),
            )
            max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    
            # Subselect training data samples 
            dataset_to_tokenize = copy.deepcopy(data)
            dataset_to_tokenize['train'] = train_data

            tokenized_dataset = data.map(lambda a: utils.preprocess_function(sample=a, tokenizer=tokenizer, max_source_length=max_source_length, max_target_length=max_target_length), batched=True, remove_columns=list(data['train'].features.keys()))
    
            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                if isinstance(preds, tuple):
                    preds = preds[0]
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
                # Some simple post-processing
                decoded_preds, decoded_labels = utils.postprocess_text(decoded_preds, decoded_labels)
            
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
                result = {k: round(v * 100, 4) for k, v in result.items()}
                prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
                result["gen_len"] = np.mean(prediction_lens)
                return result
    
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                compute_metrics=compute_metrics,
            )
    
            # Train + evaluate
            trainer.train()
            metrics = trainer.evaluate()

            # Log results
            mlflow.log_param("dataset", data_name)
            mlflow.log_param("n_shots", n_shots)
            mlflow.log_param("eval_loss", metrics['eval_loss'])
