import copy
from pathlib import Path

import mlflow
from tqdm import tqdm
from logzero import logger
from setfit import SetFitModel

import data_loading as dl
import model_training as mt

## Configure model params
FREEZE_HEAD = True
FREEZE_BODY = False

## Set up mlflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
# mlflow_exp_id = 537995142806492016
mlflow_exp_id = mlflow.create_experiment(
    "body-only setfit training 2",
    artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
    tags={"version": "v1"},
)

## Set up experiment parameters
n_shots_range = [1, 5, 10, 25, 50, 75, 100]

for data_loading_fn, data_name in zip(
    [dl.load_yelp, dl.load_spam, dl.load_subjects], ["yelp_ratings", "spam", "subjects"]
):

    # Log
    logger.info(f"Processing {data_name}")

    # Load data
    num_classes, data = data_loading_fn()
    data = data.train_test_split(test_size=0.3)
    eval_dataset = data["test"]

    # Load SetFit model from Hub
    model_backup = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        use_differentiable_head=True,
        # multi_target_strategy="one-vs-rest",
        head_params={"out_features": num_classes},
    )
    if FREEZE_HEAD:
        model_backup.freeze("head")
    if FREEZE_BODY:
        model_backup.freeze("body")

    # Loop through number of shots
    for n_shots in tqdm(n_shots_range):
        # Record experiment in mlflow
        with mlflow.start_run(experiment_id=mlflow_exp_id):

            # Re-initialise the model for fair testing
            model = copy.deepcopy(model_backup)

            # Train model, get accuracy on evaluation dataset
            train_dataset = (
                data["train"].shuffle(seed=42).select(range(n_shots * num_classes))
            )
            metrics = mt.train_setfit(
                model, train_dataset, eval_dataset
            )
            accuracy = metrics["accuracy"]

            # Log results
            mlflow.log_param("dataset", data_name)
            mlflow.log_param("n_shots", n_shots)
            mlflow.log_param("accuracy", accuracy)
