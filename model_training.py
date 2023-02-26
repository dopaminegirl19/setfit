from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitTrainer



def train_setfit(model, train_data, test_data, keep_body_frozen=False):
    """ Wrapper function for training a ST, with option to freeze body.

    Parameters
    ----------
    model : setfit.SetFitModel
    train_data : DatasetDict
        Training data, should contain the desired number of shots
    test_data : DatasetDict
        Testing set
    keep_body_frozen : bool
        Flag to freeze or unfreeze body before training. Thus, True 
        indicates head-only training.

    Returns
    -------
    metrics : dictionary
        Containing "accuracy" metric 
    """
    # Create trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        loss_class=CosineSimilarityLoss,
        batch_size=8,
        num_iterations=20, # Number of text pairs to generate for contrastive learning
        num_epochs=1 # Number of epochs to use for contrastive learning
    )
    trainer.train()
    metrics = trainer.evaluate()
    return metrics