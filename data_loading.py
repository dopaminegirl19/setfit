from datasets import load_dataset

def data_loader(n_classes, path):
    data = load_dataset("csv", data_files=path)["train"]
    return n_classes, data

def load_yelp():
    return data_loader(n_classes=2, path="datasets/yelp_ratings.csv")

def load_spam():
    return data_loader(n_classes=2, path="datasets/spam.csv")

def load_subjects():
    return data_loader(n_classes=3, path="datasets/physics_chemistry_biology.csv")