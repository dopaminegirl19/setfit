from datasets import Dataset
import pandas as pd

def data_loader(n_classes, path):
    data = pd.read_csv(path)
    data = Dataset.from_pandas(data)
    return n_classes, data

def load_yelp():
    return data_loader(n_classes=2, path="datasets/yelp_ratings.csv")

def load_spam():
    return data_loader(n_classes=2, path="datasets/spam.csv")

def load_subjects():
    return data_loader(n_classes=3, path="datasets/physics_chemistry_biology.csv")