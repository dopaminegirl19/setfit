import pandas as pd

# Spam dataset
spam = pd.read_csv("datasets/spam.csv")
spam["label_text"] = spam["label"]
spam["label"] = spam["label_text"] == "spam"
spam["label"] = spam["label"].apply(int)
spam.to_csv("datasets/spam.csv", index=False)

# Yelp reviews
yelp = pd.read_csv("datasets/yelp_ratings.csv")
yelp = yelp.drop(columns="stars")
yelp.to_csv("datasets/yelp_ratings.csv", index=False)

# Physics / chemistry / biology
subjects = pd.read_csv("datasets/physics_chemistry_biology.csv")
subjects_dict = {"Physics": 0, "Chemistry": 1, "Biology": 2}
subjects["label"] = subjects["Topic"].apply(lambda x: subjects_dict[x])
subjects = subjects.drop(columns="Id")
subjects = subjects.rename(columns={"Topic": "label_text", "Comment": "text"})
subjects.to_csv("datasets/physics_chemistry_biology.csv", index=False)
