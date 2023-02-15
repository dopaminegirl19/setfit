## Overview

Here I explore the few-shot learning performance using variations on the [SetFit](https://arxiv.org/abs/2209.11055) method for text classification modelling. 

## Datasets

Links for datasets used:
- [`yelp_ratings.csv`](https://www.kaggle.com/datasets/matleonard/nlp-course)
- [`spam.csv`](https://www.kaggle.com/datasets/matleonard/nlp-course)
- [`physics_chemistry_biology.csv`](https://www.kaggle.com/datasets/vivmankar/physics-vs-chemistry-vs-biology)


## Repo structure

The root of the repo is separated into 4 folders, which contain the following:
1. SetFit
2. SetFit but without fine-tuning the pre-trained ST (only train the classification head after ST preprocessing)
3. Regular supervised fine-tuning of the pre-trained ST connected to the classification head.
4. Prompting a language model of my choosing.

