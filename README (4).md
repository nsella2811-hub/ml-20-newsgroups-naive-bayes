# 20 Newsgroups Text Classification

Machine Learning assignment – Text Analysis / NLP

## Student
Noam S.  
Last 4 ID digits: 6331

## Project Overview
This project implements a multi-class text classification pipeline for the **20 Newsgroups (bydate)** dataset.

The objective is to classify each text document into one of **20 topic categories** using a manually implemented **Multinomial Naive Bayes** classifier.

## Dataset
**20 Newsgroups (bydate)**  
Kaggle: https://www.kaggle.com/datasets/mohamedtharwat/20news-bydate

The dataset contains predefined train and test sets:

- Train: **11,314** documents
- Test: **7,532** documents
- Number of classes: **20**

The original train/test split is preserved and no additional `train_test_split` is performed.

## Data Preparation
Each document is loaded from the original directory structure.  
The directory name is used as the class label.

Text preprocessing includes:

- Converting text to lowercase
- Removing numbers and special characters
- Normalizing whitespace
- Removing English stop words during vectorization

## Feature Engineering
Two text representations are evaluated:

- **Bag of Words**
- **TF-IDF**

Additional feature-engineering parameters include:

- `max_features = 5000 / 10000`
- `ngram_range = (1,1) / (1,2)`
- `min_df = 2`

The vectorizer is always fitted only on the training data. Validation and test data are transformed using the already fitted vectorizer in order to avoid data leakage.

## Learning Algorithm
The classifier is a manually implemented **Multinomial Naive Bayes** model.

The implementation includes:

- Class prior probabilities
- Per-class feature probabilities
- Log-probability computation
- Laplace/Lidstone smoothing
- Configurable `alpha`
- `fit`
- `predict`

The model predicts the class with the highest log-probability score.

## Evaluation Metric
Because this is a 20-class classification problem, the main evaluation metric is:

**Macro-average F1**

This metric calculates F1 independently for each class and then gives all classes equal weight in the final average.

## Model Selection
Hyperparameter and feature-engineering selection is performed using:

- **Grid Search**
- **5-Fold Stratified Cross Validation**

The following combinations are evaluated:

- Bag of Words / TF-IDF
- 5,000 / 10,000 features
- Unigrams / Unigrams + Bigrams
- `alpha = 0.5 / 1.0`

A total of **16 parameter combinations** are evaluated.

## Best Configuration
The best cross-validation configuration is:

- Feature representation: **TF-IDF**
- `max_features = 10000`
- `ngram_range = (1,1)`
- `min_df = 2`
- `alpha = 0.5`

Best mean cross-validation Macro-F1:

**0.871**

## Final Evaluation
After selecting the best configuration, the model is retrained on the full training set and evaluated on the predefined test set.

Final results:

- Test Macro-F1: **0.8016**
- Test Accuracy: **~0.81**

The notebook also includes:

- Feature-engineering examples
- First test predictions
- Classification report
- Confusion matrix

## AI Assistance
ChatGPT was used for:

- Understanding the assignment requirements
- Assistance with specific code sections
- Checking and refining dataset details
- Reviewing the implementation against the assignment requirements

The notebook includes examples of the prompts used.

## Files
- `ml_assignment_20newsgroups_naive_bayes_SUBMISSION_FINAL.ipynb` – full notebook with code, explanations and saved outputs
- `README.md` – project documentation

## Reproducibility
The notebook is saved with execution outputs so the complete workflow and results can be reviewed without rerunning the experiment.
