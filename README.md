# 20 Newsgroups Text Classification

Machine Learning assignment – Text Analysis / NLP

## Student
Noam S.  
Last 4 ID digits: 6331

## Project Overview
This project performs multi-class text classification on the **20 Newsgroups (bydate)** dataset using a manually implemented **Multinomial Naive Bayes** classifier.

## Dataset
**20 Newsgroups (bydate)**  
Kaggle: https://www.kaggle.com/datasets/mohamedtharwat/20news-bydate

- Train: **11,314** documents
- Test: **7,532** documents
- Classes: **20**

The predefined train/test split is preserved. No additional `train_test_split` is performed.

## Preprocessing and Feature Engineering
Text preprocessing includes:
- Lowercasing
- Removing numbers and special characters
- Whitespace normalization

The text is represented using **TF-IDF** with:
- `max_features = 10000`
- `ngram_range = (1, 1)`
- `min_df = 2`
- English stop-word removal

The vectorizer is fitted only on the training data. The test set is transformed using the fitted vectorizer.

## Learning Algorithm
**Multinomial Naive Bayes** is implemented manually in the notebook.

The implementation includes:
- Class priors
- Per-class feature probabilities
- Log-space calculations
- Smoothing using `alpha = 0.5`
- `fit`
- `predict`

## Evaluation
The main evaluation metric is **Macro-average F1**.

The final model is trained on the full training set and evaluated on the predefined test set.

Final Test Macro-F1: **approximately 0.80**  
Test Accuracy: **approximately 0.81**

The notebook also includes:
- Train and test examples
- Feature-engineering examples
- First test predictions
- Classification report
- Confusion matrix

## AI Assistance
ChatGPT was used for understanding the assignment requirements, assistance with specific code sections, and checking dataset and implementation details.

## Files
- `ml_assignment_20newsgroups_naive_bayes_NO_BONUS_FINAL.ipynb` – complete notebook with code, explanations and saved outputs
- `README.md` – project summary
