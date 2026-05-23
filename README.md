# Symptom to Disease Classification

A machine learning project that classifies diseases based on symptom descriptions using multiple classification algorithms.

## Dataset
- **Source**: `Symptom2Disease.csv`
- **Task**: Multi-class text classification (symptom text → disease label)

## Preprocessing Pipeline
1. Remove non-breaking spaces and normalize punctuation
2. Tokenize and convert to lowercase
3. Remove stopwords and punctuation
4. POS tagging and lemmatization using WordNet

## Feature Extraction
- **TF-IDF Vectorizer** with 1,000 max features

## Models Compared
- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier
- Naive Bayes (Multinomial)

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (weighted average)
- Confusion Matrix
- Classification Report

## Results
Models are compared using bar charts and confusion matrices to identify the best performer for disease classification.

## Usage
Run the notebook cells in sequence:
1. Data loading and preprocessing
2. Model training and evaluation
3. Visualization of results

## Example Prediction
Test input: *"I've been sneezing a lot and my nose feels so clogged. I can't even smell anything!"*

