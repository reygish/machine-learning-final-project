# Project Summary: Symptom to Disease Classification

## Objective
Develop and compare multiple machine learning models to classify diseases based on patient symptom descriptions using natural language processing and text classification techniques.

## Methodology
- **Text Preprocessing**: Comprehensive NLP pipeline including tokenization, lemmatization, POS tagging, and stopword removal
- **Feature Engineering**: TF-IDF vectorization (1,000 features) to convert text into numerical features
- **Model Selection**: Evaluated 4 different classifiers across multi-class disease classification task

## Models Evaluated
1. **Random Forest Classifier** - Ensemble method with 100 estimators
2. **Logistic Regression** - Linear classifier for baseline comparison
3. **Decision Tree Classifier** - Single tree for interpretability
4. **Naive Bayes** - Probabilistic classifier for fast training

## Results

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Random Forest | 0.9690 | 0.9667 | 0.9656 |
| Logistic Regression | 0.9772 | 0.9750 | 0.9749 |
| Naive Bayes | 0.9601 | 0.9500 | 0.9480 |
| Decision Tree | 0.8324 | 0.7917 | 0.7993 |

**Best Performer**: Logistic Regression (97.5% recall, 97.72% precision)

## Evaluation Metrics
- **Precision**: Of predicted positive cases, how many were correct
- **Recall**: Of actual positive cases, how many were identified
- **F1-Score**: Harmonic mean balancing precision and recall (weighted average)
- **Visualizations**: Accuracy comparison, per-model confusion matrices, precision/recall/F1 comparison charts

## Insights
- Logistic Regression achieved the highest overall performance
- Random Forest and Naive Bayes showed strong results with minimal performance difference
- Decision Tree significantly underperformed, suggesting overfitting to training data
- Confusion matrices reveal which disease pairs are often misclassified

## Files
- `machine_learning.ipynb`: Main analysis and model training
- `Symptom2Disease.csv`: Dataset
- `README.md`: Project documentation