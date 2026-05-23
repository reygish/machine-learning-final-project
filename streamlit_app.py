import streamlit as st
import joblib
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from string import punctuation

@st.cache_resource
def load_models():
    models = {
        'Random Forest': joblib.load('models/random_forest.joblib'),
        'Logistic Regression': joblib.load('models/logistic_regression.joblib'),
        'Decision Tree': joblib.load('models/decision_tree.joblib'),
        'Naive Bayes': joblib.load('models/naive_bayes.joblib')
    }
    vectorizer = joblib.load('vectorizer.joblib')
    return models, vectorizer

models, vectorizer = load_models()

ENGLISH_STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tag_to_wordnet(tag: str):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('R'):
        return wordnet.ADV
    if tag.startswith('N'):
        return wordnet.NOUN
    return wordnet.NOUN

def preprocess_text(text: str):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS and t not in punctuation and t.isalpha()]
    
    tagged_tokens = pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, tag_to_wordnet(tag)) for token, tag in tagged_tokens]
    
    return ' '.join(lemmatized_tokens)

st.set_page_config(page_title="Symptom to Disease Classifier", layout="wide")
st.title(":hospital: Symptom to Disease Classification")

st.write("Enter your symptoms and select a model to get a disease prediction.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Test Prompt")
    user_input = st.text_area(
        "Enter your symptoms:",
        value="I've been sneezing a lot and my nose feels so clogged. I can't even smell anything!",
        height=120
    )

with col2:
    st.subheader("Select Model")
    selected_model = st.radio(
        "Choose a classification model:",
        options=list(models.keys()),
        index=0
    )

if st.button("Get Prediction", use_container_width=True):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        input_tfidf = vectorizer.transform([preprocessed_input])
        
        prediction = models[selected_model].predict(input_tfidf)[0]
        confidence = models[selected_model].predict_proba(input_tfidf).max()
        
        st.success(f"### Predicted Disease: **{prediction}**")
        st.progress(confidence, text=f"Confidence: {confidence:.2%}")
        st.info(f"Model used: {selected_model}")
    else:
        st.warning("Please enter some symptoms first.")

st.divider()
st.subheader("Model Information")
st.write("""
- **Random Forest**: Ensemble method with 100 estimators
- **Logistic Regression**: Linear classifier (best performer - 97.5% accuracy)
- **Decision Tree**: Single tree for interpretability
- **Naive Bayes**: Probabilistic classifier
""")