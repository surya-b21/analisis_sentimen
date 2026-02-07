import os
import pandas as pd
import nltk
import json
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support


# download required nltk data
nltk.download('punkt')

# load indonesian stopwords
stop_words = set(stopwords.words('indonesian'))

# Initialize Sastrawi stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def preprocessing_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    words = [stemmer.stem(word) for word in words]

    # Return features as dictionary
    return " ".join(words)

# read csv file
def read_csv():
    df = pd.read_csv('vader_result.csv', encoding='utf-8', on_bad_lines='skip', engine='python')
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    data = []

    for _, row in df.iterrows():
        text = row['comment']
        sentiment = row['label'].strip()  # Remove whitespace from label
        processed_text = preprocessing_text(text)
        data.append((processed_text, sentiment))

    return data

# predict new input using trained model
def predict_sentiment(text):
    try:
        if not os.path.exists('naive_bayes_tfidf_model.joblib') or not os.path.exists('tfidf_vectorizer.joblib'):
            return {
                'error': 'Model or vectorizer file not found. Please train the model first.'
            }
        
        # Load the trained model and vectorizer
        model = joblib.load('naive_bayes_tfidf_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')

        # Preprocess the input text
        processed_text = preprocessing_text(text)

        # Vectorize the input text
        X_input = vectorizer.transform([processed_text])

        # Predict sentiment
        prediction = model.predict(X_input)
        return prediction[0]
    except Exception as e:
        return {
            'error': f'An error occurred during prediction: {str(e)}'
        }

def main():
    print("Loading data...")
    data = read_csv()
    print(f"Total samples: {len(data)}")

    # split data into 90% train and 10% test
    texts = [d[0] for d in data]
    labels = [d[1] for d in data]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # vectorize (tf-idf)
    print("Vectorizing data...")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # train model
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train, train_labels)

    # predict
    print("Making predictions...")
    predictions = model.predict(X_test)

    # evaluate
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))

    conf_matrix = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    
    # Get unique labels in sorted order
    labels_list = sorted(set(test_labels))
    
    # Create DataFrame for better visualization
    conf_df = pd.DataFrame(
        conf_matrix,
        index=[f'Actual {label}' for label in labels_list],
        columns=[f'Predicted {label}' for label in labels_list]
    )
    print(conf_df)

    # Calculate per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average=None
    )

    # Prepare results for JSON export
    per_class_metrics = {}
    for idx, label in enumerate(labels_list):
        per_class_metrics[label] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx])
        }

    results = {
        'model': 'Naive Bayes + TF-IDF',
        'overall_metrics': {
            'accuracy': float(accuracy),
            'accuracy_percentage': float(accuracy * 100)
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': {
            'matrix': conf_matrix.tolist(),
            'labels': labels_list
        },
        'dataset_info': {
            'total_samples': len(data),
            'train_samples': len(train_texts),
            'test_samples': len(test_texts),
            'classes': labels_list
        }
    }

    # Save to JSON
    with open('naive_bayes_tfidf_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Results saved to naive_bayes_tfidf_results.json")

    # Save trained model and vectorizer using joblib
    joblib.dump(model, 'naive_bayes_tfidf_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("✅ Model and vectorizer saved to naive_bayes_tfidf_model.joblib and tfidf_vectorizer.joblib")
    

if __name__ == "__main__":
    main()