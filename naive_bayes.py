import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


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
    df = pd.read_csv('dataset_used.csv', encoding='utf-8', on_bad_lines='skip', engine='python')
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    data = []

    for _, row in df.iterrows():
        text = row['comment']
        sentiment = row['label'].strip()  # Remove whitespace from label
        processed_text = preprocessing_text(text)
        data.append((processed_text, sentiment))

    return data

def main():
    print("Loading data...")
    data = read_csv()
    print(f"Total samples: {len(data)}")

    # split data into 80% train and 20% test
    texts = [d[0] for d in data]
    labels = [d[1] for d in data]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
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

if __name__ == "__main__":
    main()