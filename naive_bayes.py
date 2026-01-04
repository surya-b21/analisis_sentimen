import csv
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.corpus import stopwords
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Sastrawi stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Get Indonesian stopwords
try:
    stop_words = set(stopwords.words('indonesian'))
except:
    # Fallback to stopwords from file
    stop_words = set()
    with open('stopwordbahasa.csv', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stop_words.add(word)

def extract_features(text):
    """Extract word features from text with stopword removal and stemming"""
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    words = [stemmer.stem(word) for word in words]
    
    # Return features as dictionary
    return {word: True for word in words if word}

def load_data(filepath):
    """Load data from CSV file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text_clean']
            sentiment = row['sentiment']
            features = extract_features(text)
            data.append((features, sentiment))
    return data

def main():
    print("Loading training data...")
    train_data = load_data('train.csv')
    print(f"Training samples: {len(train_data)}")
    
    print("\nLoading test data...")
    test_data = load_data('test.csv')
    print(f"Test samples: {len(test_data)}")
    
    # Check class distribution
    train_labels = [label for _, label in train_data]
    test_labels = [label for _, label in test_data]
    print(f"\nTrain distribution: {Counter(train_labels)}")
    print(f"Test distribution: {Counter(test_labels)}")
    
    print("\nTraining Naive Bayes classifier...")
    classifier = NaiveBayesClassifier.train(train_data)
    
    print("\nEvaluating on test data...")
    test_accuracy = accuracy(classifier, test_data)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed evaluation
    predictions = []
    actuals = []
    for features, label in test_data:
        pred = classifier.classify(features)
        predictions.append(pred)
        actuals.append(label)
    
    # Calculate per-class metrics
    from collections import defaultdict
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    
    for actual, pred in zip(actuals, predictions):
        for sentiment_class in ['positive', 'negative', 'neutral']:
            if actual == sentiment_class and pred == sentiment_class:
                class_stats[sentiment_class]['tp'] += 1
            elif actual != sentiment_class and pred == sentiment_class:
                class_stats[sentiment_class]['fp'] += 1
            elif actual == sentiment_class and pred != sentiment_class:
                class_stats[sentiment_class]['fn'] += 1
            else:
                class_stats[sentiment_class]['tn'] += 1
    
    print("\nPer-class metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    
    for sentiment_class in ['positive', 'negative', 'neutral']:
        tp = class_stats[sentiment_class]['tp']
        fp = class_stats[sentiment_class]['fp']
        fn = class_stats[sentiment_class]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{sentiment_class:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    # Show most informative features
    print("\nMost Informative Features:")
    classifier.show_most_informative_features(20)
    
    # Example predictions
    print("\n\nExample predictions:")
    test_texts = []
    with open('test.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 5:
                test_texts.append((row['text_clean'], row['sentiment']))
    
    for text, actual in test_texts:
        features = extract_features(text)
        pred = classifier.classify(features)
        print(f"\nText: {text[:80]}...")
        print(f"Actual: {actual}, Predicted: {pred}")
    
    # Export results to JSON
    print("\nExporting results to JSON...")
    
    # Calculate confusion matrix manually
    confusion_matrix = {}
    for actual_class in ['negative', 'neutral', 'positive']:
        confusion_matrix[actual_class] = {'negative': 0, 'neutral': 0, 'positive': 0}
    
    for actual, pred in zip(actuals, predictions):
        confusion_matrix[actual][pred] += 1
    
    # Prepare example predictions
    example_predictions = []
    with open('test.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 10:
                text = row['text_clean']
                actual = row['sentiment']
                features = extract_features(text)
                pred = classifier.classify(features)
                example_predictions.append({
                    'text': text,
                    'actual': actual,
                    'predicted': pred,
                    'correct': actual == pred
                })
    
    # Get most informative features
    most_informative = []
    for i, (label, feat) in enumerate(classifier.most_informative_features(20)):
        most_informative.append({
            'feature': feat,
            'label': label
        })
    
    results_json = {
        'model_info': {
            'model_name': 'Naive Bayes',
            'model_type': 'NLTK NaiveBayesClassifier',
            'features': 'Bag of Words with Stopword Removal and Stemming',
            'stemmer': 'Sastrawi',
            'timestamp': datetime.now().isoformat()
        },
        'dataset_info': {
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'train_distribution': dict(Counter(train_labels)),
            'test_distribution': dict(Counter(test_labels))
        },
        'overall_metrics': {
            'accuracy': float(test_accuracy)
        },
        'per_class_metrics': {
            sentiment_class: {
                'precision': float(class_stats[sentiment_class]['tp'] / (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fp']) if (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fp']) > 0 else 0),
                'recall': float(class_stats[sentiment_class]['tp'] / (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn']) if (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn']) > 0 else 0),
                'f1_score': float(2 * (class_stats[sentiment_class]['tp'] / (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fp']) if (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fp']) > 0 else 0) * (class_stats[sentiment_class]['tp'] / (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn']) if (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn']) > 0 else 0) / ((class_stats[sentiment_class]['tp'] / (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fp']) if (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fp']) > 0 else 0) + (class_stats[sentiment_class]['tp'] / (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn']) if (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn']) > 0 else 0)) if ((class_stats[sentiment_class]['tp'] / (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fp']) if (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fp']) > 0 else 0) + (class_stats[sentiment_class]['tp'] / (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn']) if (class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn']) > 0 else 0)) > 0 else 0),
                'support': int(class_stats[sentiment_class]['tp'] + class_stats[sentiment_class]['fn'])
            }
            for sentiment_class in ['negative', 'neutral', 'positive']
        },
        'confusion_matrix': {
            'matrix': [[confusion_matrix['negative']['negative'], confusion_matrix['negative']['neutral'], confusion_matrix['negative']['positive']],
                      [confusion_matrix['neutral']['negative'], confusion_matrix['neutral']['neutral'], confusion_matrix['neutral']['positive']],
                      [confusion_matrix['positive']['negative'], confusion_matrix['positive']['neutral'], confusion_matrix['positive']['positive']]],
            'labels': ['negative', 'neutral', 'positive']
        },
        'most_informative_features': most_informative,
        'example_predictions': example_predictions
    }
    
    with open('naive_bayes_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print("Results saved to naive_bayes_results.json")

if __name__ == "__main__":
    main()
