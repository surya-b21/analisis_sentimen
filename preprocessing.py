import pandas as pd
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load slang dictionary once at startup
def load_slang_dict():
    """Load slang dictionary from CSV file"""
    try:
        slang_df = pd.read_csv('kamus_slang.csv', encoding='utf-8')
        slang_df.columns = slang_df.columns.str.strip()
        # Create dictionary: slang -> formal
        slang_dict = pd.Series(slang_df.formal.values, index=slang_df.slang).to_dict()
        print(f"✓ Loaded {len(slang_dict)} slang entries")
        return slang_dict
    except FileNotFoundError:
        print("✗ kamus_slang.csv not found. Skipping slang normalization.")
        return {}

# Load dictionary once
SLANG_DICT = load_slang_dict()

# read csv file
def read_csv():
    df = pd.read_csv('dataset.csv', encoding='utf-8', on_bad_lines='skip', engine='python')
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    data = []

    for _, row in df.iterrows():
        text = row['text_display']
        
        # Handle NaN values - skip rows with missing text
        if pd.isna(text):
            continue
        
        # Convert to string if it's not already
        text = str(text).strip()
        
        # Skip empty strings
        if not text:
            continue
        
        # Step 1: Remove noise
        text = remove_noise(text)

        # skip if text is empty after noise removal
        if not text:
            continue
        
        # Step 2: Normalize slang words
        if SLANG_DICT:
            words = text.split()
            normalized_words = [SLANG_DICT.get(word, word) for word in words]
            text = ' '.join(normalized_words)

        data.append(text)

    return data

def remove_noise(text):
    # Remove unwanted characters except ? and !
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: Remove newlines and replace with space
    text = re.sub(r'\n|\r\n|\r', ' ', text)
    
    # Step 2: Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Step 3: Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Step 4: Clean up extra spaces again after removing special chars
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def importLexiconDict():
    """Load Indonesian sentiment lexicons"""
    try:
        # Skip header row (use skiprows=1 with names parameter)
        negative_df = pd.read_csv('lexicon_id_negative.tsv', encoding='utf-8', sep='\t', skiprows=1, names=['word', 'weight'])
        positive_df = pd.read_csv('lexicon_id_positive.tsv', encoding='utf-8', sep='\t', skiprows=1, names=['word', 'weight'])

        lexiconIdDict = {}
        
        # Process negative words (ensure negative values)
        for _, row in negative_df.iterrows():
            word = str(row['word']).strip().lower()
            try:
                weight = float(row['weight'])
                lexiconIdDict[word] = -abs(weight)  # Ensure negative
            except (ValueError, TypeError):
                continue
        
        # Process positive words (ensure positive values)
        for _, row in positive_df.iterrows():
            word = str(row['word']).strip().lower()
            try:
                weight = float(row['weight'])
                lexiconIdDict[word] = abs(weight)  # Ensure positive
            except (ValueError, TypeError):
                continue
        
        return lexiconIdDict
    except Exception as e:
        print(f"⚠️  Error loading lexicon: {e}")
        return {}

def main():
    print("Loading and processing data...")
    data = read_csv()
    
    lexicon_dict = importLexiconDict()
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(lexicon_dict)

    # export processed data to csv and add sentiment label, show compound score, positive, negative, neutral
    processed_data = []
    totalPositive = 0
    totalNegative = 0
    totalNeutral = 0
    for text in data:
        sentiment_scores = analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        if compound_score >= 0.05:
            sentiment = 'positive'
            totalPositive += 1
        elif compound_score <= -0.05:
            sentiment = 'negative'
            totalNegative += 1
        else:
            sentiment = 'neutral'
            totalNeutral += 1
        
        processed_data.append({'comment': text, 'compound': compound_score, 'positive': sentiment_scores['pos'], 'negative': sentiment_scores['neg'], 'neutral': sentiment_scores['neu'], 'label': sentiment})
    
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv('vader_result.csv', index=False, encoding='utf-8')
    print("Processed data saved to vader_result.csv")

if __name__ == "__main__":
    main()