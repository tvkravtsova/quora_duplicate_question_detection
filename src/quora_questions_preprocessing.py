import pandas as pd
import numpy as np

from nltk.tokenize import TweetTokenizer

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
from typing import Tuple, Optional

# Define a function to extract token-based features
def extract_token_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts token-based features from 'question1' and 'question2' using TweetTokenizer.
    Tokens include both alphabetic and numeric strings, and are lowercased.

    Parameters:
        df (pd.DataFrame): DataFrame with 'question1' and 'question2' columns.

    Returns:
        pd.DataFrame: DataFrame with added token-based features:
            - Clean tokens (q1_tokens, q2_tokens)
            - Word counts
            - Unique word counts
            - Common word counts
            - Word share
            - Unique word ratio
    """
    df = df.copy()

    tt = TweetTokenizer()

    # Tokenize and lowercase; keep only alphanumeric tokens
    df['q1_tokens'] = df['question1'].astype(str).apply(
        lambda x: [t.lower() for t in tt.tokenize(x) if t.isalnum()]
    )
    df['q2_tokens'] = df['question2'].astype(str).apply(
        lambda x: [t.lower() for t in tt.tokenize(x) if t.isalnum()]
    )

    # Calculate the number of tokens (words) in question1 and question2
    df['q1_words'] = df['q1_tokens'].apply(len)
    df['q2_words'] = df['q2_tokens'].apply(len)

    # Calculate the number of unique words in question1 and question2 by
    df['q1_unique_words'] = df['q1_tokens'].apply(lambda x: len(set(x)))
    df['q2_unique_words'] = df['q2_tokens'].apply(lambda x: len(set(x)))

    # Count common words between question1 and question2
    df['common_words'] = df.apply(
        lambda row: len(set(row['q1_tokens']) & set(row['q2_tokens'])),
        axis=1
    )
    # Calculate word share: Jaccard similarity
    df['word_share'] = df.apply(
        lambda row: (
            len(set(row['q1_tokens']) & set(row['q2_tokens'])) /
            len(set(row['q1_tokens']) | set(row['q2_tokens']))
            if len(set(row['q1_tokens']) | set(row['q2_tokens'])) > 0 else 0
        ),
        axis=1
    )
    # Ratio of smaller to larger unique word count
    df['unique_word_ratio'] = df[['q1_unique_words', 'q2_unique_words']].min(axis=1) / \
                              df[['q1_unique_words', 'q2_unique_words']].max(axis=1).replace(0, 1)

    return df

def preprocess_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    max_features: int = 5000
) -> Tuple[pd.DataFrame, pd.DataFrame, csr_matrix, csr_matrix, Optional[np.ndarray]]:
    """
    Full preprocessing pipeline that assumes numerical features are already extracted.
    Performs text cleaning, TF-IDF vectorization, and constructs feature matrices.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset with 'question1', 'question2' and token-based numeric features.
    df_test : pd.DataFrame
        Test dataset with 'question1', 'question2' and token-based numeric features.
    max_features : int
        Maximum number of features for the TF-IDF vectorizer.

    Returns
    -------
    df_train : pd.DataFrame
        Training dataframe with additional 'question*_clean' columns.
    df_test : pd.DataFrame
        Test dataframe with additional 'question*_clean' columns.
    X_train : csr_matrix
        Combined sparse matrix for train data (TF-IDF + numerical features).
    X_test : csr_matrix
        Combined sparse matrix for test data (TF-IDF + numerical features).
    y_train : Optional[np.ndarray]
        Array of labels if present in df_train, otherwise None.
    """
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'nor', "n't"}
    lemmatizer = WordNetLemmatizer()

    def clean_text(text: str) -> str:
        text = text.lower() # Convert to lowercase
        text = re.sub(r"[^\w\s']", ' ', text) # Remove punctuation (except apostrophes)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    # Copy data
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Clean text
    for df in [df_train, df_test]:
        df['question1_clean'] = df['question1'].astype(str).apply(clean_text)
        df['question2_clean'] = df['question2'].astype(str).apply(clean_text)

    # Fit TF-IDF on all questions
    all_questions = pd.concat([
        df_train['question1_clean'], df_train['question2_clean'],
        df_test['question1_clean'], df_test['question2_clean']
    ])
    tfidf = TfidfVectorizer(max_features=max_features)
    tfidf.fit(all_questions)

    def build_matrix(df: pd.DataFrame) -> Tuple[csr_matrix, csr_matrix]:
        q1_tfidf = tfidf.transform(df['question1_clean'])
        q2_tfidf = tfidf.transform(df['question2_clean'])

        cosine_sim = np.array([
            cosine_similarity(q1, q2)[0][0] for q1, q2 in zip(q1_tfidf, q2_tfidf)
        ]).reshape(-1, 1)

        # Select already existing numeric features
        numeric_cols = [
            'q1_words', 'q2_words', 'q1_unique_words', 'q2_unique_words',
            'unique_word_ratio', 'common_words', 'word_share'
        ]
        # Scaling
        scaler = StandardScaler()
        numeric_values = df[numeric_cols].values  # Extract numeric features from DataFrame
        numeric_cols_scaled = scaler.fit_transform(numeric_values)  # Scale numeric features

        # Combine cosine similarity and scaled numeric features into a single numpy array
        num_feats = np.hstack([cosine_sim, numeric_cols_scaled])

        # Convert numeric features to a sparse matrix
        sparse_feats = csr_matrix(num_feats)

        # Combine TF-IDF vectors with numeric features into the final sparse feature matrix
        X = hstack([q1_tfidf, q2_tfidf, sparse_feats])

        return X, sparse_feats

    X_train, _ = build_matrix(df_train)
    X_test, _ = build_matrix(df_test)

    y_train = df_train['is_duplicate'].values if 'is_duplicate' in df_train else None

    return df_train, df_test, X_train, X_test, y_train
