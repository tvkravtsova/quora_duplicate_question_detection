# Quora Duplicate Question Detection

## Project Overview

This project addresses the challenge of identifying whether two Quora questions are semantically equivalent—a binary classification task crucial for improving search relevance and user experience on Q&A platforms.

We compare both traditional machine learning and state-of-the-art transformer-based NLP approaches to highlight the strengths and trade-offs of each modeling paradigm.

*  **Modeling Approaches:**
    - **Logistic Regression** using TF-IDF features
    - **XGBoost** trained on hand-engineered text similarity features
    - **Stacked Ensemble** combining Logistic Regression and XGBoost
    - **BERT** (fine-tuned transformer model) for sequence pair classification

*  **Evaluation Metrics:**
    - **Primary:** Log Loss
    - **Secondary:** ROC AUC, F1 Score

*  **Interpretation Strategy:** The focus of interpretation is comparing model performance to understand the trade-offs between simplicity, interpretability, and predictive power

## Project Structure

```
quora_duplicate_question_detection/
├── data/
│   ├── quora_question_pairs_train.csv.zip   # Training dataset
│   └── quora_question_pairs_test.csv.zip    # Test dataset
├── models/
│   ├── log_reg.joblib                      # Logistic Regression model
│   ├── xgb_model.joblib                     # XGBoost model
│   ├── stacking_clf.joblib                  # Stacked ensemble model
│   └── bert-duplicate-classifier.zip        # Download separately – fine-tuned BERT model (see README) 
├── notebooks/
│   ├── Quora_question_pairs_EDA.ipynb       # Exploratory Data Analysis
│   └── Quora_question_pairs_main.ipynb      # Main notebook for preprocessing, modeling, and evaluation
├── src/
│   └── quora_questions_preprocessing.py     # Script for data preprocessing & feature engineering
├── requirements.txt                         # Project dependencies
└── README.md                                # Project overview, setup, and usage
```

## Data

*   **Source:** [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data) from Kaggle.
*   **Content:** Pairs of questions with IDs and a binary label indicating if they are duplicates.
*   **Columns:**
    - `qid1`, `qid2`: Unique question IDs
    - `question1`, `question2`: Text of each question
    - `is_duplicate`: Target label (1 = duplicate, 0 = not duplicate)

## Feature Engineering & Preprocessing

Key steps performed (details in `notebooks/Quora_question_pairs_main.ipynb` and `src/quora_questions_preprocessing.py`):
*  **Token-based Features:** Extracted features such as word counts, unique word counts, number of common words between question pairs, Jaccard similarity, and unique word ratios.
*  **Text Cleaning:** Applied standard text normalization techniques — lowercasing, punctuation removal, stopword filtering, and lemmatization — to prepare input for TF-IDF vectorization.
*  **TF-IDF Vectorization & Cosine Similarity:** Computed TF-IDF vectors for both questions and derived cosine similarity as a numerical feature representing semantic closeness.
*  **Scaling:** Applied StandardScaler for numeric features.
*  **Final Feature Matrix (Traditional ML):** For classical models (Logistic Regression, XGBoost, Stacked Ensemble), all engineered features—including token-based metrics, TF-IDF vectors, cosine similarity, and scaled numerics—are merged into a single matrix.
*  **BERT Input Preparation:** For transformer-based classification, raw question pairs were tokenized using a pre-trained **BERT** tokenizer. This included automatic truncation, padding, and generation of attention masks and token type IDs, as required for sequence-pair classification tasks.

## Modeling & Evaluation

*   **Models Trained:**  **Logistic Regression**, **XGBoost**, **Stacked Ensemble**, and fine-tuned **BERT**.
*   **Evaluation:** Evaluated using Log Loss (primary), with ROC AUC and F1-score as secondary metrics.

## Results Summary

Below are key Log Loss and ROC AUC scores. For detailed metrics and model comparisons, see `notebooks/Quora_question_pairs_main.ipynb`.

| Model                        | Test Log Loss  | Test AUROC    | Key Notes                                   |
|:-----------------------------|:---------------|:--------------|:--------------------------------------------|
| **Logistic Regression**      | 0.4438         | 0.8612        | Fast, interpretable baseline                |
| **XGBoost**                  | 0.4680         | 0.8423        | Limited tuning applied                      |
| **Stacked Ensemble**         | 0.4322         | 0.8692        | Combines LR and XGB                         |
| **BERT (fine-tuned)**        | **0.3141**     | **0.9332**    | Best performance, captures semantic meaning |

## How to Use Model

### **BERT (fine-tuned) Model Download**
The best performing model is saved in `models/bert-duplicate-classifier.zip`.  Since the file is too large for GitHub, you can download it from Google Drive: [download the model](https://drive.google.com/uc?export=download&id=1crHDbGIzca4zfQtcvbOKTwUL2POVevPA)

After downloading, simply extract the `.zip` file into the `models/` directory.

Example usage for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('models/bert-duplicate-classifier')
model = AutoModelForSequenceClassification.from_pretrained('models/bert-duplicate-classifier')

q1 = "How can I increase my productivity?"
q2 = "What are some ways to be more productive?"
inputs = tokenizer(q1, q2, return_tensors='pt', truncation=True, padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    print('Duplicate' if pred == 1 else 'Not duplicate')
```

### **Classical Models** (**Logistic Regression**, **XGBoost**, **Stacked Ensemble**)

Classical models are saved in the `models/` directory as `.joblib` files. Example usage:

```python
import joblib
# Load the model (choose one: log_reg.joblib, xgb_model.joblib, stacking_clf.joblib)
model = joblib.load('models/log_reg.joblib')
# X_new should be preprocessed in the same way as during training
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

Refer to the main notebook for full preprocessing and feature engineering steps for classical models.

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/tvkravtsova/quora_duplicate_question_detection.git
    cd quora_duplicate_question_detection
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch Jupyter:** `jupyter notebook` and navigate to the `notebooks` directory.

## Next steps and potential improvements

*   Use Captum to visualize which input tokens contribute most to BERT's predictions.
*   Build a web-based demo or API to test question pairs interactively
*   Extend the task to multilingual duplicate detection using models like xlm-roberta

## Author

*   Tetiana Kravtsova
*   LinkedIn: https://www.linkedin.com/in/tetianakravtsova/
---
