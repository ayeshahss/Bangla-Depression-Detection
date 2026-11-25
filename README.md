# Depression Detection from Bangla Social Media Text

### Machine Learning, Deep Learning, Transformer Models, and Explainable AI

BSc thesis project, Department of Computer Science and Engineering, Metropolitan University, Sylhet.

> **Disclaimer:** This is an academic research project. The models classify text patterns, not people, and their output is **not** a clinical diagnosis of depression.

---

## 📌 Overview

This project compares how well three families of models detect depression-related language in Bangla social media posts under identical conditions: eight classical machine learning classifiers, two recurrent neural networks (LSTM and BiLSTM), and a fine-tuned Bangla pretrained transformer. SHAP and LIME are used to examine which words drive the predictions of the strongest classical model, and McNemar's test is used to check whether the difference between the two best models is statistically significant.

## 🎯 Research Objectives

- Compare eight classical classifiers on TF-IDF features, with 5-fold cross-validation.
- Evaluate LSTM and BiLSTM against the classical models.
- Fine-tune BanglaBERT and test whether its gap to the best competitor is significant (McNemar's test).
- Apply SHAP and LIME to the best classical model (Logistic Regression).
- Describe the sentiment profile of the dataset as a supporting, descriptive analysis.

## 📊 Dataset

- **Dataset:** Bengali Social Media Depressive Dataset (BSMDD), Chowdhury et al., Mendeley Data, doi: 10.17632/pdtyb7df6z.1
- Binary labels: 0 = Non-Depressed, 1 = Depressed

| Stage | Records |
|---|---:|
| Original | 21,910 |
| After removing 6 duplicates | 21,904 |
| After removing 2 empty posts (final) | 21,902 |
| Depressed / Non-Depressed | 10,957 / 10,945 |
| Train / Test (stratified 80:20) | 17,521 / 4,381 |

**Important:** the posts were originally written in English and then translated into Bangla. The dataset therefore does not represent naturally written Bangla social media text (no code-mixing, Romanised Bangla, or informal spelling). It also has no user, platform, or timestamp metadata, and no annotation protocol was published.

## ⚙️ Methodology

**1. Preprocessing**
- Unicode NFC normalization
- Removal of URLs, mentions, and hashtags
- Character filter keeping only Bangla script (plus ZWNJ/ZWJ, danda, digits, basic punctuation)
- Duplicate and empty-post removal
- No stopword removal and no stemming

**2. Feature representation** (all fitted on training data only)
- Classical models: TF-IDF unigrams + bigrams, whitespace tokenization, max 8,000 features, min document frequency 2 (KNN uses TruncatedSVD, 200 components)
- LSTM / BiLSTM: 15,000-word vocabulary, sequences of 120 tokens, learned embeddings
- BanglaBERT: its own WordPiece tokenizer, max 128 tokens

**3. Models**
- **Classical ML (8):** Logistic Regression, Naive Bayes, Linear SVM, XGBoost, Random Forest, AdaBoost, Decision Tree, KNN
- **Deep learning (2):** LSTM, BiLSTM (early stopping on validation loss, best weights restored)
- **Transformer (1):** `sagorsarker/bangla-bert-base`, fine-tuned (lr 2e-5, batch 16, up to 4 epochs, best epoch by validation loss)

**4. Evaluation:** accuracy, weighted precision/recall/F1, ROC-AUC, precision-recall curves, confusion matrices, McNemar's test, error analysis.

**5. Explainability:** SHAP (100 test instances) and LIME (5 test instances) on Logistic Regression.

**6. Sentiment analysis (descriptive only):** zero-shot `cardiffnlp/twitter-xlm-roberta-base-sentiment`. Sentiment labels were not used as input to any classifier.

## 📈 Results (held-out test set, n = 4,381)

| Model | Type | Accuracy | F1 (weighted) | AUC |
|---|---|---:|---:|---:|
| BanglaBERT | Transformer | 0.8909 | 0.8909 | 0.9583 |
| BiLSTM | DL | 0.8879 | 0.8876 | 0.9604 |
| Logistic Regression | ML | 0.8813 | 0.8813 | 0.9441 |
| Naive Bayes | ML | 0.8731 | 0.8730 | 0.9390 |
| SVM (LinearSVC) | ML | 0.8719 | 0.8719 | 0.9373 |
| XGBoost | ML | 0.8665 | 0.8665 | 0.9376 |
| LSTM | DL | 0.8541 | 0.8541 | 0.9066 |
| Random Forest | ML | 0.8521 | 0.8521 | 0.9249 |
| AdaBoost | ML | 0.8213 | 0.8211 | 0.9000 |
| Decision Tree | ML | 0.7574 | 0.7574 | 0.7517 |
| KNN (TF-IDF + SVD) | ML | 0.6469 | 0.6111 | 0.8189 |

### Key findings

- BanglaBERT had the highest accuracy (0.8909), but **McNemar's test showed no significant difference from BiLSTM** (χ² = 0.2981, p = 0.5851), so the two are statistically tied.
- Logistic Regression was the best classical model, within about one percentage point of BanglaBERT, and trained in 0.23 s versus 698.42 s for BanglaBERT (GPU vs CPU, so times are only indicative).
- LSTM (0.8541) scored below four classical models.
- The strongest model still misclassified 478 of 4,381 test posts (10.9%).
- Negative sentiment was more common in Depressed posts (19.33% vs 4.95%), but Neutral was the largest category in both classes, so sentiment alone does not separate them.
- SHAP/LIME showed Logistic Regression relied on depression-related words (e.g. আত্মহত্যা, বিষণ্ণতা, থেরাপি) but also on topic words such as স্কুল and কিশোর, suggesting part of the accuracy may reflect topic differences between the classes.

## 🛠️ Technologies

Python, Google Colab (T4 GPU), pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, PyTorch, Hugging Face Transformers, statsmodels, SHAP, LIME, Matplotlib, Streamlit (demo app). A fixed random seed of 42 was used throughout.

## 📁 Repository Contents

- `Bangla_Depression_Detection_Thesis_.ipynb`: main research notebook
- `README.md`: project documentation

## ⚠️ Limitations

- Translated (not native) Bangla text, from a single dataset; nearly perfectly balanced, unlike real social media streams.
- Labels used as provided; annotation process not documented.
- Neural models evaluated on a single 80:20 split; no systematic hyperparameter search.
- BanglaBERT limited to 128 subword tokens, truncating 49.2% of posts.
- `sagorsarker/bangla-bert-base` is a different model from the ELECTRA-based BanglaBERT of Bhattacharjee et al. (2022).
- SHAP/LIME explain Logistic Regression only, on small samples; they do not describe BiLSTM or BanglaBERT.
- Sentiment model was not fine-tuned on Bengali and was not validated against human annotation.
- Results should not be interpreted as clinical diagnoses.

## 👩‍💻 Authors

**Aysha Siddika** and **Abida Jin-nurine Jemi**
Supervisor: Mr. Muhammad Muzammil, Lecturer, Department of CSE, Metropolitan University, Sylhet
GitHub: https://github.com/ayeshahss

---

*Academic research project for research purposes only. Not a substitute for professional mental health assessment.*
