# Text Relevance Prediction Using Siamese LSTM Networks

This project was completed as part of a Practical Deep Learning workshop at university. It focuses on predicting the relevance between a user’s search query and a Home Depot product description using Siamese LSTM networks at both character and word levels.

## 📚 Project Description

We worked on the [Home Depot product search relevance dataset](https://www.kaggle.com/c/home-depot-product-search-relevance), where the goal is to predict how relevant a given search query is to a particular product.

The task was divided into two main parts:
1. **Character-level LSTM Siamese model**
2. **Word-level LSTM Siamese model with embeddings**

We explored neural and non-neural approaches, compared models across several performance metrics, and used feature extraction to enhance traditional ML models.

---

## 🧠 Methods and Models

### 1. Character-level Processing
- **Preprocessing**: Converted input texts into character sequences.
- **Model**: Siamese LSTM architecture using character inputs.
- **Benchmark**: Naïve model using character count vectorizer and classical regressors.
- **Feature Extractor**: Used LSTM representations as input to ML models (e.g., XGBoost, Random Forest).

### 2. Word-level Processing
- **Preprocessing**: Used word and symbol tokens (e.g. "90°", "½", "#SC").
- **Embeddings**: Trained word2vec using Gensim.
- **Model**: Siamese LSTM using pretrained embeddings.
- **Feature Extractor**: Similar feature extraction process as in character-level.

---

## 📊 Results Summary

| Model Type                           | Runtime (sec) | Train RMSE | Val RMSE | Test RMSE | Train MAE | Val MAE | Test MAE |
|-------------------------------------|----------------|------------|----------|-----------|-----------|---------|----------|
| Naïve Benchmark (Char-Level)        | 0.93           | 0.5871     | 0.5840   | 0.5885    | 0.4734    | 0.4702  | 0.4736   |
| Siamese LSTM (Char-Level)           | 53.94          | 0.5255     | 0.5246   | 0.5283    | 0.4335    | 0.4325  | 0.4354   |
| Siamese LSTM (Word-Level)           | 65.00          | 0.4565     | 0.4834   | 0.5285    | 0.3698    | 0.3892  | 0.4262   |
| Char-Level Feature Extractor + KNN  | 0.11           | 0.0161     | 0.5503   | 0.5586    | 0.0008    | 0.4477  | 0.4546   |
| Char-Level Feature Extractor + XGB  | 1.16           | 0.5239     | 0.5275   | 0.5318    | 0.4292    | 0.4327  | 0.4354   |
| Word-Level Feature Extractor + KNN  | 0.14           | 0.0050     | 0.5114   | 0.5541    | 0.0001    | 0.4104  | 0.4436   |
| Word-Level Feature Extractor + XGB  | 2.11           | 0.4456     | 0.4910   | 0.5356    | 0.3584    | 0.3960  | 0.4305   |

---

## 📈 Visualizations

The notebook includes:
- Tokenization examples
- Training/validation loss plots for both character and word LSTM models
- Comparisons between benchmark and deep learning models

---

## 🛠 Tech Stack

- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Gensim (for Word2Vec)
- XGBoost / CatBoost / LightGBM / Random Forest
- Jupyter Notebook

