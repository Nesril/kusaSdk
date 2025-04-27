import pandas as pd
import numpy as np
import re
import nltk
import spacy
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from kusa.utils import ensure_nltk_tokenizer_resources
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class PreprocessingManager:
    def __init__(self):
        self.config = {}
        self.transformers = {}
        self.processed_df = None
        
    def configure(self, config):
        
        default_config = {
            "tokenizer": "nltk",             # or 'spacy', 'split', 'none'
            "stopwords": True,
            "lowercase": True,
            "remove_punctuation": True,
            "lemmatize": False,
            "reduction": "none",             # 'tfidf', 'pca', or 'none'
            "n_components": 2,               # For PCA
            "tfidf_max_features": 500,       # For TF-IDF
            "target_column": None,
            "output_format": "pandas"        # Can support tensor/numpy later
        }
        config = config or {}
        for key in config:
            if key not in default_config:
                print(f"⚠️ Unknown config key: '{key}' – ignoring.")
        # Merge with defaults
        self.__preprocessing_config = {**default_config, **config}

    def run(self, raw_df):
        if raw_df is None:
            raise DatasetSDKException("Raw dataset not loaded. Fetch a batch first.")

        df = raw_df.copy()
        config = self.config
        target_column = config.get("target_column")

        # Check if target column exists
        if target_column and target_column not in df.columns:
            raise DatasetSDKException(f"Target column '{target_column}' not found in dataset.")

        # Extract target column if exists
        y = None
        if target_column:
            y = df[target_column]
            df = df.drop(columns=[target_column])

        # Preprocessing on text columns
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols:
            df[col] = df[col].astype(str)

            # Lowercase
            if config.get("lowercase", True):
                df[col] = df[col].str.lower()

            # Remove punctuation
            if config.get("remove_punctuation", True):
                df[col] = df[col].str.replace(r"[^\w\s]", "", regex=True)

            # Tokenization
            tokenizer_type = config.get("tokenizer", "nltk")
            if tokenizer_type == "nltk":
                ensure_nltk_tokenizer_resources()
                df[col] = df[col].apply(word_tokenize)
            elif tokenizer_type == "spacy":
                nlp = spacy.load("en_core_web_sm")
                df[col] = df[col].apply(lambda x: [token.text for token in nlp(x)])
            elif tokenizer_type == "split":
                df[col] = df[col].apply(lambda x: x.split())
            elif tokenizer_type == "none":
                df[col] = df[col].apply(lambda x: [x])  # wrap as list

            # Stopword removal
            if config.get("stopwords", True):
                stop_words = set(stopwords.words("english"))
                df[col] = df[col].apply(lambda tokens: [w for w in tokens if w.lower() not in stop_words])

            # Lemmatization (optional, spaCy only)
            if config.get("lemmatize", False) and tokenizer_type == "spacy":
                df[col] = df[col].apply(lambda tokens: [token.lemma_ for token in nlp(" ".join(tokens))])

            # Rejoin tokens to string for vectorizers
            df[col] = df[col].apply(lambda tokens: " ".join(tokens))

        # ===============================
        # Feature Reduction (TF-IDF / PCA)
        # ===============================
        
        # "reduction": "pca" → it applies only to numeric columns.
        # "reduction": "tfidf" → it applies only to text columns.
        # "reduction": "tfidf_pca" → they need to handle both separately (TF-IDF for text, then PCA for numeric).
        
        reduction = config.get("reduction", "none")

        if reduction == "tfidf":
            self.transformers["tfidf_vectorizers"] = {}

            max_feats = config.get("tfidf_max_features", 500)
            vectorizer = TfidfVectorizer(max_features=max_feats)

            for col in text_cols:
                tfidf_matrix = vectorizer.fit_transform(df[col])
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f"{col}_tfidf_{w}" for w in vectorizer.get_feature_names_out()]
                )
                df = df.drop(columns=[col]).join(tfidf_df)
                self.transformers["tfidf_vectorizers"][col] = vectorizer # ✅ Save vectorizer
            

        elif reduction == "tfidf_pca":
            self.transformers["tfidf_vectorizers"] = {}
            tfidf_outputs = []
            tfidf_feature_names = []

            for col in text_cols:
                vectorizer = TfidfVectorizer(max_features=config.get("tfidf_max_features", 500))
                tfidf_matrix = vectorizer.fit_transform(df[col])
                
                self.transformers["tfidf_vectorizers"][col] = vectorizer
                tfidf_outputs.append(tfidf_matrix)
                tfidf_feature_names += [f"{col}_tfidf_{feat}" for feat in vectorizer.get_feature_names_out()]

            from scipy.sparse import hstack
            combined_tfidf = hstack(tfidf_outputs).toarray()

            # Apply PCA to combined TF-IDF
            n_components = config.get("n_components", 2)
            pca = PCA(n_components=n_components)
            reduced_tfidf = pca.fit_transform(combined_tfidf)

            self.transformers["pca_tfidf"] = {
                "pca": pca,
                "tfidf_feature_names": tfidf_feature_names,
                "pca_feature_names": [f"pca_tfidf_{i+1}" for i in range(n_components)]
            }

            # Numeric features (keep them)
            df_numeric = df.select_dtypes(include=[np.number]).reset_index(drop=True)
            df_reduced = pd.DataFrame(reduced_tfidf, columns=self.transformers["pca_tfidf"]["pca_feature_names"])
            df = pd.concat([df_numeric.reset_index(drop=True), df_reduced], axis=1)


        elif reduction == "pca":
            n_components = config.get("n_components", 2)

            # Keep only numeric columns for PCA
            numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1)

            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_df)

            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(scaled)

            # ✅ Drop ALL non-numeric columns (object, datetime, etc.)
            df = df.select_dtypes(include=[np.number])
            df = df.drop(columns=numeric_df.columns)  # Drop numeric columns used in PCA

            # ✅ Add PCA components
            pca_feature_names = [f"pca_{i+1}" for i in range(n_components)]
            print("pca_feature_names ",pca_feature_names)
            for i, name in enumerate(pca_feature_names):
                df[name] = reduced[:, i]

            # ✅ Save transformer info
            self.transformers["pca"] = {
                "scaler": scaler,
                "pca": pca,
                "trained_numeric_columns": numeric_df.columns.tolist(),
                "trained_pca_feature_names": pca_feature_names
            }
         
        # ===============================
        # Target Encoding (Auto / Custom)
        # ===============================
        if y is not None:
            encoding_mode = config.get("target_encoding", "auto")

            if encoding_mode == "auto":
                if y.dtype == object or y.dtype == "O":
                    unique_values = y.unique()
                    if len(unique_values) == 2:
                        mapping = {unique_values[0]: 0, unique_values[1]: 1}
                        y = y.map(mapping)
                        print(f"🔢 Auto target encoding applied: {mapping}")
                    else:
                        raise DatasetSDKException(f"Cannot auto-encode target with >2 classes: {unique_values}")
            elif isinstance(encoding_mode, dict):
                y = y.map(encoding_mode)
                print(f"🔢 Custom target encoding applied: {encoding_mode}")
            elif encoding_mode == "none":
                pass  # leave y unchanged
            else:
                raise DatasetSDKException(f"Invalid target_encoding value: {encoding_mode}")

            df[target_column] = y.reset_index(drop=True)

        # Final output format
        output_format = config.get("output_format", "pandas")
        if output_format != "pandas":
            raise DatasetSDKException(f"Output format '{output_format}' is not supported yet.")

        self.processed_df = df
        print("✅ Preprocessing completed successfully.")

    def get_processed_data(self):
        return self.processed_df

    def get_transformers(self):
        return self.transformers
