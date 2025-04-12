import os
import json
import base64
import pandas as pd
import numpy as np

from io import StringIO
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

from kusa.config import Config
from kusa.utils import make_request
from kusa.exceptions import DatasetSDKException

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import spacy


class SecureDatasetClient:
    def __init__(self, public_id=None, secret_key=None, encryption_key=None):
        self.public_id = public_id or os.getenv("PUBLIC_ID")
        self.secret_key = secret_key or os.getenv("SECRET_KEY")
        self.encryption_key = encryption_key or Config.get_encryption_key()
        self.base_url = Config.get_base_url()

        self.__raw_df = None      # Private memory store for raw data
        self.__processed = None   # Private memory for preprocessed
        self.__metadata = {}

        self._validate_keys()
        self.headers = self._build_headers()
        
        self.__trained_model = None
        self.__X_val = None
        self.__y_val = None

    def _validate_keys(self):
        if not self.public_id or not self.secret_key:
            raise DatasetSDKException("Missing PUBLIC_ID or SECRET_KEY.")
        if not self.encryption_key or len(self.encryption_key.encode()) != 32:
            raise DatasetSDKException("ENCRYPTION_KEY must be 32 bytes (AES-256).")

    def _build_headers(self):
        return {
            "Authorization": f"key {self.secret_key}"
        }

    def initialize(self):
        """Initializes metadata and preview without exposing raw data."""
        url = f"{self.base_url}/initialize/{self.public_id}"
        data = make_request(url, headers=self.headers)

        self.__metadata = {
            "totalRows": data.get("totalRows", 0),
            "columns": data.get("columns", []),
            "first10Rows":pd.read_csv(StringIO(data["first10Rows"]))
        }

        preview_df = pd.read_csv(StringIO(data["first10Rows"]))
        print("Sdk initialized successfuly !! ")
        return {
            "preview": preview_df.head(10),
            "metadata": self.__metadata
        }

    def fetch_and_decrypt_batch(self, batch_size=500, batch_number=0):
        """Fetches and securely decrypts data internally only."""

        if not self.__metadata:
          raise DatasetSDKException("Initialization is required")
          return
      
        url = f"{self.base_url}/get/{self.public_id}/encryptbatch"
        params = {
            "batchSize": batch_size,
            "batchNumber": batch_number
        }

        data = make_request(url, headers=self.headers, params=params)

        encrypted_data = base64.b64decode(data["batchData"])
        encrypted_key = base64.b64decode(data["encryptedKey"])

        symmetric_key = self._decrypt_data(encrypted_key, self.encryption_key.encode()).decode()
        raw_csv = self._decrypt_data(encrypted_data, symmetric_key.encode()).decode()

        df = pd.read_csv(StringIO(raw_csv))
        self.__raw_df = df.copy()  # Raw data never exposed
        return True  # Indicate internal load complete

    def _decrypt_data(self, encrypted_bytes: bytes, key: bytes) -> bytes:
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()
    
    # run preprocessing ( phase 2)
    # sdk.configure_preprocessing({
    #     "reduction": "pca",
    #     "n_components": 3,
    #     "tokenizer": "nltk",
    #     "stopwords": True,
    # })
    # sdk.run_preprocessing()
    
    # ✅ This Now Supports:
    # Custom text preprocessing (tokenizer, stopwords, lemmatization, etc.)

    # TF-IDF and PCA feature reduction

    # Clean future support for numpy/tensors

    # Compatible with scikit-learn, TF, PyTorch, etc.



    def configure_preprocessing(self, config: dict):
        """
            Accepts user-defined preprocessing config.
            Validates and stores it internally.
        """
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


    def ensure_nltk_tokenizer_resources(self):
        """Securely verify and download required NLTK resources"""
        try:
            # List of required NLTK resources
            resources = ['punkt', 'punkt_tab', 'stopwords']
            for res in resources:
                try:
                    # Check if the resource is already available
                    nltk.data.find(f'tokenizers/{res}' if res.startswith('punkt') else f'corpora/{res}')
                except LookupError:
                    # Download the resource if not found
                    nltk.download(res, quiet=True)
        except Exception as e:
            self._emergency_cleanup()
            raise DatasetSDKException(f"Resource verification failed: {str(e)}")

    def run_preprocessing(self):
        """Internal pipeline: clean → tokenize → reduce → store"""

        if self.__raw_df is None:
            raise DatasetSDKException("Raw dataset not loaded. Fetch a batch first.")

        df = self.__raw_df.copy()
        target_column = self.__preprocessing_config.get("target_column", None)

        if target_column and target_column not in df.columns:
            raise DatasetSDKException(f"Target column '{target_column}' not found in data.")

        # Store label column separately
        if target_column:
            y = df[target_column]
            df = df.drop(columns=[target_column])
        else:
            y = None


        # Get all text columns (typically 'object' dtype)
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols:
            df[col] = df[col].astype(str)

            # Step 1: Lowercase
            if self.__preprocessing_config.get("lowercase", True):
                df[col] = df[col].str.lower()

            # Step 2: Remove punctuation
            if self.__preprocessing_config.get("remove_punctuation", True):
                df[col] = df[col].str.replace(r"[^\w\s]", "", regex=True)

            # Step 3: Tokenization
            tokenizer = self.__preprocessing_config.get("tokenizer", "nltk")
            if tokenizer == "nltk":
                self.ensure_nltk_tokenizer_resources()
                df[col] = df[col].apply(word_tokenize)
            elif tokenizer == "spacy":
                nlp = spacy.load("en_core_web_sm")
                df[col] = df[col].apply(lambda x: [token.text for token in nlp(x)])
            elif tokenizer == "split":
                df[col] = df[col].apply(lambda x: x.split())
            elif tokenizer == "none":
                df[col] = df[col].apply(lambda x: [x])

            # Step 4: Stopword removal
            if self.__preprocessing_config.get("stopwords", True):
                stop_words = set(stopwords.words("english"))
                df[col] = df[col].apply(lambda tokens: [w for w in tokens if w.lower() not in stop_words])

            # Step 5: Lemmatization (if enabled and using spaCy)
            if self.__preprocessing_config.get("lemmatize", False) and tokenizer == "spacy":
                df[col] = df[col].apply(lambda tokens: [token.lemma_ for token in nlp(" ".join(tokens))])

            # Step 6: Rejoin tokens back into string
            df[col] = df[col].apply(lambda tokens: " ".join(tokens))

        # ======================
        # Feature Reduction
        # ======================

        reduction_type = self.__preprocessing_config.get("reduction", "none")

        if reduction_type == "tfidf":
            max_feats = self.__preprocessing_config.get("tfidf_max_features", 500)
            vectorizer = TfidfVectorizer(max_features=max_feats)
            for col in text_cols:
                tfidf_matrix = vectorizer.fit_transform(df[col])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[
                    f"{col}_tfidf_{feat}" for feat in vectorizer.get_feature_names_out()
                ])
                df = df.drop(columns=[col]).join(tfidf_df)

        elif reduction_type == "pca":
            n_components = self.__preprocessing_config.get("n_components", 2)
            numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1)

            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_df)

            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(scaled)

            for i in range(n_components):
                df[f"pca_{i+1}"] = reduced[:, i]

        if y is not None:
            df[target_column] = y.reset_index(drop=True)

        # Output format (default is pandas for now)
        output_format = self.__preprocessing_config.get("output_format", "pandas")
        if output_format != "pandas":
            raise DatasetSDKException(f"Output format '{output_format}' is not supported yet.")

        # Store processed data in memory
        self.__processed = df

    
    # 🧠 Phase 3: Model Training Framework
    # def my_model(X, y, **params):
    #     # Train a model here
    #     model = SomeModel(**params)
    #     model.fit(X, y)
    #     return model

    # sdk.train(my_model, {
    #     "learning_rate": 0.01,
    #     "epochs": 10
    # })

    # Think of your SDK like a gym.
    # You bring your own coach (train_model), and the gym just gives them the tools, machines, and environment.

    def train(self, user_train_func, hyperparams: dict = None, target_column: str = None):
        """User passes in their training function; SDK internally calls it with processed data."""
        if self.__processed is None:
            raise DatasetSDKException("No processed data available. Run preprocessing first.")

        print(target_column not in self.__processed.columns,target_column,self.__processed.columns)
        if target_column is None or target_column not in self.__processed.columns:
            raise DatasetSDKException(f"Invalid or missing target_column: '{target_column}'.")

        X = self.__processed.drop(columns=[target_column])
        y = self.__processed[target_column]

        print("Class counts:\n", y.value_counts())
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        
        print("X_train ",X_train)
        # Call the user-defined training function
        model = user_train_func(X_train, y_train, **(hyperparams or {}))

        self.__trained_model = model
        self.__X_val = X_val
        self.__y_val = y_val

        return model
            
    def evaluate(self):
        if self.__trained_model is None or self.__X_val is None:
            raise DatasetSDKException("No trained model or validation data available.")

        preds = self.__trained_model.predict(self.__X_val)
        accuracy = accuracy_score(self.__y_val, preds)
        report = classification_report(self.__y_val, preds)

        return {
            "accuracy": accuracy,
            "report": report
        }

    def predict(self, input_data):
        if self.__trained_model is None:
            raise DatasetSDKException("Model not trained or loaded.")

        return self.__trained_model.predict(input_data)

    def save_model(self, filepath: str):
        if self.__trained_model is None:
            raise DatasetSDKException("No trained model to save.")

        joblib.dump(self.__trained_model, filepath)
        print("\n🚀 Model training and evaluation completed securely.")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise DatasetSDKException(f"No model found at: {filepath}")

        self.__trained_model = joblib.load(filepath)
