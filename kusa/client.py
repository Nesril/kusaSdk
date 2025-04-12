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
import tensorflow as tf
import torch



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
        """
        Full secure preprocessing pipeline for tabular/text data.
        Steps: Clean → Tokenize → Reduce → Target Encode → Store.
        """

        if self.__raw_df is None:
            raise DatasetSDKException("Raw dataset not loaded. Fetch a batch first.")

        df = self.__raw_df.copy()
        config = self.__preprocessing_config
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
                self.ensure_nltk_tokenizer_resources()
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
        reduction = config.get("reduction", "none")

        if reduction == "tfidf":
            max_feats = config.get("tfidf_max_features", 500)
            vectorizer = TfidfVectorizer(max_features=max_feats)

            for col in text_cols:
                tfidf_matrix = vectorizer.fit_transform(df[col])
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f"{col}_tfidf_{w}" for w in vectorizer.get_feature_names_out()]
                )
                df = df.drop(columns=[col]).join(tfidf_df)

        elif reduction == "pca":
            n_components = config.get("n_components", 2)
            numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1)

            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_df)

            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(scaled)

            for i in range(n_components):
                df[f"pca_{i+1}"] = reduced[:, i]

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

        self.__processed = df
        print("✅ Preprocessing completed successfully.")

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

    def train(self,user_train_func,hyperparams: dict = None,target_column: str = None,task_type: str = "classification",framework: str = "sklearn"):
            """
            Train a model using the user's function, securely inside the SDK.
            Supports sklearn, tensorflow, and pytorch frameworks.
            """
            if self.__processed is None:
                raise DatasetSDKException("No processed data available. Run preprocessing first.")

            if not target_column or target_column not in self.__processed.columns:
                raise DatasetSDKException(f"Invalid or missing target_column: '{target_column}'.")

            # Separate features and labels
            X = self.__processed.drop(columns=[target_column])
            y = self.__processed[target_column]

            print("📊 Class counts:\n", y.value_counts())
            # Prepare train/val split
            stratify_y = y if task_type == "classification" else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y
            )

            self.__training_framework = framework
            self.__task_type = task_type

            model = None

            try:
                # Framework-aware logic
                if framework == "sklearn":
                      model = user_train_func(X_train, y_train, **(hyperparams or {}))

                elif framework == "tensorflow":
                    if not isinstance(X_train, tf.Tensor):
                        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
                        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
                        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

                    model = user_train_func(X_train, y_train, X_val, y_val, **(hyperparams or {}))

                elif framework == "pytorch":
                    def to_tensor(df):
                        if isinstance(df, pd.Series):
                            if df.dtype == "O":
                                df = df.astype("float32")
                            return torch.tensor(df.values.reshape(-1, 1), dtype=torch.float32)

                        # If it's a DataFrame
                        if df.dtypes.eq("O").any():
                            df = df.astype("float32")
                        return torch.tensor(df.values, dtype=torch.float32)


                    X_train_tensor = to_tensor(X_train)
                    y_train_tensor = to_tensor(y_train)
                    X_val_tensor = to_tensor(X_val)
                    y_val_tensor = to_tensor(y_val)

                    model = user_train_func(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, **(hyperparams or {}))

                else:
                    raise DatasetSDKException(f"Unsupported framework: '{framework}'")

            except Exception as e:
                raise DatasetSDKException(f"Model training failed: {str(e)}")

            # Save model and validation set
            self.__trained_model = model
            self.__X_val = X_val
            self.__y_val = y_val

            print("✅ Training complete.")
            return model

            
    def evaluate(self):
        if self.__trained_model is None or self.__X_val is None:
            raise DatasetSDKException("No trained model or validation data available.")

        try:
            if self.__training_framework == "sklearn":
                preds = self.__trained_model.predict(self.__X_val)

            elif self.__training_framework == "tensorflow":
                preds = (self.__trained_model.predict(self.__X_val) > 0.5).astype("int32").flatten()

            elif self.__training_framework == "pytorch":
                self.__trained_model.eval()
                with torch.no_grad():
                    inputs = torch.tensor(self.__X_val.values, dtype=torch.float32)
                    outputs = self.__trained_model(inputs)
                    preds = torch.argmax(outputs, dim=1).numpy()
            else:
                raise DatasetSDKException(f"Unsupported framework: {self.__training_framework}")

        except Exception as e:
            raise DatasetSDKException(f"Evaluation failed: {str(e)}")

        if self.__task_type == "classification":
            accuracy = accuracy_score(self.__y_val, preds)
            report = classification_report(self.__y_val, preds)
            return {
                "accuracy": accuracy,
                "report": report
            }

        elif self.__task_type == "regression":
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(self.__y_val, preds)
            r2 = r2_score(self.__y_val, preds)
            return {
                "mse": mse,
                "r2_score": r2
            }

        else:
            raise DatasetSDKException(f"Unsupported task_type: '{self.__task_type}'")


    def predict(self, input_data):
        print("__training_framework ",self.__training_framework)
        if self.__trained_model is None:
            raise DatasetSDKException("Model not trained or loaded.")

    
        try:
            if self.__training_framework == "sklearn":
                return self.__trained_model.predict(input_data)

            elif self.__training_framework == "tensorflow":
                return (self.__trained_model.predict(input_data) > 0.5).astype("int32").flatten()

            elif self.__training_framework == "pytorch":
                self.__trained_model.eval()
                with torch.no_grad():
                    inputs = torch.tensor(input_data.values, dtype=torch.float32)
                    outputs = self.__trained_model(inputs)
                    return torch.argmax(outputs, dim=1).numpy()

            else:
                raise DatasetSDKException(f"Unsupported framework: {self.__training_framework}")
        except Exception as e:
            raise DatasetSDKException(f"Prediction failed: {str(e)}")


    def save_model(self, filepath: str):
        if self.__trained_model is None:
            raise DatasetSDKException("No trained model to save.")

        if self.__training_framework == "sklearn":
            joblib.dump(self.__trained_model, filepath)

        elif self.__training_framework == "tensorflow":
            # Ensure valid extension
            if not filepath.endswith(".keras") and not filepath.endswith(".h5"):
                filepath += ".keras"
            self.__trained_model.save(filepath)

        elif self.__training_framework == "pytorch":
            torch.save(self.__trained_model.state_dict(), filepath)

        else:
            raise DatasetSDKException(f"Unsupported framework for saving: {self.__training_framework}")

        print(f"\n✅ Model saved to: {filepath}")


    def load_model(self, filepath: str, training_framework: str) -> None:
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file
            training_framework: Framework the model was trained with ("sklearn", "tensorflow", or "pytorch")
        
        Raises:
            DatasetSDKException: If loading fails or framework is unsupported
        """
        if not training_framework:
            raise DatasetSDKException("Training framework is required to load model")
        
        if not os.path.exists(filepath):
            raise DatasetSDKException(f"No model found at: {filepath}")
        
        if training_framework not in ["pytorch", "tensorflow", "sklearn"]:
            raise DatasetSDKException(f"Unsupported framework: {training_framework}. Must be 'pytorch', 'tensorflow', or 'sklearn'")
        
        self.__training_framework = training_framework
        
        try:
            if training_framework == "sklearn":
                self.__trained_model = joblib.load(filepath)
                
            elif training_framework == "tensorflow":
                self.__trained_model = tf.keras.models.load_model(filepath)
                
            elif training_framework == "pytorch":
                if self.__trained_model is None:
                    raise DatasetSDKException(
                        "For PyTorch, model architecture must be initialized before loading weights. "
                        "Call initialize_pytorch_model() first or provide model architecture."
                    )
                self.__trained_model.load_state_dict(torch.load(filepath))
                self.__trained_model.eval()
                
        except Exception as e:
            raise DatasetSDKException(f"Failed to load {training_framework} model: {str(e)}")