import os
import base64
import pandas as pd
from io import StringIO

from kusa.config import Config
from kusa.utils import make_request
from kusa.exceptions import DatasetSDKException

from kusa.preprocessing_manager import PreprocessingManager
from kusa.model_manager import ModelManager
from kusa.encryption_manager import EncryptionManager


class SecureDatasetClient:
    def __init__(self, public_id=None, secret_key=None, encryption_key=None):
        self.public_id = public_id or os.getenv("PUBLIC_ID")
        self.secret_key = secret_key or os.getenv("SECRET_KEY")
        self.encryption_key = encryption_key or Config.get_encryption_key()
        self.base_url = Config.get_base_url()

        self.__raw_df = None      # Private memory store for raw data
        self.__processed = None   # Private memory for preprocessed
        self.__metadata = {}
        self.__transformers={}

        self._validate_keys()
        self.headers = self._build_headers()
        
        self.__trained_model = None
        self.__X_val = None
        self.__y_val = None
        self.__input_feature_names = None
        
        self.preprocessing_manager = PreprocessingManager()
        self.model_manager = ModelManager()
        self.encryption_manager = EncryptionManager()
        
        def __getattribute__(self, name):
            if name in ['_SecureDatasetClient__encryption_key', 
                       '_SecureDatasetClient__secret_key',
                       'encryption_key', 'secret_key']:
                raise AttributeError("Access to sensitive attribute denied")
            return object.__getattribute__(self, name)

        

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
        """Fetches encrypted batch from API, decrypts, and loads into internal memory."""
        if not self.__metadata:
            raise DatasetSDKException("Initialization is required.")

        url = f"{self.base_url}/get/{self.public_id}/encryptbatch"
        params = {
            "batchSize": batch_size,
            "batchNumber": batch_number
        }

        data = make_request(url, headers=self.headers, params=params)

        # Decode base64 payloads
        encrypted_data = base64.b64decode(data["batchData"])
        encrypted_key = base64.b64decode(data["encryptedKey"])

        # Decrypt symmetric key
        decryptor =  self.encryption_manager
        symmetric_key = decryptor.decrypt(encrypted_key, self.encryption_key.encode()).decode()

        # Decrypt CSV data
        raw_csv = decryptor.decrypt(encrypted_data, symmetric_key.encode()).decode()

        # Parse decrypted CSV into DataFrame
        df = pd.read_csv(StringIO(raw_csv))
        self.__raw_df = df.copy()

        print("✅ Batch fetched and decrypted successfully.")
        return True

    
    def configure_preprocessing(self, config: dict):
        """
            Accepts user-defined preprocessing config.
            Validates and stores it internally.
        """
        self.preprocessing_manager.configure(config)

    def run_preprocessing(self):
        """
        Full secure preprocessing pipeline for tabular/text data.
        Steps: Clean → Tokenize → Reduce → Target Encode → Store.
        """
        self.preprocessing_manager.run(self.__raw_df)
        
    def train(self, user_train_func, hyperparams: dict = None, target_column: str = None, 
            task_type: str = "classification", framework: str = "sklearn"):
        """
        Train a model using the user's function, securely inside the SDK.
        Supports sklearn, tensorflow, and pytorch frameworks.
        
        Args:
            user_train_func: User-defined training function
            hyperparams: Dictionary of hyperparameters
            target_column: Name of the target column
            task_type: Type of ML task ('classification' or 'regression')
            framework: ML framework to use ('sklearn', 'tensorflow', or 'pytorch')
        """
        self.model_manager.train(
            processed_df=self.preprocessing_manager.get_processed_data(),
            user_train_func=user_train_func,
            hyperparams=hyperparams,
            target_column=target_column,
            task_type=task_type,
            framework=framework
        )
        self.__trained_model = self.model_manager.get_modal()
        self.__X_val = self.model_manager.get_x_val()
        self.__y_val = self.model_manager.get_y_val()
        

            
    def evaluate(self):
        return self.model_manager.evaluate()
       
    def predict(self, input_df):
        print("get_transformers ",self.preprocessing_manager.get_transformers())
        return self.model_manager.predict(input_df, self.preprocessing_manager.get_transformers())


    def save_model(self, filepath: str):
        self.model_manager.save(filepath)
        

    def load_model(self, filepath: str, training_framework: str) -> None:
        self.model_manager.load(filepath, training_framework)