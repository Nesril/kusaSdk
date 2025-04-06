# kusa/client.py

import pandas as pd
from io import StringIO
from .utils import make_request
from .exceptions import DatasetSDKException
import os
from dotenv import load_dotenv, find_dotenv
import json
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from pathlib import Path
from kusa.config import Config
import numpy as np
from pandas.api.types import is_dict_like


BASE_URL = Config.get_base_url()
ENCRYPTION_KEY = "R@BrG8XQh1A6d%%PZz5Uh0P$YeouD4Z*"

class DatasetClient:
    def __init__(self, public_id=None, secret_key=None, base_url=None):
        """
        Initializes the DatasetClient with authentication details.

        Args:
            public_id (str, optional): Public ID of the dataset. If not provided, fetched from environment.
            secret_key (str, optional): Secret key for authorization. If not provided, fetched from environment.
            base_url (str, optional): Base URL of the dataset server. If not provided, fetched from environment.
        """
        self.public_id = public_id 
        self.secret_key = secret_key
        self.base_url = BASE_URL
        self.encryption_key = ENCRYPTION_KEY
        self._data_fingerprint = None  # Lazy initialization
        self._last_decrypted = None    # For emergency cleanup
        
        print(BASE_URL," encryption_key ",ENCRYPTION_KEY)

        if not self.public_id or not self.secret_key:
            raise DatasetSDKException("PUBLIC_ID and SECRET_KEY must be set either as parameters or in the .env file.")
        if not self.encryption_key:
            raise DatasetSDKException("The SDK is not yet ready to be published")

        try:
            if len(self.encryption_key.encode('utf-8')) != 32:
                raise DatasetSDKException(f"ENCRYPTION_KEY must be 32 bytes long for AES-256.  {self.encryption_key}")
        except:
            raise DatasetSDKException(f"ENCRYPTION_KEY must be 32 bytes long for AES-256. {self.encryption_key}")
        # Debug statement (remove or comment out in production)

        self.headers = {
            "Authorization": f"key {self.secret_key}"
        }

    def initialize(self):
        """
        Initializes the dataset by fetching total rows and the first 10 rows.

        Returns:
            dict: Contains 'totalRows' and 'first10Rows' as a pandas DataFrame.
        """
        url = f"{self.base_url}/initialize/{self.public_id}"
        data = make_request(url, headers=self.headers)
        
        total_rows = data.get("totalRows")
        first10Rows_str = data.get("first10Rows")
        
        if not first10Rows_str:
            raise DatasetSDKException("Initialization failed: 'first10Rows' is missing or empty.")
        
        try:
            # Use StringIO to convert the CSV string into a file-like object
            first_10_rows = pd.read_csv(StringIO(first10Rows_str))
        except pd.errors.ParserError as e:
            raise DatasetSDKException(f"Failed to parse 'first10Rows' CSV data: {e}")
        
        if total_rows is None or first_10_rows.empty:
            raise DatasetSDKException("Initialization failed: Invalid response from server.")
        
        return {
            "totalRows": total_rows,
            "first10Rows": first_10_rows
        }

    def fetch_batch(self, batch_size, batch_number):
        """
        Fetches a specific batch of data.

        Args:
            batch_size (int): Number of samples per batch.
            batch_number (int): The batch number to fetch (1-based index).

        Returns:
            pandas.DataFrame: The fetched batch of data.
        """
        url = f"{self.base_url}/get/{self.public_id}/batch"
        params = {
            "batchSize": batch_size,
            "batchNumber": batch_number
        }
        data = make_request(url, headers=self.headers, params=params)
        
        batch_data_str = data.get("batchData")
        
        if not batch_data_str:
            raise DatasetSDKException(f"No data found for batch number {batch_number}.")
        
        try:
            # Convert the CSV string to a DataFrame
            batch_data = pd.read_csv(StringIO(batch_data_str))
        except pd.errors.ParserError as e:
            raise DatasetSDKException(f"Failed to parse 'batchData' CSV data: {e}")
        
        if batch_data.empty:
            raise DatasetSDKException(f"No data found for batch number {batch_number}.")
        
        return batch_data

    def _fetch_batch_data(self, batch_size, batch_number):
      
        url = f"{self.base_url}/get/{self.public_id}/encryptbatch"
        params = {
            "batchSize": batch_size,
            "batchNumber": batch_number
        }
        data = make_request(url, headers=self.headers, params=params)
        
        encrypted_data = data.get("batchData")
        encrypted_key = data.get("encryptedKey")
        
        if not encrypted_data or not encrypted_key:
            raise DatasetSDKException(f"No data found for batch number {batch_number}.")
        
        return encrypted_data,encrypted_key
    
    def _get_decrypted_batch(self, batch_size: int, batch_number: int) -> pd.DataFrame:
        """Secure batch decryption with automatic memory tracking"""
        # 1. Fetch encrypted data
        encrypted_data, encrypted_key = self._fetch_batch_data(batch_size, batch_number)
        
        # 2. Decrypt in memory
        decrypted_key = self._decrypt_key(encrypted_key)
        decrypted_data = self._decrypt_batch(encrypted_data, decrypted_key)
        
        # 3. Parse to DataFrame
        df = pd.read_csv(StringIO(decrypted_data))
        
        # 4. Track for cleanup
        self._last_decrypted = (df, decrypted_data)
        return df

    def _decrypt_batch(self, encrypted_data: str, key: str) -> str:
        """Optimized AES-256 CBC decryption"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            iv = encrypted_bytes[:16]
            ciphertext = encrypted_bytes[16:]
            
            cipher = Cipher(
                algorithms.AES(key.encode('utf-8')),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove PKCS7 padding
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted) + unpadder.finalize()
            return decrypted_data.decode('utf-8')  # now it's fully decoded after unpadding

        except Exception as e:
            self._emergency_cleanup()
            raise DatasetSDKException(f"Decryption failed: {str(e)}")

    def _emergency_cleanup(self):
        """Nuclear option for security failures"""
        if self._last_decrypted:
            df, raw_data = self._last_decrypted
            self._secure_wipe(df)
            self._secure_wipe(raw_data)
            self._last_decrypted = None

    def _secure_wipe(self, data):
        """Military-grade memory wiping (optimized)"""
        if isinstance(data, pd.DataFrame):
            # Fast numeric column wiping
            for col in data.select_dtypes(include=np.number).columns:
                data[col].values[:] = np.random.bytes(data[col].memory_usage())
            
            # String/object columns
            for col in data.select_dtypes(exclude=np.number).columns:
                data[col] = data[col].apply(lambda x: 'X'*len(str(x)))
            
            # Force memory release
            data.drop(data.index, inplace=True)
            
        elif isinstance(data, str):
            return 'X' * len(data)
        
        elif isinstance(data, bytes):
            return b'\x00' * len(data)

    def fetch_and_process_batch(self, batch_size: int, batch_number: int, process_func=None):
        """Main secure processing pipeline"""
        try:
            df = self._get_decrypted_batch(batch_size, batch_number)
            
            # Initialize fingerprint using first row hash
            if self._data_fingerprint is None:
                self._data_fingerprint = f"FP_{abs(hash(df.iloc[0].to_json())):X}"
                df['__secure__'] = self._data_fingerprint  # Hidden column
            
            results = process_func(df) if process_func else self._default_processing(df)
            self._validate_results(results)
            
            return results
        finally:
            self._emergency_cleanup()

    def _validate_results(self, results):
        """Lightning-fast validation (3 layers under 1ms)"""
        # Layer 1: Type check
        if isinstance(results, (pd.DataFrame, np.ndarray, bytes)):
            raise DatasetSDKException("Raw data structures prohibited")
        
        # Layer 2: String content check
        if isinstance(results, str) and self._data_fingerprint in results:
            raise DatasetSDKException("Data fingerprint detected in output")
        
        # Layer 3: Nested structure check
        if is_dict_like(results):
            stack = [results]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    stack.extend(obj.values())
                elif isinstance(obj, (list, tuple)):
                    stack.extend(obj)
                elif isinstance(obj, str) and self._data_fingerprint in obj:
                    raise DatasetSDKException("Nested data leak detected")

    def _decrypt_key(self, encrypted_key: str) -> str:
        """
        Decrypt the encrypted key using the environment's encryption key.
        
        :param encrypted_key: Base64 encoded encrypted key
        :return: Decrypted key as a string
        """
        encrypted_key_bytes = base64.b64decode(encrypted_key)
        decrypted_key = self._decrypt_data(encrypted_key_bytes, self.encryption_key)
        
        return decrypted_key.decode("utf-8")

    # def _decrypt_batch(self, encrypted_data: str, decrypted_key: str) -> str:
    #     """
    #     Decrypt the encrypted batch data using the decrypted key.
        
    #     :param encrypted_data: Base64 encoded encrypted batch data
    #     :param decrypted_key: Decrypted key to decrypt batch data
    #     :return: Decrypted batch data as a string (CSV format)
    #     """
    #     encrypted_data_bytes = base64.b64decode(encrypted_data)
    #     decrypted_data = self._decrypt_data(encrypted_data_bytes, decrypted_key)
    #     return decrypted_data.decode("utf-8")

    def _decrypt_data(self, encrypted_data: bytes, key: str) -> bytes:
        """
        Internal method to handle decryption logic for batch data and keys.
        
        :param encrypted_data: Encrypted data (bytes)
        :param key: Key used to decrypt the data
        :return: Decrypted data (bytes)
        """
        if not key:
         raise ValueError("Key cannot be None")
        try:
            # Convert key to bytes
            key_bytes = key.encode('utf-8')
            if len(key_bytes) != 32:
                raise DatasetSDKException("Invalid key length. Key must be 32 bytes for AES-256.")

            # Extract IV and ciphertext
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]

            # Initialize cipher
            cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()

            # Unpad the decrypted data
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

            return decrypted_data
        except Exception as e:
            raise DatasetSDKException(f"Decryption failed: {e}")

    def _default_preprocessing(self, data: str) -> pd.DataFrame:
        """
        Internal method for default preprocessing of decrypted data.
        
        :param data: Decrypted CSV data
        :return: Preprocessed data (e.g., parsed DataFrame)
        """
        try:
            df = pd.read_csv(StringIO(data),dtype_backend='pyarrow', engine='pyarrow')
            return df
        except pd.errors.ParserError as e:
            raise DatasetSDKException(f"Failed to parse decrypted CSV data: {e}")

    def _perform_operation(self, processed_data: pd.DataFrame):
        """
        Internal method to perform the operation requested by the client.
        The client never gets access to the decrypted data, only the result.
        
        :param processed_data: Data after preprocessing (pandas DataFrame)
        :return: The processed DataFrame or summary statistics
        """
        # Example: Return the DataFrame
        return processed_data