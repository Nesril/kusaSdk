# kusa/client.py

import pandas as pd
from io import StringIO
from .utils import make_request
from .exceptions import DatasetSDKException
import os
from dotenv import load_dotenv

import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


# Load environment variables
load_dotenv()

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
        self.base_url = os.getenv('BASE_URL')
        self.encryption_key = os.getenv("ENCRYPTION_KEY")
        
        if not self.public_id or not self.secret_key:
            raise DatasetSDKException("PUBLIC_ID and SECRET_KEY must be set either as parameters or in the .env file.")
        
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
    
    def fetch_and_process_batch(self, batch_size: int, batch_number: int, preprocess_func=None):
        """
        Fetch encrypted batch data, decrypt it in memory, and process it according to the client's needs.
        
        :param batch_size: Size of the batch to retrieve
        :param batch_number: Batch number to fetch
        :param preprocess_func: Optional custom preprocessing function to apply
        :return: The result after processing the batch (e.g., model training result, summary statistics)
        """
        # Fetch encrypted batch data and encrypted key from the server
        encrypted_data, encrypted_key = self._fetch_batch_data(batch_size, batch_number)

        # Decrypt the key using the environment encryption key
        decrypted_key = self._decrypt_key(encrypted_key)

        # Decrypt the batch data in memory using the decrypted key
        decrypted_data = self._decrypt_batch(encrypted_data, decrypted_key)

        # Preprocess the data (if a preprocessing function is provided)
        if preprocess_func:
            processed_data = preprocess_func(decrypted_data)
        else:
            processed_data = self._default_preprocessing(decrypted_data)

        # Return results based on processed data (e.g., model training result, summary statistics)
        return self._perform_operation(processed_data)

    def _decrypt_key(self, encrypted_key: str) -> str:
        """
        Decrypt the encrypted key using the environment's encryption key.
        
        :param encrypted_key: Base64 encoded encrypted key
        :return: Decrypted key as a string
        """
        encrypted_key_bytes = base64.b64decode(encrypted_key)
        decrypted_key = self._decrypt_data(encrypted_key_bytes, self.encryption_key)
        
        return decrypted_key.decode("utf-8")

    def _decrypt_batch(self, encrypted_data: str, decrypted_key: str) -> str:
        """
        Decrypt the encrypted batch data using the decrypted key.
        
        :param encrypted_data: Base64 encoded encrypted batch data
        :param decrypted_key: Decrypted key to decrypt batch data
        :return: Decrypted batch data as a string (CSV format)
        """
        encrypted_data_bytes = base64.b64decode(encrypted_data)
        decrypted_data = self._decrypt_data(encrypted_data_bytes, decrypted_key)
        return decrypted_data.decode("utf-8")

    def _decrypt_data(self, encrypted_data: bytes, key: str) -> bytes:
        """
        Internal method to handle decryption logic for batch data and keys.
        
        :param encrypted_data: Encrypted data (bytes)
        :param key: Key used to decrypt the data
        :return: Decrypted data (bytes)
        """
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
            df = pd.read_csv(StringIO(data))
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