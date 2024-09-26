# your_sdk/client.py

import pandas as pd
from io import StringIO
from .utils import make_request
from .exceptions import DatasetSDKException
import os
from dotenv import load_dotenv

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
