import pandas as pd
from .utils import make_request
from .exceptions import DatasetSDKException

class DatasetClient:
    def __init__(self, public_id, secret_key, base_url="http://localhost:5000/dataset"):
        """
        Initializes the DatasetClient with authentication details.

        Args:
            public_id (str): Public ID of the dataset.
            secret_key (str): Secret key for authorization.
            base_url (str): Base URL of the dataset server.
        """
        self.public_id = public_id
        self.secret_key = secret_key
        self.base_url = base_url
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
        first_10_rows = pd.DataFrame(data.get("first10Rows"))

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
        
        batch_data = pd.DataFrame(data.get("batchData"))

        if batch_data.empty:
            raise DatasetSDKException(f"No data found for batch number {batch_number}.")
        
        return batch_data
