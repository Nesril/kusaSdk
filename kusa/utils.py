# kusa/utils.py

import requests
from .exceptions import DatasetSDKException

def make_request(url, headers=None, params=None):
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        raise DatasetSDKException(f"API request failed: {e}")
    except ValueError as e:
        # Attempt to include response content in the exception
        response_content = response.text if 'response' in locals() else 'N/A'
        raise DatasetSDKException(f"API request failed: {e}. Response content: {response_content}")
