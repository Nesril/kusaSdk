import requests
from .exceptions import DatasetSDKException

def make_request(url, headers=None, params=None):
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        raise DatasetSDKException(f"API request failed: {e}")
