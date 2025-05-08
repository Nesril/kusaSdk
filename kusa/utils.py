# kusa/utils.py

import requests
from .exceptions import DatasetSDKException
import nltk


def make_request(url, headers=None, params=None):
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.HTTPError as e:
        # For HTTP errors, try to get the error details from the response
        try:
            error_details = response.json()
            raise DatasetSDKException(f"API request failed: Server response: {error_details}")
        except ValueError:
            # If response isn't JSON, include the raw text
            raise DatasetSDKException(f"API request failed: {str(e)}. Response content: {response.text}")
    except requests.exceptions.RequestException as e:
        raise DatasetSDKException(f"API request failed: {str(e)}")
    except ValueError as e:
        # For JSON decode errors
        response_content = response.text if 'response' in locals() else 'N/A'
        raise DatasetSDKException(f"Failed to decode JSON response: {str(e)}. Response content: {response_content}")
    

def ensure_nltk_tokenizer_resources():
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
            raise DatasetSDKException(f"Resource verification failed: {str(e)}")
        
        