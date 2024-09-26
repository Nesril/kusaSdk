still there is a mis under standiing, fiirst let mee show you thee initialization,
import KusaaSDK

secret_key=''
dataset_public_id=''
ds = KusaaSDK.initialize(secret_key,publicId)

# please deecrypte and encryption process will be inside the sdk our  user is not authorized to make that it should be done internally

this simple code iis to show you how users wiill intialize so our sdk should be according with  that, users just initialize using secrete key and dataset public id then we will make 

so, when initializes you will call to http://localhost:5000/dataset/initialize/:publicId (Authorizetion:key secret_key )
 from this you will get totalRows of data and the first 10 files of csv so user can see the  total row and  the  first 10  rows any tiime if  the initialization successd

the other thing is i don't know like where the user should use  fetch_encrypted_batch, like is it after
 fetch_encrypted_batch
 here user will send  a request to http://localhost:5000/dataset/get/:publicId/batch?batchSize=&&batchNumber=


 class KusaaSdk:
    

    def fetch_encrypted_batch(self, batch_size: int) -> bytes:
        """
        Fetches an encrypted batch of data from the server.
        
        Parameters:
            batch_size (int): The number of data samples to retrieve in the batch.
        
        Returns:
            bytes: Encrypted data batch.
        """
        pass

    def decrypt_batch(self, encrypted_data: bytes) -> dict:
        """
        Decrypts the received encrypted data batch in-memory.
        
        Parameters:
            encrypted_data (bytes): The encrypted data batch to decrypt.
        
        Returns:
            dict: Decrypted data batch as a dictionary.
        """
        pass

    def get_data_generator(self, batch_size: int):
        """
        Provides a generator that yields decrypted data batches for model training.
        
        Parameters:
            batch_size (int): The number of data samples per batch.
        
        Yields:
            dict: Decrypted data batch.
        """
        pass

    def preprocess_batch(self, data_batch: dict, custom_func=None) -> dict:
        """
        Applies preprocessing to a single data batch. Supports built-in and custom preprocessing functions.
        
        Parameters:
            data_batch (dict): The decrypted data batch to preprocess.
            custom_func (callable, optional): A custom preprocessing function provided by the user.
        
        Returns:
            dict: Preprocessed data batch.
        """
        pass

    def balance_data(self, data_batch: dict) -> dict:
        """
        Balances the data within a batch to address class imbalances.
        
        Parameters:
            data_batch (dict): The data batch to balance.
        
        Returns:
            dict: Balanced data batch.
        """
        pass

    def clear_memory(self):
        """
        Clears all in-memory data to ensure that no decrypted data persists after training or preprocessing.
        """
        pass

    def integrate_with_framework(self, framework: str, model, preprocessing_steps: list):
        """
        Integrates the SDK with a specified machine learning or deep learning framework to facilitate model training.
        
        Parameters:
            framework (str): The ML/DL framework to integrate with (e.g., 'tensorflow', 'pytorch', 'scikit-learn').
            model: The machine learning model instance to train.
            preprocessing_steps (list): A list of preprocessing functions to apply to each data batch.
        
        Returns:
            None
        """
        pass

    def manage_encryption_keys(self):
        """
        Handles the retrieval and secure storage of encryption keys required for decrypting data.
        
        This function interacts with a Key Management Service (KMS) or secure key storage solution.
        """
        pass

    def log_activity(self, activity: str):
        """
        Logs user activities and data access for auditing and monitoring purposes.
        
        Parameters:
            activity (str): A description of the user activity to log.
        
        Returns:
            None
        """
        pass

    def handle_errors(self, error: Exception):
        """
        Handles errors gracefully by logging them and providing meaningful messages to the user.
        
        Parameters:
            error (Exception): The exception that occurred.
        
        Returns:
            None
        """
        pass

    def set_preprocessing_function(self, func: callable):
        """
        Allows users to set a custom preprocessing function that will be applied to each data batch.
        
        Parameters:
            func (callable): A user-defined function for preprocessing.
        
        Returns:
            None
        """
        pass

    def close_connection(self):
        """
        Safely closes the connection to the server and performs any necessary cleanup operations.
        """
        pass


