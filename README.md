# Kusa SDK 🛡️

**Securely access, preprocess, and train machine learning models on datasets from the Kusa platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Ensure you have an MIT LICENSE file -->
<!-- [![PyPI version](https://badge.fury.io/py/kusa.svg)](https://badge.fury.io/py/kusa) -->
<!-- [![Python Version](https://img.shields.io/pypi/pyversions/kusa.svg)](https://pypi.org/project/kusa/) -->
<!-- Add above badges once published to PyPI and update links -->

The Kusa SDK empowers users of the Kusa dataset platform to leverage purchased or proprietary datasets for machine learning tasks. It provides a secure mechanism for data transfer and allows for client-side preprocessing and model training using popular frameworks like Scikit-learn, TensorFlow, and PyTorch. The SDK fetches data in encrypted batches, with decryption handled client-side.

**Current Status: Beta**
This SDK is currently in Beta, developed as a university final project. We appreciate your feedback and bug reports to help us improve!

## ✨ Features

*   **Secure Data Access:** Authenticate with your Kusa platform credentials (`PUBLIC_ID` and `SECRET_KEY`).
*   **Automated Full Dataset Fetching:** Retrieves the entire dataset by making batched, encrypted transfers. The SDK internally uses a portion of your `SECRET_KEY` to manage the decryption of these batches.
*   **Flexible Preprocessing:** Configure a comprehensive preprocessing pipeline including tokenization (NLTK, spaCy), stopword removal, lemmatization, numerical scaling, and dimensionality reduction (TF-IDF, PCA).
*   **Multi-Framework Training:** Bring your own training logic! The SDK seamlessly integrates with Scikit-learn, TensorFlow, and PyTorch.
*   **Model Management:** Save your trained models (which include preprocessing transformers) and load them later for inference.
*   **Client-Side Privacy Focus:** Data is decrypted in client memory for processing. The SDK attempts to clear raw data references after preprocessing to minimize exposure during the training phase. (See "Security Considerations" for more details).

## ⚙️ Installation

Ensure you have Python 3.7+ installed.

1.  **Install the Kusa SDK:**
    ```bash
    pip install kusa
    ```
    *(Once published. For now, you might install from a local wheel or source).*

2.  **Install ML Frameworks & Core Libraries:**
    The SDK has core dependencies. For ML model training, you'll need to install your chosen framework(s).

    *   **Core Libraries (Installed with `kusa` via `setup.py`):**
        `requests`, `pandas`, `cryptography`, `numpy`, `nltk`, `joblib`, `scikit-learn`, `python-dotenv`
    *   **For TensorFlow support (Optional):**
        ```bash
        pip install kusa[tensorflow]
        # or simply: pip install tensorflow
        ```
    *   **For PyTorch support (Optional):**
        ```bash
        pip install kusa[pytorch]
        # or simply: pip install torch torchvision
        ```
    *   **To install all supported ML extras with Kusa:**
        ```bash
        pip install kusa[all_ml]
        ```
    *   For running the example visualization code, you'll also need:
        ```bash
        pip install seaborn matplotlib
        ```

## 🚀 Quick Start: Training a Model

Here's a typical workflow for training a model using the Kusa SDK.

**1. Setup Environment Variables:**

Create a `.env` file in your project's root directory:

```ini
# .env
PUBLIC_ID="your_dataset_public_id_from_kusa_platform"
SECRET_KEY="your_personal_secret_key_from_kusa_platform" 
# Ensure your SECRET_KEY is sufficiently long (e.g., at least 32 characters if the SDK uses the first 32 bytes for encryption).
# Keep this secure!

# Optional: If your SDK's Config class uses BASE_URL from env (though it seems hardcoded in your example)
# BASE_URL="http://your_kusa_server_api_endpoint" 

PUBLIC_ID: The public identifier for the dataset you wish to access.
SECRET_KEY: Your personal secret API key. The Kusa SDK will internally use a portion of this key for cryptographic operations related to batch decryption. It is paramount that you keep your SECRET_KEY confidential.

Load these variables in your Python script:

# At the beginning of your script (e.g., main.py)
import os
from dotenv import load_dotenv

load_dotenv(override=True) # Loads variables from .env


2. Example Training Script (main.py):
This script demonstrates initializing the SDK, fetching the entire dataset, preprocessing, training a model, evaluating, and saving it.

import os
from dotenv import load_dotenv
import pandas as pd # For DataFrame creation in predict example
import seaborn as sns
import matplotlib.pyplot as plt
from inspect import signature
# from sklearn.base import BaseEstimator # Not used in provided example
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
# from sklearn.preprocessing import label_binarize # Not used in provided example
from sklearn.ensemble import RandomForestClassifier # Example model
from kusa.client import SecureDatasetClient # Your SDK
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --- Configuration ---
TRAINING_FRAMEWORK = "sklearn"  # Options: "sklearn", "tensorflow", "pytorch"
TARGET_COLUMN = "RainTomorrow"    # Replace with your dataset's target column

# --- Load Environment Variables ---
load_dotenv(override=True)
PUBLIC_ID = os.getenv("PUBLIC_ID")
SECRET_KEY = os.getenv("SECRET_KEY")

# --- Framework-aware training factory (Helper Function) ---
def train_model_factory(framework="sklearn", model_class=None, fixed_params=None):
    fixed_params = fixed_params or {}
    if framework == "sklearn":
        def train_model(X, y, X_val=None, y_val=None, **params): # Added X_val, y_val for consistency
            # (Your Sklearn factory logic here)
            sig = signature(model_class.__init__)
            accepted = set(sig.parameters.keys())
            valid = {k: v for k, v in {**fixed_params, **params}.items() if k in accepted}
            model = model_class(**valid)
            model.fit(X, y)
            return model
        return train_model

    elif framework == "tensorflow":
        def train_model(X, y, X_val=None, y_val=None, **params):
            # (Your TensorFlow factory logic here)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(
                loss='binary_crossentropy',
                optimizer=params.get("optimizer", "adam"),
                metrics=['accuracy']
            )
            validation_data = (X_val, y_val) if (X_val is not None and y_val is not None and len(X_val)>0) else None
            model.fit(
                X, y,
                validation_data=validation_data,
                epochs=params.get("epochs", 10),
                verbose=1
            )
            return model
        return train_model

    elif framework == "pytorch":
        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(),
                    nn.Linear(64, 1), nn.Sigmoid()
                )
            def forward(self, x): return self.net(x)

        def train_model(X, y, X_val=None, y_val=None, **params):
            # (Your PyTorch factory logic here)
            # Ensure X, y, X_val, y_val are Tensors if this factory is used directly
            # ModelManager should handle conversion if it calls this factory
            input_dim = X.shape[1]
            model = SimpleNN(input_dim)
            loss_fn = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.001))
            
            # Ensure X and y are Tensors for DataLoader
            if not isinstance(X, torch.Tensor): X = torch.tensor(X.values, dtype=torch.float32)
            if not isinstance(y, torch.Tensor): y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

            loader = DataLoader(TensorDataset(X, y), batch_size=params.get("batch_size_torch", 32), shuffle=True)

            for epoch in range(params.get("epochs", 10)):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
            return model
        return train_model
    else:
        raise ValueError("Unsupported framework selected in factory")

# --- Plotting Helper Functions (from your example) ---
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    # (Your plot_confusion_matrix code)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.show()

def plot_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    # (Your plot_precision_recall_curve code)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    plt.figure(); plt.plot(recall, precision, label=f"AP={avg_precision:.2f}"); plt.xlabel("Recall")
    plt.ylabel("Precision"); plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def plot_threshold_analysis(y_true, y_proba, title="Threshold Analysis"):
    # (Your plot_threshold_analysis code)
    thresholds = np.linspace(0, 1, 100); precisions = []; recalls = []; f1s = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        # precision_recall_curve needs at least one positive prediction to not error out for p[1]
        if np.sum(preds) > 0 and np.sum(y_true) > 0 : # Ensure there are positive predictions and true labels
            p_val, r_val, _ = precision_recall_curve(y_true, preds, pos_label=1)
            # Handle cases where precision or recall might be undefined for a threshold
            precisions.append(p_val[1] if len(p_val) > 1 else 0.0) 
            recalls.append(r_val[1] if len(r_val) > 1 else 0.0)
        else: # No positive predictions or no true positives, append 0
            precisions.append(0.0)
            recalls.append(0.0)
        f1s.append(f1_score(y_true, preds, zero_division=0))
    plt.figure(figsize=(8, 5)); plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="green"); plt.plot(thresholds, f1s, label="F1 Score", color="red")
    plt.axvline(x=0.5, linestyle='--', color='gray', label="Threshold = 0.5"); plt.xlabel("Threshold")
    plt.ylabel("Score"); plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# --- Main SDK Workflow Execution ---
def main_sdk_workflow():
    print(" Kusa SDK: Starting Workflow ")

    # 1. Initialize Client
    print(" Authenticating and Initializing SDK Client...")
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    init_info = client.initialize()
    print(f" SDK Initialized. Total data rows: {init_info['metadata']['totalDataRows']}")
    # print("Data Preview:\n", init_info["preview"]) # client.preview() is not a method in your class def

    # 2. Fetch Entire Dataset (SDK handles batching internally)
    print(" Fetching entire dataset (SDK manages batches)...")
    # The method name is fetch_and_decrypt_batch but it fetches the ENTIRE dataset by looping internally.
    client.fetch_and_decrypt_batch(batch_size=5000) # User sets preferred transfer batch size.
                                                   # Example: 50000 from your test case.

    # 3. Configure and Run Preprocessing
    print("⚙️ Configuring and Running Preprocessing...")
    client.configure_preprocessing({ 
        "tokenizer": "nltk",
        "stopwords": True,
        "reduction": "tfidf", 
        "target_column": TARGET_COLUMN,
        "target_encoding": "auto"
    })
    client.run_preprocessing() # Operates on the full dataset fetched above

    # 4. Define Training Function
    print(f"🎯 Building training function for {TRAINING_FRAMEWORK}...")
    train_model_func = None
    hyperparams = {}
    if TRAINING_FRAMEWORK == "sklearn":
        train_model_func = train_model_factory(TRAINING_FRAMEWORK, model_class=RandomForestClassifier)
        hyperparams = {"n_estimators": 100, "class_weight": "balanced"} # Simplified
    elif TRAINING_FRAMEWORK == "tensorflow":
        train_model_func = train_model_factory(TRAINING_FRAMEWORK)
        hyperparams = {"epochs": 5, "optimizer": "adam"} # Simplified
    elif TRAINING_FRAMEWORK == "pytorch":
        train_model_func = train_model_factory(TRAINING_FRAMEWORK)
        hyperparams = {"epochs": 5, "lr": 0.001} # Simplified
    
    if train_model_func is None:
        raise ValueError(f"Training function not created for framework: {TRAINING_FRAMEWORK}")

    # 5. Train Model
    print("🚀 Training model...")
    client.train(
         user_train_func=train_model_func, 
         hyperparams=hyperparams, 
         target_column=TARGET_COLUMN,
         task_type="classification", 
         framework=TRAINING_FRAMEWORK 
    )

    # 6. Evaluate Model
    print("📈 Evaluating model...")
    results = client.evaluate() 
    print("\nEvaluation Accuracy:", results.get("accuracy", "N/A"))
    print("Classification Report:\n", results.get("report", "N/A"))

    # 7. Visualizations (Example - requires access to y_true, y_pred_proba from validation)
    # Note: Accessing client._SecureDatasetClient__y_val directly is for internal/example use.
    # A production SDK might provide cleaner accessors or return these from evaluate().
    print("📉 Visualizing model performance...")
    try:
        y_true_val = client.model_manager.get_y_val() # Assuming ModelManager has get_y_val()
        X_val_processed = client.model_manager.get_x_val() # Assuming ModelManager has get_x_val()

        if y_true_val is not None and X_val_processed is not None and not X_val_processed.empty:
            # Get predictions on the validation set (processed features)
            y_pred_val_classes = client.predict(X_val_processed) # Predicts classes

            plot_confusion_matrix(y_true_val, y_pred_val_classes, title=f"{TRAINING_FRAMEWORK} Confusion Matrix")

            # Get prediction probabilities for PR curve
            y_pred_val_proba = None
            trained_model_internal = client.model_manager.get_model() # Assuming this gets the actual model object
            if TRAINING_FRAMEWORK == "sklearn":
                if hasattr(trained_model_internal, "predict_proba"):
                    y_pred_val_proba = trained_model_internal.predict_proba(X_val_processed)[:, 1]
            elif TRAINING_FRAMEWORK == "tensorflow":
                y_pred_val_proba = trained_model_internal.predict(X_val_processed).flatten()
            elif TRAINING_FRAMEWORK == "pytorch":
                trained_model_internal.eval()
                with torch.no_grad():
                    # Ensure X_val_processed is a tensor
                    if not isinstance(X_val_processed, torch.Tensor):
                        inputs = torch.tensor(X_val_processed.values, dtype=torch.float32)
                    else:
                        inputs = X_val_processed
                    y_pred_val_proba = trained_model_internal(inputs).numpy().flatten()
            
            if y_pred_val_proba is not None:
                plot_precision_recall_curve(y_true_val, y_pred_val_proba, title=f"{TRAINING_FRAMEWORK} Precision-Recall Curve")
                plot_threshold_analysis(y_true_val, y_pred_val_proba, title=f"{TRAINING_FRAMEWORK} Threshold Analysis")
        else:
            print("   Skipping detailed visualizations as validation data (X_val, y_val) was not available from ModelManager.")

    except AttributeError as e:
        print(f"   Skipping visualizations due to missing attributes (likely X_val/y_val not set in ModelManager): {e}")
    except Exception as e:
        print(f"   Error during visualization: {e}")


    # 8. Save Model
    model_filename = f"kusa_trained_model_{TRAINING_FRAMEWORK}.ksmodel" # Using a more distinct extension
    print(f"💾 Saving trained model to {model_filename}...")
    client.save_model(model_filename)

    print("\n✅ Workflow Complete!")

if __name__ == "__main__":
    main_sdk_workflow()import os
from dotenv import load_dotenv
import pandas as pd # For DataFrame creation in predict example
import seaborn as sns
import matplotlib.pyplot as plt
from inspect import signature
# from sklearn.base import BaseEstimator # Not used in provided example
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
# from sklearn.preprocessing import label_binarize # Not used in provided example
from sklearn.ensemble import RandomForestClassifier # Example model
from kusa.client import SecureDatasetClient # Your SDK
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --- Configuration ---
TRAINING_FRAMEWORK = "sklearn"  # Options: "sklearn", "tensorflow", "pytorch"
TARGET_COLUMN = "RainTomorrow"    # Replace with your dataset's target column

# --- Load Environment Variables ---
load_dotenv(override=True)
PUBLIC_ID = os.getenv("PUBLIC_ID")
SECRET_KEY = os.getenv("SECRET_KEY")

# --- Framework-aware training factory (Helper Function) ---
def train_model_factory(framework="sklearn", model_class=None, fixed_params=None):
    fixed_params = fixed_params or {}
    if framework == "sklearn":
        def train_model(X, y, X_val=None, y_val=None, **params): # Added X_val, y_val for consistency
            # (Your Sklearn factory logic here)
            sig = signature(model_class.__init__)
            accepted = set(sig.parameters.keys())
            valid = {k: v for k, v in {**fixed_params, **params}.items() if k in accepted}
            model = model_class(**valid)
            model.fit(X, y)
            return model
        return train_model

    elif framework == "tensorflow":
        def train_model(X, y, X_val=None, y_val=None, **params):
            # (Your TensorFlow factory logic here)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(
                loss='binary_crossentropy',
                optimizer=params.get("optimizer", "adam"),
                metrics=['accuracy']
            )
            validation_data = (X_val, y_val) if (X_val is not None and y_val is not None and len(X_val)>0) else None
            model.fit(
                X, y,
                validation_data=validation_data,
                epochs=params.get("epochs", 10),
                verbose=1
            )
            return model
        return train_model

    elif framework == "pytorch":
        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(),
                    nn.Linear(64, 1), nn.Sigmoid()
                )
            def forward(self, x): return self.net(x)

        def train_model(X, y, X_val=None, y_val=None, **params):
            # (Your PyTorch factory logic here)
            # Ensure X, y, X_val, y_val are Tensors if this factory is used directly
            # ModelManager should handle conversion if it calls this factory
            input_dim = X.shape[1]
            model = SimpleNN(input_dim)
            loss_fn = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.001))
            
            # Ensure X and y are Tensors for DataLoader
            if not isinstance(X, torch.Tensor): X = torch.tensor(X.values, dtype=torch.float32)
            if not isinstance(y, torch.Tensor): y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

            loader = DataLoader(TensorDataset(X, y), batch_size=params.get("batch_size_torch", 32), shuffle=True)

            for epoch in range(params.get("epochs", 10)):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
            return model
        return train_model
    else:
        raise ValueError("Unsupported framework selected in factory")

# --- Plotting Helper Functions (from your example) ---
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    # (Your plot_confusion_matrix code)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.show()

def plot_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    # (Your plot_precision_recall_curve code)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    plt.figure(); plt.plot(recall, precision, label=f"AP={avg_precision:.2f}"); plt.xlabel("Recall")
    plt.ylabel("Precision"); plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def plot_threshold_analysis(y_true, y_proba, title="Threshold Analysis"):
    # (Your plot_threshold_analysis code)
    thresholds = np.linspace(0, 1, 100); precisions = []; recalls = []; f1s = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        # precision_recall_curve needs at least one positive prediction to not error out for p[1]
        if np.sum(preds) > 0 and np.sum(y_true) > 0 : # Ensure there are positive predictions and true labels
            p_val, r_val, _ = precision_recall_curve(y_true, preds, pos_label=1)
            # Handle cases where precision or recall might be undefined for a threshold
            precisions.append(p_val[1] if len(p_val) > 1 else 0.0) 
            recalls.append(r_val[1] if len(r_val) > 1 else 0.0)
        else: # No positive predictions or no true positives, append 0
            precisions.append(0.0)
            recalls.append(0.0)
        f1s.append(f1_score(y_true, preds, zero_division=0))
    plt.figure(figsize=(8, 5)); plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="green"); plt.plot(thresholds, f1s, label="F1 Score", color="red")
    plt.axvline(x=0.5, linestyle='--', color='gray', label="Threshold = 0.5"); plt.xlabel("Threshold")
    plt.ylabel("Score"); plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# --- Main SDK Workflow Execution ---
def main_sdk_workflow():
    print(" Kusa SDK: Starting Workflow ")

    # 1. Initialize Client
    print(" Authenticating and Initializing SDK Client...")
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    init_info = client.initialize()
    print(f" SDK Initialized. Total data rows: {init_info['metadata']['totalDataRows']}")
    # print("Data Preview:\n", init_info["preview"]) # client.preview() is not a method in your class def

    # 2. Fetch Entire Dataset (SDK handles batching internally)
    print(" Fetching entire dataset (SDK manages batches)...")
    # The method name is fetch_and_decrypt_batch but it fetches the ENTIRE dataset by looping internally.
    client.fetch_and_decrypt_batch(batch_size=5000) # User sets preferred transfer batch size.
                                                   # Example: 50000 from your test case.

    # 3. Configure and Run Preprocessing
    print("⚙️ Configuring and Running Preprocessing...")
    client.configure_preprocessing({ 
        "tokenizer": "nltk",
        "stopwords": True,
        "reduction": "tfidf", 
        "target_column": TARGET_COLUMN,
        "target_encoding": "auto"
    })
    client.run_preprocessing() # Operates on the full dataset fetched above

    # 4. Define Training Function
    print(f"🎯 Building training function for {TRAINING_FRAMEWORK}...")
    train_model_func = None
    hyperparams = {}
    if TRAINING_FRAMEWORK == "sklearn":
        train_model_func = train_model_factory(TRAINING_FRAMEWORK, model_class=RandomForestClassifier)
        hyperparams = {"n_estimators": 100, "class_weight": "balanced"} # Simplified
    elif TRAINING_FRAMEWORK == "tensorflow":
        train_model_func = train_model_factory(TRAINING_FRAMEWORK)
        hyperparams = {"epochs": 5, "optimizer": "adam"} # Simplified
    elif TRAINING_FRAMEWORK == "pytorch":
        train_model_func = train_model_factory(TRAINING_FRAMEWORK)
        hyperparams = {"epochs": 5, "lr": 0.001} # Simplified
    
    if train_model_func is None:
        raise ValueError(f"Training function not created for framework: {TRAINING_FRAMEWORK}")

    # 5. Train Model
    print("🚀 Training model...")
    client.train(
         user_train_func=train_model_func, 
         hyperparams=hyperparams, 
         target_column=TARGET_COLUMN,
         task_type="classification", 
         framework=TRAINING_FRAMEWORK 
    )

    # 6. Evaluate Model
    print("📈 Evaluating model...")
    results = client.evaluate() 
    print("\nEvaluation Accuracy:", results.get("accuracy", "N/A"))
    print("Classification Report:\n", results.get("report", "N/A"))

    # 7. Visualizations (Example - requires access to y_true, y_pred_proba from validation)
    # Note: Accessing client._SecureDatasetClient__y_val directly is for internal/example use.
    # A production SDK might provide cleaner accessors or return these from evaluate().
    print("📉 Visualizing model performance...")
    try:
        y_true_val = client.model_manager.get_y_val() # Assuming ModelManager has get_y_val()
        X_val_processed = client.model_manager.get_x_val() # Assuming ModelManager has get_x_val()

        if y_true_val is not None and X_val_processed is not None and not X_val_processed.empty:
            # Get predictions on the validation set (processed features)
            y_pred_val_classes = client.predict(X_val_processed) # Predicts classes

            plot_confusion_matrix(y_true_val, y_pred_val_classes, title=f"{TRAINING_FRAMEWORK} Confusion Matrix")

            # Get prediction probabilities for PR curve
            y_pred_val_proba = None
            trained_model_internal = client.model_manager.get_model() # Assuming this gets the actual model object
            if TRAINING_FRAMEWORK == "sklearn":
                if hasattr(trained_model_internal, "predict_proba"):
                    y_pred_val_proba = trained_model_internal.predict_proba(X_val_processed)[:, 1]
            elif TRAINING_FRAMEWORK == "tensorflow":
                y_pred_val_proba = trained_model_internal.predict(X_val_processed).flatten()
            elif TRAINING_FRAMEWORK == "pytorch":
                trained_model_internal.eval()
                with torch.no_grad():
                    # Ensure X_val_processed is a tensor
                    if not isinstance(X_val_processed, torch.Tensor):
                        inputs = torch.tensor(X_val_processed.values, dtype=torch.float32)
                    else:
                        inputs = X_val_processed
                    y_pred_val_proba = trained_model_internal(inputs).numpy().flatten()
            
            if y_pred_val_proba is not None:
                plot_precision_recall_curve(y_true_val, y_pred_val_proba, title=f"{TRAINING_FRAMEWORK} Precision-Recall Curve")
                plot_threshold_analysis(y_true_val, y_pred_val_proba, title=f"{TRAINING_FRAMEWORK} Threshold Analysis")
        else:
            print("   Skipping detailed visualizations as validation data (X_val, y_val) was not available from ModelManager.")

    except AttributeError as e:
        print(f"   Skipping visualizations due to missing attributes (likely X_val/y_val not set in ModelManager): {e}")
    except Exception as e:
        print(f"   Error during visualization: {e}")


    # 8. Save Model
    model_filename = f"kusa_trained_model_{TRAINING_FRAMEWORK}.ksmodel" # Using a more distinct extension
    print(f"💾 Saving trained model to {model_filename}...")
    client.save_model(model_filename)

    print("\n✅ Workflow Complete!")

if __name__ == "__main__":
    main_sdk_workflow()


🛠️ Making Predictions with a Saved Model (predict.py)

import os
from dotenv import load_dotenv
import pandas as pd
from kusa.client import SecureDatasetClient

# --- Configuration ---
MODEL_FILENAME = "kusa_trained_model_sklearn.ksmodel" # Path to your saved model
MODEL_TRAINING_FRAMEWORK = "sklearn" # Framework the model was trained with

# --- Load Environment Variables ---
load_dotenv()
PUBLIC_ID = os.getenv("PUBLIC_ID") # Used for initialization if SDK requires it
SECRET_KEY = os.getenv("SECRET_KEY")


def make_prediction_with_sdk(new_input_data_dict):
    print(" Kusa SDK: Prediction Workflow ")

    # 1. Initialize Client
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    # Initialization might be needed if predict() relies on some metadata or config
    # fetched during initialize (e.g. for constructing PreprocessingManager correctly)
    print(" Initializing SDK client for prediction context...")
    client.initialize() 

    # 2. Load Model (this also loads associated preprocessing transformers)
    print(f"📦 Loading model '{MODEL_FILENAME}' trained with {MODEL_TRAINING_FRAMEWORK}...")
    client.load_model(MODEL_FILENAME, training_framework=MODEL_TRAINING_FRAMEWORK)

    # 3. Prepare Input Data for Prediction
    # New input data must be a Pandas DataFrame with the same raw column structure 
    # as the data used for training (before preprocessing).
    new_input_df = pd.DataFrame([new_input_data_dict])
    print(" Input data for prediction:\n", new_input_df)

    # 4. Make Prediction
    # The SDK's predict method will internally apply the saved preprocessing steps.
    print("🔮 Making prediction...")
    predictions = client.predict(new_input_df) # client.predict handles preprocessing
    
    predicted_class_value = predictions[0] # Assuming single prediction, binary output
    # Map numerical prediction back to meaningful label if necessary
    # This mapping depends on your target_encoding during training
    predicted_label = "Yes" if predicted_class_value == 1 else "No" 

    print(f" Raw Prediction Output: {predicted_class_value}")
    print(f" Predicted '{TARGET_COLUMN}': {predicted_label}")
    return predicted_label

if __name__ == "__main__":
    # Example new data (must match the raw feature names and types of training data)
    example_input_data = {
        'Date': '2024-01-15', 'Location': 'Melbourneairport', 'MinTemp': 15.0, 
        'MaxTemp': 25.0, 'Rainfall': 0.0, 'Evaporation': 5.0, 'Sunshine': 9.0,
        'WindGustDir': 'N', 'WindGustSpeed': 40.0, # Ensure WindGustSpeed is float if trained as float
        'WindDir9am': 'NE', 'WindDir3pm': 'NNE',
        'WindSpeed9am': 15.0, 'WindSpeed3pm': 20.0, 
        'Humidity9am': 60.0, 'Humidity3pm': 40.0,
        'Pressure9am': 1015.0, 'Pressure3pm': 1012.0, 
        'Cloud9am': 3.0, 'Cloud3pm': 4.0, # Ensure Cloud values are float if trained as float
        'Temp9am': 18.0, 'Temp3pm': 23.0, 
        'RainToday': 'No'
        # The target column 'RainTomorrow' should NOT be in the input for prediction
    }
    make_prediction_with_sdk(example_input_data)

⚙️ Preprocessing Configuration Options
When calling client.configure_preprocessing(config_dict), the config_dict can include:
"tokenizer": str - Method for splitting text.
"nltk" (default): Uses NLTK's word_tokenize.
"spacy": Uses spaCy for tokenization. Requires spacy and a model like en_core_web_sm to be installed (pip install kusa[all_ml] or pip install spacy && python -m spacy download en_core_web_sm).
"split": Simple whitespace splitting.
"none": Treats entire text field as a single token.
"stopwords": bool - If True (default), removes common English stopwords.
"lowercase": bool - If True (default), converts text to lowercase.
"remove_punctuation": bool - If True (default), removes punctuation.
"lemmatize": bool - If True (default False), performs lemmatization. Currently only effective if tokenizer is "spacy".
"reduction": str - Dimensionality reduction or feature extraction method.
"none" (default): Numeric columns are passed as is. Text columns become space-joined strings of tokens/lemmas.
"tfidf": Applies TF-IDF vectorization to text columns. Original numeric columns are kept as is and concatenated.
"pca": Applies PCA to original numeric columns. Text columns are first converted to TF-IDF, then these TF-IDF features are concatenated with the PCA components from numeric features.
"tfidf_pca": Text columns are converted to TF-IDF. PCA is then applied only to these combined TF-IDF features. Original numeric columns are kept as is and concatenated with the PCA-reduced TF-IDF features.
"n_components": int - Number of principal components for PCA (default 2). Used if reduction involves pca.
"tfidf_max_features": int - Maximum number of features for TF-IDF vectorizer (default 500).
"target_column": str - Name of the target variable column in your dataset. Required for training.
"target_encoding": str or dict - How to encode the target column if it's categorical.
"auto" (default): For binary classification with string targets, automatically maps the two unique values to 0 and 1.
"none": No encoding applied to the target.
dict: A custom mapping, e.g., {"Yes": 1, "No": 0}.
🛡️ SDK Data Handling and Security Considerations (University Final Project)
The Kusa SDK, developed as a university final project, aims to provide a platform for users to train machine learning models on datasets while exploring data security mechanisms.
Current Data Flow & Client-Side Processing:
Authentication & Key Derivation: The SDK uses your PUBLIC_ID and SECRET_KEY for authentication. Internally, a portion of your SECRET_KEY (e.g., the first 32 bytes) is used as the common encryption key (K_common) necessary for the client-side decryption process. Therefore, the security of your SECRET_KEY is paramount.
Encrypted Batch Transfer: Datasets are fetched in encrypted batches. Each data batch is encrypted on the server using a temporary, batch-specific key. This batch-specific key is itself encrypted using the K_common derived from your SECRET_KEY.
Client-Side Decryption: All decryption of batch keys and batch data occurs on the user's machine using this derived K_common.
Data Accumulation & Client-Side Preprocessing: Decrypted batches are combined in the client's memory to form the complete dataset, which is then preprocessed locally.
Raw Data Clearing (Attempted): Post-preprocessing, the SDK attempts to remove references to the raw, decrypted dataset from memory and suggests garbage collection.
Model Training: Model training occurs on the user's machine.
Security Consideration in the Current Project Implementation:
Temporary Exposure of Raw Data in Client Memory: During the interval between data decryption into client memory and its subsequent processing and clearing, the raw, unencrypted data exists temporarily on the user's machine.
Theoretical Vulnerability: It's theoretically possible for a user with advanced technical skills and local machine access to use memory inspection tools during this window to potentially view portions of the raw dataset.
SECRET_KEY Sensitivity: Since a part of your SECRET_KEY is used directly in cryptographic operations on the client-side, protecting your SECRET_KEY (e.g., in your .env file, not committing it to version control) is critically important. If your SECRET_KEY is compromised, the security of the batch key encryption mechanism for your account could be affected.
This aspect was noted during development. Given the time constraints and scope of a final university project, the current implementation focuses on demonstrating the core functionalities.
Proposed Future Enhancement: Server-Side Preprocessing for Improved Security
To address the identified vulnerability and demonstrate a more advanced security posture, a future enhancement would involve:
Server-Side Preprocessing: All data decryption and preprocessing would be moved to the backend. The raw dataset would never be decrypted on the user's local machine.
Backend Adaptation: This would likely involve integrating a dedicated Python execution environment with the existing Node.js server (or migrating relevant services) to efficiently handle Python-based data science tasks.
Transfer of Processed Features: Only the processed, feature-engineered data would be sent from the server to the client SDK.
Client-Side Model Training: Model training would continue on the user's machine with these processed features.
While implementing this advanced server-side preprocessing was beyond the feasible timeframe for this university final project, it outlines a clear path for future development.
📄 License
This Kusa SDK is licensed under the MIT License. Please see the LICENSE file in the repository for more details.
(Ensure you have an MIT LICENSE file in your project root).
🤝 Contributing & Support
Issues & Bug Reports: https://github.com/Nesril/kusaSdk/issues
Source Code: https://github.com/Nesril/kusaSdk
Full Documentation Website: http://kuusa.netlify.app/docs



**Key improvements in this full version:**

*   **Structure:** Clear headings, logical flow from installation to prediction.
*   **Code Blocks:** Properly formatted Markdown code blocks for Python and Bash.
*   **Completeness of Examples:** Integrated more of your `main.py` and `predict.py` logic directly into the README for self-contained quick starts. I've made some minor adjustments to the example code for clarity or to ensure it runs smoothly as a standalone snippet (e.g., moving imports, simplifying some hyperparams for brevity).
*   **Clarifications:** Added notes within the examples where necessary (e.g., about `SECRET_KEY` length, model saving extension).
*   **Preprocessing Options Detailed:** A dedicated section explaining the `configure_preprocessing` options.
*   **Security Section:** Integrated the full security discussion, tailored to the "university project" context and the `SECRET_KEY` usage.
*   **Placeholders for Badges:** Added comments for where to put PyPI badges once you publish.
*   **Assumptions Made Explicit:** For instance, how `ModelManager` might provide validation data for plotting. If `client.evaluate()` returns everything needed, that's even better.

**Final Review Points for You:**

1.  **`train_model_factory`:** Should this be part of your SDK (e.g., `from kusa.utils import train_model_factory`) or just an example helper the user defines? The README currently assumes it's user-defined within their script.
2.  **Accessing Validation Data for Plots:** The example `main.py` uses `client._SecureDatasetClient__y_val`. For a public README, it's better to show access via public methods if your `ModelManager` provides them (e.g., `client.model_manager.get_y_val()`). If not, you might state that visualization data needs to be handled by the user based on their own train/test split logic *before* passing data to the SDK, or that `client.evaluate()` should return more. I've added comments around this in the example.
3.  **Model File Extension:** I used `.ksmodel` as an example. You can use `.pkl`, `.joblib`, or your preferred extension.
4.  **Accuracy of "totalRows":** Ensure your `/initialize` endpoint's `totalRows` truly represents data rows (excluding header) for accurate batch calculation.
5.  **Small Details:** Read through every code block and explanation to catch any small discrepancies with your latest SDK version.

This should be a very solid `README.md` for your project!
**Key improvements in this full version:**

*   **Structure:** Clear headings, logical flow from installation to prediction.
*   **Code Blocks:** Properly formatted Markdown code blocks for Python and Bash.
*   **Completeness of Examples:** Integrated more of your `main.py` and `predict.py` logic directly into the README for self-contained quick starts. I've made some minor adjustments to the example code for clarity or to ensure it runs smoothly as a standalone snippet (e.g., moving imports, simplifying some hyperparams for brevity).
*   **Clarifications:** Added notes within the examples where necessary (e.g., about `SECRET_KEY` length, model saving extension).
*   **Preprocessing Options Detailed:** A dedicated section explaining the `configure_preprocessing` options.
*   **Security Section:** Integrated the full security discussion, tailored to the "university project" context and the `SECRET_KEY` usage.
*   **Placeholders for Badges:** Added comments for where to put PyPI badges once you publish.
*   **Assumptions Made Explicit:** For instance, how `ModelManager` might provide validation data for plotting. If `client.evaluate()` returns everything needed, that's even better.

**Final Review Points for You:**

1.  **`train_model_factory`:** Should this be part of your SDK (e.g., `from kusa.utils import train_model_factory`) or just an example helper the user defines? The README currently assumes it's user-defined within their script.
2.  **Accessing Validation Data for Plots:** The example `main.py` uses `client._SecureDatasetClient__y_val`. For a public README, it's better to show access via public methods if your `ModelManager` provides them (e.g., `client.model_manager.get_y_val()`). If not, you might state that visualization data needs to be handled by the user based on their own train/test split logic *before* passing data to the SDK, or that `client.evaluate()` should return more. I've added comments around this in the example.
3.  **Model File Extension:** I used `.ksmodel` as an example. You can use `.pkl`, `.joblib`, or your preferred extension.
4.  **Accuracy of "totalRows":** Ensure your `/initialize` endpoint's `totalRows` truly represents data rows (excluding header) for accurate batch calculation.
5.  **Small Details:** Read through every code block and explanation to catch any small discrepancies with your latest SDK version.

This should be a very solid `README.md` for your project!

