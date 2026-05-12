# Import Libraries and Modules
import joblib
import numpy as np
import pandas as pd

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide tensorflow warnings 
from tensorflow.keras.models import load_model

# Main model process
def run_model(model_name, model_path, pipe):
    """
    Loads model artifacts and continuously processes files
    received from the main server process
    """

    # Load model
    model = None
    model_file = None

    joblib_path = os.path.join(model_path, "final_model.joblib")
    keras_path = os.path.join(model_path, "final_model.keras")

    if os.path.exists(joblib_path):

        model = joblib.load(joblib_path)
        model_file = joblib_path

    elif os.path.exists(keras_path):

        model = load_model(keras_path)
        model_file = keras_path

    else:
        pipe.send("DOWN")
        return

    # Load artifacts
    features = load_optional_file(model_path, "features.joblib")
    scaler = load_optional_file(model_path, "scaler.joblib")
    pca = load_optional_file(model_path, "pca.joblib")

    # Notify server that model is ready
    pipe.send("UP")

    # Prediction loop
    while True:

        file_path = pipe.recv()

        if file_path == "EXIT":
            break


        DROP_COLS = [
            "Binary Label",
            "Attack Group",
            "Label",
            "Dataset"
        ]

        chunk_id = 0

        # Read file in chunks
        for chunk in pd.read_csv(
            file_path,
            chunksize=500_000
        ):

            # Feature selection
            if features is not None:
                chunk = chunk[features]

            # Drop non-feature columns
            chunk = chunk.drop(
                columns=DROP_COLS,
                errors="ignore"
            )

            # Apply preprocessing
            if scaler is not None:
                chunk = scaler.transform(chunk)

            if pca is not None:
                chunk = pca.transform(chunk)

            # Predict
            predictions = get_predictions(
                model,
                chunk,
                model_file
            )

            # Send predictions back to server
            pipe.send((chunk_id, predictions.tolist()))

            chunk_id += 1


        print(f"{model_name.upper()} finished file")

        pipe.send(("DONE", None))

# Prediction function
def get_predictions(model, data, model_file):
    """
    Generates prediction probabilities depending on model type
    """

    # LSTM Model
    if "lstm" in model_file.lower():

        window = 10

        # Pad first rows
        pad = np.repeat(data[:1], window - 1, axis=0)

        padded_data = np.vstack([pad, data])

        sequences = create_sequences(
            padded_data,
            window
        )

        return model.predict(
            sequences,
            batch_size=65536,
            verbose=0
        )
    
    # DNN Model
    elif model_file.endswith(".keras"):

        return model.predict(
            data,
            batch_size=8192,
            verbose=0
        )
    
    # RF Model
    else:

        return model.predict_proba(data)

def load_optional_file(directory, filename):
    """
    Loads optional model artifact if it exists
    """

    path = os.path.join(directory, filename)

    if os.path.exists(path):
        return joblib.load(path)

    return None

# LSTM SEQUENCE CREATION
def create_sequences(data, window):
    """
    Creates overlapping sliding windows for LSTM
    """

    shape = (
        data.shape[0] - window + 1,
        window,
        data.shape[1]
    )

    # Move one row at a time without copying data
    strides = (
        data.strides[0],
        data.strides[0],
        data.strides[1]
    )

    return np.lib.stride_tricks.as_strided(
        data,
        shape=shape,
        strides=strides
    )