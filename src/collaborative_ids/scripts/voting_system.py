# Import Libraries and Modules
import os
import time
import numpy as np
import pandas as pd

# Configurations
PREDICTIONS_DIR = "predictions/"
INPUT_FILE = os.path.join(PREDICTIONS_DIR, "predictions.csv")
OUTPUT_FILE = os.path.join(PREDICTIONS_DIR, "predictions_final.csv")
CHUNK_SIZE = 500_000

# Model columns and weights
model_cols = ["LSTM", "DNN", "RF"]

weights = {
    "Benign":         {"RF":0.5, "DNN":0.167, "LSTM":0.333},
    "Botnet":         {"RF":0.5, "DNN":0.167, "LSTM":0.333},
    "BruteForce":     {"RF":0.5, "DNN":0.167, "LSTM":0.333},
    "DDoS":           {"RF":0.5, "DNN":0.167, "LSTM":0.333},
    "DoS":            {"RF":0.5, "DNN":0.333, "LSTM":0.167},
    "Infiltration":   {"RF":0.333, "DNN":0.167, "LSTM":0.5},
    "Other":          {"RF":0.167, "DNN":0.333, "LSTM":0.5},
    "Reconnaissance": {"RF":0.5, "DNN":0.167, "LSTM":0.333}
}

classes = list(weights.keys())

# Prepare output file with header
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

header = ["Index", "Actual"] + model_cols + ["Majority", "Weighted"]
with open(OUTPUT_FILE, "w") as f:
    f.write(",".join(header) + "\n")

# Performance tracking
global_index = 0
total_processing_time = 0

# Process predictions file in chunks
for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    start_time = time.time()
    
    # Store final outputs
    weighted_scores = [] 
    majority_scores = [] 
    rf_final = [] 
    dnn_final = [] 
    lstm_final = []

    # Process each prediction row
    for i in range(len(chunk)): 
        rf = np.fromstring(str(chunk["RF"].iloc[i]).strip('"[]"'), sep=',') 
        dnn = np.fromstring(str(chunk["DNN"].iloc[i]).strip('"[]"'), sep=',') 
        lstm = np.fromstring(str(chunk["LSTM"].iloc[i]).strip('"[]"'), sep=',')

        # Combined weighted probability vector
        combined = np.zeros_like(rf)

        # Weighted ensemble voting
        # V = sum(weight * probability)
        for class_idx, class_name in enumerate(classes):
            combined[class_idx] = ( 
                weights[class_name]["RF"] * rf[class_idx] + 
                weights[class_name]["DNN"] * dnn[class_idx] + 
                weights[class_name]["LSTM"] * lstm[class_idx] 
            )

        # Final weighted prediction
        weighted_scores.append(np.argmax(combined))

        # Individual model predictions
        rf_pred = np.argmax(rf) 
        dnn_pred = np.argmax(dnn) 
        lstm_pred = np.argmax(lstm) 
        
        rf_final.append(rf_pred) 
        dnn_final.append(dnn_pred) 
        lstm_final.append(lstm_pred)

        # Majority voting
        votes = [rf_pred, dnn_pred, lstm_pred] 
        majority_class = max(set(votes), key=votes.count) 
        majority_scores.append(majority_class)

    # Convert prediction indices to class names
    chunk["Weighted"] = [classes[i] for i in weighted_scores] 
    chunk["Majority"] = [classes[i] for i in majority_scores] 
    
    # Replace model probabilities with final predicted class label
    chunk["RF"] = [classes[i] for i in rf_final] 
    chunk["DNN"] = [classes[i] for i in dnn_final] 
    chunk["LSTM"] = [classes[i] for i in lstm_final]

    end_time = time.time()
    total_processing_time += end_time - start_time
    print(f"Time taken to process chunk: {end_time - start_time}")

    # Write chunk to CSV 
    chunk["Index"] = range(global_index, global_index + len(chunk)) 
    global_index += len(chunk) 
    output_cols = ["Index", "Actual"] + model_cols + ["Majority", "Weighted"] 
    chunk[output_cols].to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

print(f"Finished writing predictions to {OUTPUT_FILE}")
print(f"Time taken to process: {total_processing_time}")

# Calculate latency and throughput
avg_latency_sec = total_processing_time / global_index # seconds per row
avg_latency_us = avg_latency_sec * 1_000_000 # microseconds per row
throughput = global_index / total_processing_time # rows per second

print(f"\nTotal rows processed: {global_index:,}")
print(f"Average latency per row: {avg_latency_us:.2f} μs")
print(f"Throughput: {throughput:,.2f} rows/sec")