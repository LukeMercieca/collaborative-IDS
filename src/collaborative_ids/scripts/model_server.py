# Import Libraries and Modules
import os
import numpy as np
import pandas as pd

from multiprocessing import Process, Pipe
from multiprocessing.connection import wait

from run_model import run_model

# Configuration
MODELS_DIR = "models/"
PREDICTIONS_DIR = "predictions/"

OUTPUT_FILE = os.path.join(PREDICTIONS_DIR, "predictions.csv")
LABEL_COL = "Attack Group"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Discover Available Models
MODELS = {}

for model_name in os.listdir(MODELS_DIR):
    model_path = os.path.join(MODELS_DIR, model_name)
    MODELS[model_name] = model_path

# Start Model Processes
model_processes = {}
connections = {}

print("Loading models...")

for model_name, model_path in MODELS.items():

    # Create communication pipe
    parent_conn, child_conn = Pipe()

    # Start model process
    process = Process(
        target=run_model,
        args=(model_name, model_path, child_conn)
    )

    process.start()

    # Store process + connection
    model_processes[model_name] = process
    connections[model_name] = parent_conn

# Verify Models Loaded Correctly
failed_models = []

for model_name, conn in connections.items():

    status = conn.recv()

    print(f"STATUS: {model_name.upper()} model is {status}")

    if status == "DOWN":
        failed_models.append(model_name)

# Remove failed models
for model_name in failed_models:

    model_processes[model_name].terminate()
    model_processes[model_name].join()

    connections[model_name].close()

    del model_processes[model_name]
    del connections[model_name]
    del MODELS[model_name]

# Main Loop
try:
    while True:
        
        # Get input file
        file_to_process = input("\nEnter file path (or 'EXIT'): ")

        if file_to_process.lower() == "exit":
            break

        if not os.path.exists(file_to_process):
            print("ERROR: File not found")
            continue


        # Send file to all active models
        print("Sending file to models...")

        for conn in connections.values():
            conn.send(file_to_process)

        # Prepare output file
        model_names = list(connections.keys())

        with open(OUTPUT_FILE, "w") as f:

            header = (
                "Index,Actual,"
                + ",".join(name.upper() for name in model_names)
            )

            f.write(header + "\n")

        # Process actual label
        label_reader = pd.read_csv(
            file_to_process,
            usecols=[LABEL_COL],
            chunksize=500_000
        )

        chunk_buffers = {name: {} for name in model_names}
        label_chunks = {}

        active_connections = set(connections.values())
        conn_to_name = {
            conn: name
            for name, conn in connections.items()
        }

        global_index = 0

        print("Receiving predictions...")

        # Receive predictions from models
        while active_connections:

            # Wait for any model to send data
            ready_connections = wait(active_connections)

            for conn in ready_connections:

                model_name = conn_to_name[conn]

                msg = conn.recv()

                # Model finished
                if msg[0] == "DONE":

                    print(f"{model_name.upper()} finished")

                    active_connections.remove(conn)
                    continue

                # Receive Chunk Predictions
                chunk_id, predictions = msg

                print(f"{model_name.upper()} sent chunk {chunk_id}")

                chunk_buffers[model_name][chunk_id] = predictions

                # Load label chunk if required
                if chunk_id not in label_chunks:

                    label_chunks[chunk_id] = (
                        next(label_reader)[LABEL_COL].tolist()
                    )

                # Write chunk when all models are ready
                if all(chunk_id in chunk_buffers[m] for m in model_names):

                    # Get predictions from all models
                    preds_per_model = [
                        chunk_buffers[m].pop(chunk_id)
                        for m in model_names
                    ]

                    labels = label_chunks.pop(chunk_id)

                    # Ensure equal chunk sizes
                    min_len = min(
                        len(labels),
                        *[len(p) for p in preds_per_model]
                    )

                    # Append Predictions To CSV
                    with open(OUTPUT_FILE, "a") as f:

                        for i in range(min_len):

                            formatted_models = []

                            # Format probabilities
                            for preds in preds_per_model:

                                vec = np.array(
                                    preds[i],
                                    dtype=np.float64
                                )

                                vec = vec / vec.sum()

                                formatted = (
                                    '"[' +
                                    ", ".join(
                                        f"{x:.6f}" for x in vec
                                    ) +
                                    ']"'
                                )

                                formatted_models.append(formatted)

                            # Write row
                            f.write(
                                f"{global_index},{labels[i]},"
                                + ",".join(formatted_models)
                                + "\n"
                            )

                            global_index += 1

                    print(f"Written chunk {chunk_id}")


        print(
            f"\nFinished writing "
            f"{global_index:,} rows to {OUTPUT_FILE}"
        )

# Shutdown
finally:

    print("Shutting down models...")

    for conn in connections.values():
        conn.send("EXIT")
        conn.close()

    for process in model_processes.values():
        process.terminate()
        process.join()