import os
from tensorflow.python.summary.summary_iterator import summary_iterator
import struct


def get_latest_model_path(model_dir):
    files = os.listdir(model_dir)
    paths = [os.path.join(model_dir, basename) for basename in files]
    return max(paths, key=os.path.getctime)



import os
import json
import matplotlib.pyplot as plt

def plot_training_metrics(save_dir):
    """
    Reads JSON metric files saved after each epoch and plots them.
    
    Parameters
    ----------
    save_dir : str
        Directory containing the metrics JSON files.
    """
    # Collect all metric files
    metric_files = sorted([f for f in os.listdir(save_dir) if f.startswith("metrics_epoch") and f.endswith(".json")])
    if not metric_files:
        print(f"No metric files found in {save_dir}")
        return

    # Aggregate metrics
    all_metrics = {}
    epochs = []

    for file in metric_files:
        epoch_num = int(file.split("epoch")[1].split(".")[0])
        epochs.append(epoch_num)
        path = os.path.join(save_dir, file)
        with open(path, "r") as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            all_metrics.setdefault(key, []).append(value)

    # Plot each metric
    plt.figure(figsize=(10, 6))
    for key, values in all_metrics.items():
        plt.plot(epochs, values, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

