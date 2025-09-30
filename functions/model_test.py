import os
import json
from functions.data_preperation import process_model_data
from functions.q_generator import two_qubit_circuit_tickers
from sklearn.metrics import mean_squared_error
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import transpile
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

def get_latest_model_path(model_dir):
    files = os.listdir(model_dir)
    paths = [os.path.join(model_dir, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def plot_training_metrics(save_dir, generator_loss_scaling =1):
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
        if key in ["d_loss", "g_loss"]:
            if key == "g_loss":
                values = [v * generator_loss_scaling for v in values]
            
                plt.plot(epochs, values, label=f"{key}*{generator_loss_scaling}")
            else:
                plt.plot(epochs, values, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def infer_parameter_groups(circuit):
        """
        Basic heuristic: names starting with x/data -> data params,
        names starting with w/theta/th -> trainable params.
        """
        data_prefixes = ("x", "data")
        weight_prefixes = ("w", "theta", "th")
        data_params, weight_params, other = [], [], []

        for p in circuit.parameters:
            name = p.name.lower()
            if name.startswith(data_prefixes):
                data_params.append(p)
            elif name.startswith(weight_prefixes):
                weight_params.append(p)
            else:
                other.append(p)

        # If ambiguous, put leftovers into weight_params by default
        weight_params += other
        return data_params, weight_params


def calculate_rmse(generator_weights, targets, tickers, features, sample_size = 100):
    # Load QGenerator circuit
    qc= two_qubit_circuit_tickers(tickers)

    # Load data
    feature_data, target_data = process_model_data(targets, features, tickers)

    n_samples = feature_data.shape[0]
    indices = np.random.choice(n_samples, size=sample_size, replace=False)

    # Select the rows
    target_data= np.array(target_data)
    feature_data = feature_data[indices]
    target_data = target_data[indices]

    # Assign weights to the circuit
    data_params, weight_params = infer_parameter_groups(qc)
    weight_param_dict = {param: val for param, val in zip(weight_params, generator_weights)}
    qc_weighted = qc.assign_parameters(weight_param_dict)

    # Get observables
    num_qubits = qc_weighted.num_qubits
    observables = [SparsePauliOp.from_list([(f"{'I'*i}Z{'I'*(num_qubits-i-1)}", 1)]) for i in range(num_qubits)]

    #service = QiskitRuntimeService()
    #backend = service.least_busy()
    backend = GenericBackendV2(num_qubits=num_qubits)
    estimator = EstimatorV2(mode=backend)

    # Transpile circuit for backend
    qc_weighted_transpiled = transpile(qc_weighted, backend=backend, optimization_level=3)

    # Run the inputs through the circuit
    #change this to batch?
    outputs =[]
    for input_data in tqdm(feature_data, desc="Processing inputs"):
        data_param_dict = {param: val for param, val in zip(data_params, input_data)}
        
        # Assign data encodings to the circuit
        qc_data = qc_weighted_transpiled .assign_parameters(data_param_dict) 

        pub = (qc_data, observables) #primitive unified bloc program input for estimator
        job = estimator.run([pub])
        result = job.result()[0]
        
        expectation_values = result.data.evs
        outputs.append(expectation_values)

    outputs = np.array(outputs)          # shape: (n_samples, n_features)
    target_data_np = np.array(target_data)  

    # Calculate RMSE for each ticker and output feature
    rmse_per_feature = []
    for i in range(outputs.shape[1]):
        rmse_i = np.sqrt(mean_squared_error(target_data_np[:, i], outputs[:, i]))
        rmse_per_feature.append(rmse_i)

    feature_labels = [f"{ticker}_{target}" for ticker in tickers for target in targets]
    rmse_dict = {label: rmse for label, rmse in zip(feature_labels, rmse_per_feature)}
    for k, v in rmse_dict.items():
        print(f"{k}: {v:.4f}")
    
    return rmse_dict


#plot rmse per epoch to see if its going down
def plot_rmse_per_epoch(model_path, targets, tickers, features, sample_size=100):
    weights_files = sorted([f for f in os.listdir(f"{model_path}/logs") if f.startswith("generator_weights") and f.endswith(".npy")])
    epochs=[]
    weights_list=[]

    for file in weights_files:
        epoch_num = int(file.split("epoch")[1].split(".")[0])
        epochs.append(epoch_num)
        path = os.path.join(f"{model_path}/logs", file)
        weights = np.load(path)
        weights_list.append(weights)
    
    weights_dict= {
        "weights": weights_list,
        "epochs": epochs,
    }
    
    all_rmse = []  # list of dicts

    for loaded_weights in weights_dict["weights"]:
        rmse_dict = calculate_rmse(loaded_weights, targets, tickers, features, sample_size)
        all_rmse.append(rmse_dict)

    rmse_over_epochs = {key: [] for key in all_rmse[0].keys()}

    for rmse_dict in all_rmse:
        for key, value in rmse_dict.items():
            rmse_over_epochs[key].append(value)
    
    rmse_over_epochs["epochs"] = weights_dict["epochs"]

    for feature, values in rmse_over_epochs.items():
        if feature == "epochs":
            continue
        plt.plot(rmse_over_epochs["epochs"], values, label=feature)

    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()
    return rmse_over_epochs