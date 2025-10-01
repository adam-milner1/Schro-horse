import os
import json
from functions.data_preperation import process_model_data
from functions.q_generator import two_qubit_circuit_tickers
from sklearn.metrics import mean_squared_error
from qiskit_ibm_runtime import QiskitRuntimeService#, EstimatorV2
#from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.backends import AerSimulator as GenericBackendV2
from qiskit import transpile
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

# Use Aer simulator for local estimation
from qiskit_aer.primitives import EstimatorV2

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
    
    # Fix ordering
    sorted_indices = np.argsort(epochs)
    epochs = [epochs[i] for i in sorted_indices]
    for k in all_metrics:
        all_metrics[k] = [all_metrics[k][i] for i in sorted_indices]

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

def generate_data(generator_weights, qc, feature_data, target_data, backend=None, sampling=True, sample_size=100):
    # --- Sampling subset if required ---
    if sampling:
        n_samples = feature_data.shape[0]
        indices = np.random.choice(n_samples, size=sample_size, replace=False)

        feature_data = feature_data[indices]
        target_data = np.array(target_data)[indices]
    else:
        target_data = np.array(target_data)

    # --- Assign generator weights to the circuit ---
    data_params, weight_params = infer_parameter_groups(qc)
    weight_param_dict = {param: val for param, val in zip(weight_params, generator_weights)}
    #qc_weighted = qc.assign_parameters(weight_param_dict)

    # --- Build Z observables ---
    num_qubits = qc.num_qubits
    observables = [SparsePauliOp.from_list([(f"{'I'*i}Z{'I'*(num_qubits-i-1)}", 1)]) for i in range(num_qubits)]

    # --- Backend and Estimator ---
    backend = backend if backend else GenericBackendV2()#(num_qubits=num_qubits)
    estimator = EstimatorV2()#(mode=backend)
    

    # --- Transpile circuit ---
    qc_weighted_transpiled = transpile(qc, backend=backend, optimization_level=3)

    # --- Prepare batched parameters ---
    n_samples = feature_data.shape[0]
    batched_params = np.hstack([feature_data, np.tile(generator_weights, (n_samples, 1))])

    # --- Build pubs for batch execution ---
    pubs = [(qc_weighted_transpiled, observables, batched_params[i]) for i in range(n_samples)]

    # --- Run the estimator once ---
    result = estimator.run(pubs).result()

    # --- Collect expectation values ---
    outputs = np.array([res.data.evs for res in result], dtype=np.float32) * np.pi

    return outputs, target_data



def calculate_rmse(generated_data, target_data, ticker_labels, target_labels):
    # Calculate RMSE for each ticker and output feature
    rmse_per_feature = []
    for i in range(generated_data.shape[1]):
        rmse_i = np.sqrt(mean_squared_error(target_data[:, i], generated_data[:, i]))
        rmse_per_feature.append(rmse_i)

    feature_labels = [f"{ticker}_{target}" for ticker in ticker_labels for target in target_labels]
    rmse_dict = {label: rmse for label, rmse in zip(feature_labels, rmse_per_feature)}
    
    return rmse_dict


#plot rmse per epoch to see if its going down
def plot_rmse_per_epoch(model_path, qc, feature_data, target_data, target_labels, ticker_labels, every_n_epochs=1, backend= None, sampling=True, sample_size=100):
    weights_files = sorted(
        [f for f in os.listdir(f"{model_path}/logs") if f.startswith("generator_weights") and f.endswith(".npy")],
        key=lambda x: int(x.split("epoch")[1].split(".")[0])
    )
    
    epochs=[]
    weights_list=[]

    for file in weights_files:
        epoch_num = int(file.split("epoch")[1].split(".")[0])
        if epoch_num % every_n_epochs == 0:
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
        outputs, target_outputs = generate_data(
            generator_weights = loaded_weights,
            qc = qc,
            feature_data = feature_data,
            target_data = target_data,
            backend = backend,
            sampling = sampling,
            sample_size = sample_size,
        )

        rmse_dict = calculate_rmse(
            generated_data= outputs,
            target_data= target_outputs, 
            ticker_labels= ticker_labels,
            target_labels = target_labels
        )

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

def compare_tickers_scatter(df, ticker1, ticker2,title, feature="OC_next"):
    """
    Make a scatter plot comparing the chosen feature (OC_next or CO_next)
    between two tickers using a wide-format DataFrame.
    
    Parameters:
        df (pd.DataFrame): Wide-format DataFrame with columns like 'GOOG_OC_next'
        ticker1 (str): First ticker symbol
        ticker2 (str): Second ticker symbol
        feature (str): Feature to compare ('OC_next' or 'CO_next')
    """
    col1 = f"{ticker1}_{feature}"
    col2 = f"{ticker2}_{feature}"
    
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns {col1} or {col2} not found in DataFrame")
    
    plt.figure(figsize=(6,6))
    plt.scatter(df[col1], df[col2], alpha=0.7)
    plt.xlabel(f"{ticker1} {feature}")
    plt.ylabel(f"{ticker2} {feature}")
    plt.title(title)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.grid(True)
    plt.show()