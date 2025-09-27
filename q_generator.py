import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info.operators.base_operator import BaseOperator

import numpy as np

def data_loading_layer(num_data_points, tickers):
    """
    Creates a new QuantumCircuit with EfficientSU2 blocks for each ticker.

    Parameters:
        num_data_points : int
            Number of data points (controls block size).
        tickers : list[str]
            List of ticker symbols (used as parameter prefixes).

    Returns:
        QuantumCircuit
            A new quantum circuit with appended EfficientSU2 blocks.
    """
    num_stocks = len(tickers)

    # Determine group sizes
    if num_data_points <= 2:
        group_sizes = num_stocks * [1]
    else:
        group_size = int(np.ceil(num_data_points / 4))
        group_sizes = num_stocks * [group_size]

    # Total number of qubits needed
    total_qubits = sum(group_sizes)
    qc = QuantumCircuit(total_qubits)

    # Add EfficientSU2 blocks
    start = 0
    for ticker, size in zip(tickers, group_sizes):
        qubit_indices = list(range(start, start + size))
        su2_block = EfficientSU2(
            size,
            entanglement='circular', # can be changed to linear if needed
            reps=1,
            insert_barriers=True,
            parameter_prefix=ticker
        )
        qc.append(su2_block, qubit_indices)
        start += size

    return qc

def custom_parameterized_circuit(num_data_points, tickers,
                                 rotations=['rx', 'ry', 'rz'],
                                 inter_gate='cz', intra_gate='cz', reps=2):

    num_stocks = len(tickers)

    # Determine qubits per ticker
    if num_data_points <= 2:
        group_sizes = num_stocks * [1]
    else:
        group_size = int(np.ceil(num_data_points / 4))
        group_sizes = num_stocks * [group_size]

    total_qubits = sum(group_sizes)
    qc = QuantumCircuit(total_qubits)

    # Assign qubit indices per ticker
    start = 0
    block_indices = []
    for size in group_sizes:
        block_indices.append(list(range(start, start + size)))
        start += size

    # Helper to apply selected rotations
    def apply_rotations(qc, ticker, qubit, suffix):
        for rot in rotations:
            rot_lower = rot.lower()
            if rot_lower == 'rx':
                qc.rx(Parameter(f"{ticker}_rx{suffix}_q{qubit}"), qubit)
            elif rot_lower == 'ry':
                qc.ry(Parameter(f"{ticker}_ry{suffix}_q{qubit}"), qubit)
            elif rot_lower == 'rz':
                qc.rz(Parameter(f"{ticker}_rz{suffix}_q{qubit}"), qubit)
            else:
                raise ValueError("Rotation must be one of ['rx','ry','rz']")

    # ---- REPEATED LAYERS ----
    for rep in range(reps):

        # --- Rotations ---
        for ticker, qubits in zip(tickers, block_indices):
            for q in qubits:
                apply_rotations(qc, ticker, q, suffix=f"{rep}_a")

        # --- Inter-block entanglement ---
        for i in range(len(block_indices)):
            current_block = block_indices[i]
            next_block = block_indices[(i + 1) % num_stocks]
            q1 = current_block[-1]
            q2 = next_block[0]
            if inter_gate.lower() == 'cz':
                qc.cz(q1, q2)
            elif inter_gate.lower() == 'cx':
                qc.cx(q1, q2)

        # --- Intra-block entanglement + second rotations ---
        for ticker, qubits in zip(tickers, block_indices):
            if len(qubits) > 1:
                # Intra-block entanglement
                for i in range(len(qubits)-1):
                    q1 = qubits[i]
                    q2 = qubits[i+1]
                    if intra_gate.lower() == 'cz':
                        qc.cz(q1, q2)
                    elif intra_gate.lower() == 'cx':
                        qc.cx(q1, q2)

                # Second set of rotations
                for q in qubits:
                    apply_rotations(qc, ticker, q, suffix=f"{rep}_b")

    return qc



def alternate_parameterised_circuit(num_data_points, tickers):
    pass

def observables(num_qubits):
    """
    Constructs a measurement circuit for given observables.

    Parameters:
        num_qubits : int
            Total number of qubits in the circuit.
        observables : list[SparsePauliOp]
            List of observables to measure.
"""
    return [SparsePauliOp.from_list([(f"{'I'*i}Z{'I'*(num_qubits-i-1)}", 1)]) for i in range(num_qubits)]

def ancilla_qubits(num_ancilla): # Does not work yet
    """
    Creates a quantum circuit with specified number of ancilla qubits.

    Parameters:
        num_ancilla : int
            Number of ancilla qubits to include.

    Returns:
        QuantumCircuit
            A quantum circuit with the specified number of ancilla qubits.
    """
    qc_ancilla = QuantumCircuit(num_ancilla, name='ancilla')
    qc_ancilla.h(range(num_ancilla))
    return qc_ancilla


def entangle_ancilla(qc, ancilla_qc, block_indices): # Does not work yet

    qc.add_register(ancilla_qc.qregs[0])

    for block_idx, ancilla_idx in enumerate(range(ancilla_qc.num_qubits)):
        if block_idx >= len(block_indices):
            break
        data_qubits = block_indices[block_idx]
        ancilla_global = qc.num_qubits - ancilla_qc.num_qubits + ancilla_idx

        # Entangle first, middle, last qubit (if they exist)
        targets = [data_qubits[0]]
        if len(data_qubits) > 2:
            middle = data_qubits[len(data_qubits)//2]
            targets.append(middle)
        if len(data_qubits) > 1:
            targets.append(data_qubits[-1])

        for q in targets:
            qc.cx(ancilla_global, q)

    return qc

def forward(
    circuit: QuantumCircuit,
    input_params: np.ndarray,
    weight_params: np.ndarray,
    estimator: BaseEstimatorV2,
    observable: BaseOperator,
) -> np.ndarray:
    """
    Forward pass of the neural network.
 
    Args:
        circuit: circuit consisting of data loader gates and the neural network ansatz.
        input_params: data encoding parameters.
        weight_params: neural network ansatz parameters.
        estimator: EstimatorV2 primitive.
        observable: a single observable to compute the expectation over.
 
    Returns:
        expectation_values: an array (for one observable) or a matrix (for a sequence of observables) of expectation values.
        Rows correspond to observables and columns to data samples.
    """
    num_samples = input_params.shape[0]
    weights = np.broadcast_to(weight_params, (num_samples, len(weight_params)))
    params = np.concatenate((input_params, weights), axis=1)
    pub = (circuit, observable, params)
    job = estimator.run([pub])
    result = job.result()[0]
    expectation_values = result.data.evs
 
    return expectation_values

def split_parameters(params):
    """
    Splits Qiskit ParameterVectorElements into input params and weight params
    based on '_' in their names.
    """
    input_params = []
    weight_params = []
    
    for p in params:
        if "_" in p.name:  # weight parameters
            weight_params.append(p)
        else:  # input/data parameters
            input_params.append(p)
    
    return input_params, weight_params


def two_qubit_circuit_tickers(tickers):
    """
    Creates a quantum circuit with 2 qubits per ticker.
    """
    num_stocks = len(tickers)
    total_qubits = num_stocks * 2
    qc = QuantumCircuit(total_qubits)
    data_points = 8
    group_sizes = num_stocks * [2]
    qc.compose(data_loading_layer(data_points, tickers), inplace=True)
    qc.compose(custom_parameterized_circuit(data_points, tickers,
                                           rotations=['rx', 'ry', 'rz'],
                                           inter_gate='cz',
                                           intra_gate='cz', reps=2), inplace=True)
    # Assign qubit indices per ticker

    return qc
            