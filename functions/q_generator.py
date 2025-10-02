import qiskit
from qiskit import QuantumCircuit
#from qiskit.circuit.library import Parameter
from qiskit.circuit import ParameterVector
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import BaseEstimatorV2
from qiskit import transpile


from qiskit.quantum_info.operators.base_operator import BaseOperator

import numpy as np


def data_loading_layer(num_data_points, tickers):
    """
    Creates EfficientSU2 blocks for each ticker, but renames
    the data parameters to x0, x1, x2 ... sequentially across tickers.
    """
    num_stocks = len(tickers)

    # Determine group sizes
    if num_data_points <= 2:
        group_sizes = num_stocks * [1]
    else:
        group_size = int(np.ceil(num_data_points / 4))
        group_sizes = num_stocks * [group_size]

    total_qubits = sum(group_sizes)
    qc = QuantumCircuit(total_qubits)

    start = 0
    global_param_idx = 0

    for ticker, size in zip(tickers, group_sizes):
        qubits = list(range(start, start + size))

        # Create SU2 block
        su2_block = EfficientSU2(
            num_qubits=size,
            entanglement='circular',
            reps=1,
            insert_barriers=False,
            parameter_prefix=ticker  # we will rename parameters next
        )

        # Rename parameters sequentially
        param_map = {}
        for old_param in su2_block.parameters:
            param_map[old_param] = Parameter(f"A{num_to_alpha3(global_param_idx)}")
            global_param_idx += 1

        su2_block = su2_block.assign_parameters(param_map, inplace=False)

        qc.append(su2_block, qubits)
        start += size

    return qc

def num_to_alpha3(n: int) -> str:
    """Convert an integer to a 3-letter base-26 string using A-Z."""
    if n < 0 or n >= 26**3:
        raise ValueError("Number must be between 0 and 17575 (inclusive) for 3 alphabet digits")

    a = n // (26 * 26)
    b = (n // 26) % 26
    c = n % 26

    return chr(65 + a) + chr(65 + b) + chr(65 + c)


def custom_parameterized_circuit(group_sizes, tickers, reps=2, entanglement_maps=None):
    if entanglement_maps is None:
        entanglement_maps = 'circular'
    num_stocks = len(tickers)

    block_maps = []
    for entanglement_map in entanglement_maps:
        block_maps.append([])
        for entangly in entanglement_map:
            block_maps[-1].append([tickers.index(entangly[0]),tickers.index(entangly[1])])

    total_qubits = sum(group_sizes)
    qc = QuantumCircuit(total_qubits)

    # Assign qubit indices per ticker
    start = 0
    block_indices = []
    for size in group_sizes:
        block_indices.append(list(range(start, start+size)))
        start += size

    theta = ParameterVector("theta0",3*group_sizes[0])

    for i in range(group_sizes[0]):
        for t in range(len(tickers)):
            qc.rx(theta[0+i*3],block_indices[t][i])
            qc.ry(theta[1+i*3],block_indices[t][i])
            qc.rz(theta[2+i*3],block_indices[t][i])

    i = 0
    #entanglement only for intra ticker
    for ticker, size in zip(tickers, group_sizes):
        qubits = block_indices[i]
        i=i+1
        entangle_qc = QuantumCircuit(len(qubits))
        qmap= [(j, j+1) for j in range(len(qubits)-1)]
        if size>2:
            qmap.append([(len(qubits)-1, 0)])
        for a,b in qmap:
            entangle_qc.cz(a,b)
        qc.compose(entangle_qc, qubits,inplace=True)

    theta1 = ParameterVector("theta1",2*total_qubits)

    for i in range(total_qubits):
            qc.ry(theta1[2*i],i)
            qc.rz(theta1[2*i+1],i)
    
        #entanglement only for intra ticker
    for i,block_map in enumerate(block_maps):
        qubits = [item[i] for item in block_indices]

        entangle_qc = QuantumCircuit(len(qubits))
        for a,b in block_map:
            entangle_qc.cz(a,b)
        qc.compose(entangle_qc, qubits,inplace=True)
    
    theta2 = ParameterVector("theta2",2*total_qubits)

    for i in range(total_qubits):
            qc.ry(theta2[2*i],i)
            qc.rz(theta2[2*i+1],i)

    return qc




def big_su2_circuit(total_qubits, reps=1, entanglement='linear'):
    """
    Build a single big EfficientSU2 circuit over all qubits.

    Args:
        total_qubits (int): total number of qubits in the circuit
        reps (int): number of repetitions of the SU2 layers
        entanglement (str): type of entanglement ('linear' or 'circular')

    Returns:
        QuantumCircuit: parameterized SU2 circuit
    """
    qc = QuantumCircuit(total_qubits)

    # Create single big EfficientSU2
    su2_block = EfficientSU2(
        num_qubits=total_qubits,
        entanglement=entanglement,
        reps=reps,
        insert_barriers=False
    )

    # Rename parameters sequentially as W0, W1, ...
    param_map = {old: Parameter(f"W{num_to_alpha3(i)}") for i, old in enumerate(su2_block.parameters)}
    su2_block = su2_block.assign_parameters(param_map, inplace=False)

    # Append the SU2 to the main circuit
    qc.append(su2_block, range(total_qubits))

    return qc



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
    #qc.compose(big_su2_circuit(total_qubits, reps=2), inplace=True)
    #qc.compose(custom_parameterized_circuit(data_points, tickers,
    #                                       rotations=['rx', 'ry', 'rz'],
    #                                       inter_gate='cz',
    #                                       intra_gate='cz', reps=1), inplace=True)
    # Assign qubit indices per ticker
    #qc.measure_all()
    
    return qc
            