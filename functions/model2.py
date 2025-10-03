# =========================
# model2.py
# =========================
# Pure local QGAN with Qiskit + TensorFlow
# Generator is a Qiskit circuit optimized with COBYLA
# Discriminator is a classical Keras network
# =========================

# === Dependencies ===
# pip install qiskit==2.* qiskit-aer scipy tensorflow
import numpy as np
import tensorflow as tf

print(tf.__version__)  # should print something like 2.14.x
print("GPU devices:", tf.config.list_physical_devices("GPU"))

from tensorflow.keras import layers, models, metrics
from scipy.optimize import minimize

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.batch import Batch
# from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2  # Uncomment if using IBM hardware
from qiskit_aer.backends import AerSimulator

# Use Aer simulator for local estimation
from qiskit_aer.primitives import EstimatorV2 as BackendEstimator


# ==========================
# QGenerator (local)
# ==========================
class QGenerator:
    """
    Pure-Qiskit generator wrapper using the Estimator primitive.
    Produces <Z> on each qubit as the generator output.
    """

    def __init__(
        self,
        qiskit_circuit: QuantumCircuit,
        batch_size: int,
        on_hardware: bool = False,
        data_params=None,  # Optional: list[Parameter] for data embedding
        weight_params=None,  # Optional: list[Parameter] for trainable weights
        estimator=None,  # Optional external Estimator; defaults to Aer simulator
        backend=None,  # Optional backend for Batch context
    ):
        self.on_hardware = on_hardware
        self.qiskit_circuit = qiskit_circuit
        self.num_qubits = qiskit_circuit.num_qubits
        self.batch_size = batch_size

        # Parameter partitioning
        if data_params is None or weight_params is None:
            data_params, weight_params = self._infer_parameter_groups(qiskit_circuit)
        self.data_params = list(data_params)
        self.weight_params = list(weight_params)

        # Cache Z observables for each qubit
        self._z_ops = self._build_z_ops(self.num_qubits)

        # Estimator primitive (default to local Aer backend)
        self.estimator = estimator if estimator is not None else BackendEstimator()
        self.backend = backend if backend is not None else AerSimulator()

    @staticmethod
    def _infer_parameter_groups(circuit):
        """
        Basic heuristic: names starting with x/data -> data params,
        names starting with w/theta/th -> trainable params.
        """
        data_prefixe = "A"
        weight_prefixe = "W"
        data_params, weight_params, other = [], [], []

        for p in circuit.parameters:
            name = p.name
            if name[0] == data_prefixe:
                data_params.append(p)
            elif name[0] == weight_prefixe:
                weight_params.append(p)
            else:
                other.append(p)

        # If ambiguous, put leftovers into weight_params by default
        weight_params += other
        return data_params, weight_params

    @staticmethod
    def _build_z_ops(nq: int):
        labels = []
        observables = []
        for i in range(nq):
            label = ["I"] * nq
            label[i] = "Z"
            labels.append("".join(label))
        for label in labels:
            observables.append(SparsePauliOp.from_list([(label, 1.0)]))
        return observables

    def _bind(self, data_vec: np.ndarray, weight_vec: np.ndarray):
        bind = {}
        if len(self.data_params) != len(data_vec):
            raise ValueError(f"Expected {len(self.data_params)} data params, got {len(data_vec)}")
        if len(self.weight_params) != len(weight_vec):
            raise ValueError(f"Expected {len(self.weight_params)} weight params, got {len(weight_vec)}")
        for p, v in zip(self.data_params, data_vec):
            bind[p] = float(v)
        for p, v in zip(self.weight_params, weight_vec):
            bind[p] = float(v)
        return bind
    
    def optimise_circuit_best(self, qc: QuantumCircuit, backend=None, trials: int = 5):
        """
        Transpile the quantum circuit multiple times and pick the one with the lowest 2-qubit depth.
        
        Parameters:
            qc : QuantumCircuit
                The quantum circuit to optimize.
            backend : Optional[BaseBackend]
                The backend to transpile for (None uses default simulator).
            trials : int
                Number of transpilation trials.
        
        Returns:
            best_qc : QuantumCircuit
                Transpiled circuit with lowest 2-qubit gate depth.
            best_2q_depth : int
                Depth of 2-qubit gates of the best circuit.
        """
        best_qc = None
        best_2q_depth = float('inf')

        for _ in range(trials):
            transpiled = transpile(qc, backend=self.backend, optimization_level=3)

            # Filter 2-qubit gates using a lambda
            two_qubit_gates = list(filter(lambda inst: inst[0].num_qubits == 2, transpiled.data))
            
            # Create a temporary circuit with only 2-qubit gates to compute depth
            temp_qc = QuantumCircuit(transpiled.num_qubits)
            for inst, qargs, cargs in two_qubit_gates:
                temp_qc.append(inst, qargs, cargs)
            
            two_q_depth = temp_qc.depth()

            print(f"Trial 2Q Depth: {two_q_depth}")
            if two_q_depth < best_2q_depth:
                best_2q_depth = two_q_depth
                best_qc = transpiled

        return best_qc, best_2q_depth
    
    def run_single(self, data_vec: np.ndarray, weight_vec: np.ndarray) -> np.ndarray:
        bindings = self._bind(data_vec, weight_vec)
        bound = self.qiskit_circuit.assign_parameters(bindings)
        values = []
        for obs in self._z_ops:
            res = self.estimator.run(bound, obs).result()
            values.append(res.values[0])
        return np.asarray(values, dtype=np.float32)
    
    def run_batch(self, data_batch: np.ndarray, weight_batch: np.ndarray) -> np.ndarray:
        if data_batch.shape[0] != weight_batch.shape[0]:
            raise ValueError("Mismatched batch sizes")

        # Vectorized parameter concatenation
        batched_params = np.hstack([data_batch, weight_batch])

        # Build pubs
        pubs = [(self.qiskit_circuit, self._z_ops, batched_params[i]) for i in range(data_batch.shape[0])]

        # Run estimator once
        result = self.estimator.run(pubs).result()

        # Convert to array
        results = np.array([res.data.evs for res in result], dtype=np.float32)
        return results*np.pi


    def __call__(self, data_batch, weights_batch):
        data_np = np.asarray(data_batch, dtype=np.float32)
        w_np = np.asarray(weights_batch, dtype=np.float32)
        outs = self.run_batch(data_np, w_np)
        return tf.convert_to_tensor(outs, dtype=tf.float32)


# ==========================
# Discriminator (Keras)
# ==========================
def make_discriminator(input_size, use_bias=True, multiplier=1):
    dis_input = layers.Input(shape=(input_size,), name="discriminator_input")
    unit_size = input_size * multiplier
    x = layers.Dense(units=unit_size, use_bias=use_bias)(dis_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    unit_size = unit_size // 2
    while unit_size > 1:
        x = layers.Dense(units=unit_size, use_bias=use_bias)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        unit_size = unit_size // 2
    dis_output = layers.Dense(units=1, use_bias=True)(x)
    return models.Model(dis_input, dis_output, name="discriminator")


# ==========================
# GAN with COBYLA generator
# ==========================
class GAN(models.Model):
    """
    WGAN-GP where the generator is a Qiskit circuit optimized with COBYLA,
    and the discriminator is a classical Keras network trained with Adam.
    """

    def __init__(
        self,
        discriminator_steps,
        gp_weight,
        n_tickers,
        n_features,
        n_outputs,
        generator_qiskit,  # QuantumCircuit
        discriminator_layer_multiplier=1,
        on_hardware=False,
        batch_size=64,
        data_params=None,
        weight_params=None,
        estimator=None,
    ):
        super().__init__()
        self.discriminator_steps = discriminator_steps
        self.gp_weight = gp_weight
        self.batch_size = batch_size
        self.n_tickers = n_tickers
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.latent_dim = n_features * n_tickers
        self.num_qubits = generator_qiskit.num_qubits

        self.generator = QGenerator(
            generator_qiskit,
            batch_size=batch_size,
            on_hardware=on_hardware,
            data_params=data_params,
            weight_params=weight_params,
            estimator=estimator,
        )
        self.generator.qiskit_circuit,_ = self.generator.optimise_circuit_best(self.generator.qiskit_circuit, backend=self.generator.backend, trials=1)

        self.discriminator = make_discriminator(
            self.latent_dim + self.num_qubits,
            multiplier=discriminator_layer_multiplier,
        )

        if weight_params is None:
            num_generator_params = len(self.generator.weight_params)
        else:
            num_generator_params = len(weight_params)

        self.generator_weights = [
            tf.Variable(tf.random.normal([]), trainable=False)
            for _ in range(num_generator_params)
        ]

        # Metrics
        self.d_wass_loss_metric = metrics.Mean(name="d_wass_loss")
        self.d_gp_metric = metrics.Mean(name="d_gp")
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")
        self.d_real_metric = metrics.Mean(name="real_score")
        self.d_gen_metric = metrics.Mean(name="gen_score")

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.g_loss_metric,
            self.d_gp_metric,
            self.d_wass_loss_metric,
            self.d_real_metric,
            self.d_gen_metric,
        ]

    def compile(self, d_optimizer, **kwargs):
        super().compile()
        self.d_optimizer = d_optimizer

    # ---------- WGAN-GP helpers ----------
    def gradient_penalty(self, real_data, fake_data):
        alpha = tf.random.normal([self.batch_size, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    # ---------- COBYLA objective ----------
    def _g_objective_numpy(self, w_vec: np.ndarray, feature_batch_np: np.ndarray) -> float:
        B = feature_batch_np.shape[0]
        w_tiled = np.tile(w_vec[None, :], (B, 1))
        gen_out = self.generator(feature_batch_np, w_tiled).numpy()
        d_inp = np.concatenate([feature_batch_np, gen_out], axis=-1)
        d_scores = self.discriminator(d_inp, training=False).numpy().reshape(-1)
        return float(-np.mean(d_scores))
    
    def _g_objective_tf(self, w_vec: np.ndarray, feature_batch_np: np.ndarray) -> float:

        # Convert input to TF tensor once
        feature_batch = tf.convert_to_tensor(feature_batch_np, dtype=tf.float32)
        w_vec_tf = tf.convert_to_tensor(w_vec[None, :], dtype=tf.float32)
        B = tf.shape(feature_batch)[0]

        # Tile latent vector
        w_tiled = tf.tile(w_vec_tf, [B, 1])

        # Forward pass through generator
        gen_out = self.generator(feature_batch, w_tiled)

        # Concatenate features and generated output
        d_inp = tf.concat([feature_batch, gen_out], axis=-1)

        # Forward pass through discriminator
        d_scores = self.discriminator(d_inp, training=False)
        d_scores = tf.reshape(d_scores, [-1])

        # Compute negative mean (to maximize discriminator score)
        return float(-tf.reduce_mean(d_scores).numpy())  # Only convert the scalar to float


    # ---------- Single training step ----------
    def train_step(self, data):
        feature_data, real_data = data
        B = feature_data.shape[0]
        self.batch_size = B

        feature_data = tf.convert_to_tensor(feature_data, dtype=tf.float32)
        real_data    = tf.convert_to_tensor(real_data, dtype=tf.float32)  # cast eager tensor
        real_concat  = tf.concat([feature_data, real_data], axis=-1)

        # Train discriminator
        for _ in range(self.discriminator_steps):
            with tf.GradientTape() as tape:
                w_vec = np.array([w.numpy() for w in self.generator_weights], dtype=np.float32)
                w_tiled = tf.tile(tf.expand_dims(w_vec, 0), [B, 1])
                fake = self.generator(feature_data, w_tiled)
                fake_concat = tf.concat([feature_data, fake], axis=-1)
                d_fake = self.discriminator(fake_concat, training=True)
                d_real = self.discriminator(real_concat, training=True)
                d_wass = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
                d_gp = self.gradient_penalty(real_concat, fake_concat)
                d_loss = d_wass + self.gp_weight * d_gp
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # COBYLA generator optimization
        feat_np = feature_data.numpy().astype(np.float32)
        w0 = np.array([w.numpy() for w in self.generator_weights], dtype=np.float32)

        def objective(w_vec):
            return self._g_objective_tf(w_vec, feat_np)

        res = minimize(
            fun=objective,
            x0=w0,
            method="COBYLA",
            options={"maxiter": 60, "rhobeg": 0.1, "tol": 1e-3},
        )
        w_opt = res.x.astype(np.float32)
        for i, val in enumerate(w_opt):
            self.generator_weights[i].assign(val)

        g_loss = self._g_objective_tf(w_opt, feat_np)

        # Metrics logging
        w_tiled = tf.tile(tf.expand_dims(tf.convert_to_tensor(w_opt), 0), [B, 1])
        fake = self.generator(feature_data, w_tiled)
        fake_concat = tf.concat([feature_data, fake], axis=-1)
        d_fake = self.discriminator(fake_concat, training=False)
        d_real = self.discriminator(real_concat, training=False)
        d_wass = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
        d_gp = self.gradient_penalty(real_concat, fake_concat)
        d_loss = d_wass + self.gp_weight * d_gp

        self.d_gen_metric.update_state(-g_loss)
        self.d_real_metric.update_state(-(g_loss + d_wass))
        self.d_loss_metric.update_state(d_loss)
        self.d_wass_loss_metric.update_state(d_wass)
        self.d_gp_metric.update_state(d_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

    # ---------- Fit ----------
    def fit(self, feature_data, real_data, epochs, batch_size, callbacks=None, verbose=True):
        num_samples = feature_data.shape[0]
        steps_per_epoch = int(np.ceil(num_samples / batch_size))
        callbacks = callbacks or []

        for epoch in range(epochs):
            epoch_metrics = {m.name: 0.0 for m in self.metrics}
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min((step + 1) * batch_size, num_samples)
                feature_batch = tf.convert_to_tensor(feature_data[start:end])
                real_batch = tf.convert_to_tensor(real_data[start:end])
                batch_logs = self.train_step((feature_batch, real_batch))
                for k, v in batch_logs.items():
                    epoch_metrics[k] += v.numpy() if hasattr(v, "numpy") else float(v)
            for k in epoch_metrics:
                epoch_metrics[k] /= steps_per_epoch
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - {epoch_metrics}")
            for cb in callbacks:
                cb.on_epoch_end(epoch, logs=epoch_metrics)
            for m in self.metrics:
                m.reset_state()

    # ---------- Save generator weights ----------
    def generator_save(self, path):
        np.save(path, np.array([w.numpy() for w in self.generator_weights]))
        print(f"Generator weights saved to {path}")
