import tensorflow as tf
import numpy as np
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    losses,
    utils,
    metrics,
    optimizers,
    Input,
)
import pennylane as qml
from pennylane_qiskit import qiskit_session


class QGenerator():
    """
    QGenerator is a wrapper class for executing parameterized quantum circuits using PennyLane and Qiskit.
    Args:
        qiskit_circuit (qiskit.QuantumCircuit): The Qiskit quantum circuit to be executed.
        batch_size (int): The number of data samples to process in a batch.
        device (str, optional): The name of the quantum device to use (default is "default.qubit").
        on_hardware (bool, optional): Whether to execute the circuit on actual quantum hardware (default is False).
    Attributes:
        on_hardware (bool): Indicates if execution should be on quantum hardware.
        qiskit_circuit (qiskit.QuantumCircuit): The provided Qiskit circuit.
        num_qubits (int): Number of qubits in the circuit.
        batch_size (int): Batch size for input data.
        dev (pennylane.Device): The PennyLane device instance.
        circuit (callable): The compiled PennyLane QNode for the circuit.
    Methods:
        __call__(data_batch, weights_batch):
            Executes the quantum circuit for each sample in the batch, either on hardware or simulator.
            Args:
                data_batch (tf.Tensor or np.ndarray): Batch of input data for the circuit.
                weights_batch (tf.Tensor or np.ndarray): Batch of parameter weights for the circuit.
            Returns:
                tf.Tensor: The batch of circuit outputs as a TensorFlow tensor.
    """
    def __init__(self, qiskit_circuit, batch_size, device= "default.qubit", on_hardware=False):
        self.on_hardware = on_hardware
        self.qiskit_circuit = qiskit_circuit
        self.num_qubits = qiskit_circuit.num_qubits
        self.batch_size = batch_size
        self.dev = qml.device(device, wires=self.num_qubits)

        @qml.qnode(self.dev, interface="tf", diff_method="parameter-shift") 
        def circuit(data_batch, weights_batch):
            qml.from_qiskit(self.qiskit_circuit)(data_batch, weights_batch)
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]
            #change measurement returns?
        
        self.circuit = circuit


    def __call__(self, data_batch, weights_batch):
        batch_outputs = []
        # Loop through the batch for nowtf
        # TODO figure out a way to pass the whole batch at once
        # TODO figure out a way to pass all input data at once for different tickers,
        #possibly just changing the data layer to be one parameter vector
        batch_size = data_batch.shape[0]
        if self.on_hardware:
            with qiskit_session(self.dev) as session:
                for i in range(batch_size):
                    batch_outputs.append(self.circuit(data_batch[i], weights_batch[i]))
        else:
            for i in range(batch_size):
                    batch_outputs.append(self.circuit(data_batch[i], weights_batch[i]))


        return tf.convert_to_tensor(batch_outputs, dtype=tf.float32)


def discriminator(input_size, use_bias = True, multiplier = 1):
    """
    Builds a discriminator neural network model using Keras functional API.
    Args:
        input_size (int): The size of the input layer.
        use_bias (bool, optional): Whether to use bias in the Dense layers. Defaults to True.
        multiplier (int, optional): Multiplier to scale the initial number of units in the first Dense layer. Defaults to 1.
    Returns:
        keras.Model: A Keras Model representing the discriminator network.
    Notes:
        - The network consists of several Dense layers with LeakyReLU activations.
        - The number of units in each subsequent Dense layer is halved until it reaches 1.
        - The final output layer has a single unit (no activation specified).
    """
    dis_input = layers.Input(shape = (input_size,), name = "discriminator_input")

    unit_size = input_size*multiplier
    x = layers.Dense(units = unit_size, use_bias = use_bias)(dis_input)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    
    #scale down by factor of 2 each time, scales with input size
    unit_size = unit_size//2
    while unit_size>1:
        x = layers.Dense(units = unit_size, use_bias = use_bias)(x)
        x = layers.LeakyReLU(alpha = 0.2)(x)
        unit_size = unit_size//2


    dis_output = layers.Dense(units = 1, use_bias = True)(x)
    discriminator = models.Model(dis_input, dis_output, name="discriminator")
    return discriminator

class GAN(models.Model):
    """
    A Generative Adversarial Network (GAN) model class for quantum circuit-based generation.
    This class implements a custom GAN architecture where the generator is a quantum circuit (QGenerator)
    and the discriminator is a classical neural network. The model supports Wasserstein loss with gradient penalty
    and custom training loops to accommodate quantum circuit constraints.

    Args:
        discriminator_steps (int): Number of discriminator updates per generator update.
        gp_weight (float): Weight for the gradient penalty term in the discriminator loss.
        n_tickers (int): Number of tickers (entities) in the dataset.
        n_features (int): Number of features per ticker.
        n_outputs (int): Number of outputs per ticker.
        generator_qiskit: Quantum circuit object for the generator, must have a `num_parameters` attribute.
        generator_device (str, optional): Device string for quantum simulation/hardware. Defaults to "default.qubit".
        on_hardware (bool, optional): Whether to run the generator on quantum hardware. Defaults to False.
        batch_size (int, optional): Batch size for training. Defaults to 64.

    """
    def __init__(
                self,
                discriminator_steps, 
                gp_weight,
                n_tickers, #number of tickers
                n_features, #number of features per ticker
                n_outputs, #number of outputs per ticker
                generator_qiskit,
                discriminator_layer_multiplier =1,
                generator_device = "default.qubit",
                on_hardware = False,
                batch_size = 64, #what should we be using as batch size?,
        ): 
        super(GAN, self).__init__()
        
        self.discriminator_steps = discriminator_steps
        self.gp_weight = gp_weight

        self.batch_size = batch_size
        self.n_tickers = n_tickers
        self.n_features = n_features
        self.n_outputs = n_outputs

        

        # The input size to the generator
        self.latent_dim = n_features * n_tickers
        
        #calculate number of qubits required (calculation for this may need to change depending on encoding schema?)
        self.num_qubits = self.n_tickers * int(np.ceil(self.n_features/4))

        #initialise generator
        self.generator = QGenerator(generator_qiskit, batch_size, generator_device, on_hardware)

        #initialise discriminator
        #inputs is num generator inputs +num generatout outputs
        self.discriminator = discriminator(self.latent_dim+self.num_qubits, multiplier = discriminator_layer_multiplier)

        #initialise starting weights for generator
        num_generator_params = generator_qiskit.num_parameters - (n_features * n_tickers) #subtracting the number of feature inputs as these are not trainable weights
        self.generator_weights = [
            tf.Variable(tf.random.normal([]), trainable=True)
            for _ in range(num_generator_params)
        ]
    
    
    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_wass_loss_metric = metrics.Mean(name = "d_wass_loss")
        self.d_gp_metric = metrics.Mean(name = "d_gp")
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")
        self.q_loss_metric = metrics.Mean(name="q_loss")
        self.d_real_metric = metrics.Mean(name = "real_socre")
        self.d_gen_metric = metrics.Mean(name = "gen_score")
        self.d_optimizer.build(self.discriminator.trainable_variables)
        self.g_optimizer.build(self.generator_weights)
        
        
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
    
    def gradient_penalty(self, real_data, fake_data):
        alpha = tf.random.normal([self.batch_size, 1], 0.0, 1.0) 
        diff = fake_data - real_data
        interpolated = real_data +alpha*diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training = True)

        #gradient of the preds wrt the inputs
        grads = gp_tape.gradient(pred, [interpolated])[0]

        norm= tf.sqrt(tf.reduce_sum(tf.square(grads), axis = 1))
        gp = tf.reduce_mean((norm -1.0)**2) #returns avg square distance between L2 norm and 1
        return gp
    
        
    def train_step(self, feature_data,real_data):
        # TODO test performance when feeding in feature data + generated/real data to the discriminator
        # feature data should be the same for both real and generated data

        #tiling the weights so each item in the batch has the same weights
        weights_2d = tf.expand_dims(self.generator_weights, 0) 
        weights_tiled = tf.tile(weights_2d, [self.batch_size, 1])

        #move this out of the train step if it works
        real_data = tf.concat([feature_data, real_data], axis = -1)

        #update discriminator a few times
        for i in range(self.discriminator_steps):

            with tf.GradientTape() as tape:
                # Input features to the generator
                generated_data = self.generator(feature_data, weights_tiled)

                #append generated data to the inputs
                generated_data = tf.concat([feature_data, generated_data], axis = -1)

                generated_predictions = self.discriminator(generated_data, training = True)
                real_predictions = self.discriminator(real_data, training = True)

                d_wass_loss = tf.reduce_mean(generated_predictions) - tf.reduce_mean(real_predictions)
                d_gp = self.gradient_penalty(real_data, generated_data)
                d_loss = d_wass_loss + d_gp*self.gp_weight
            
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
        
        with tf.GradientTape(persistent= True) as tape:
            # tile params whilst the trainable weights are being watched
            tape.watch(self.generator_weights)
            weights_2d = tf.expand_dims(self.generator_weights, 0) 
            weights_tiled = tf.tile(weights_2d, [self.batch_size, 1])

            generated_data = self.generator(feature_data, weights_tiled)
            #concat feature vector
            generated_data = tf.concat([feature_data, generated_data], axis = -1)
            generated_predictions = self.discriminator(generated_data, training = True)
            g_loss = -tf.reduce_mean(generated_predictions)
          
        #update generator weights based on g loss
        gen_gradient = tape.gradient(g_loss, self.generator_weights)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator_weights))


        # discriminator accuracy metric
        real_predictions = self.discriminator(real_data, training = True)
        #set predictions to -1 is <0 or 1 if >=0
        real_predictions = tf.where(tf.greater_equal(real_predictions, 0), tf.ones_like(real_predictions), tf.ones_like(real_predictions) * -1)
        generated_predictions = tf.where(tf.greater_equal(generated_predictions, 0), tf.ones_like(generated_predictions), tf.ones_like(generated_predictions) * -1)

        self.d_gen_metric.update_state(-g_loss)
        self.d_real_metric.update_state(-(g_loss +d_wass_loss))
        self.d_loss_metric.update_state(d_loss)
        self.d_wass_loss_metric.update_state(d_wass_loss)
        self.d_gp_metric.update_state(d_gp)
        self.g_loss_metric.update_state(g_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    #Have to define a custom fit step because the quantum circuit doesn't like the symbolic step
    def fit(self, feature_data, real_data, epochs, batch_size, callbacks = None, verbose=True):

        num_samples = feature_data.shape[0]
        steps_per_epoch = int(np.ceil(num_samples / self.batch_size))

        if callbacks is None:
            callbacks = []

        for epoch in range(epochs):

            # Reset batch size after last batch smaller
            self.batch_size= batch_size


            epoch_metrics = {m.name: 0.0 for m in self.metrics}

            for step in range(steps_per_epoch):
                start = step * self.batch_size
                end = min((step + 1) * self.batch_size, num_samples)

                feature_batch = feature_data[start:end]
                real_batch = real_data[start:end]

                # Update batch size in GAN if last batch is smaller
                self.batch_size = feature_batch.shape[0]

                # Perform a single train step for this batch
                batch_metrics = self.train_step(feature_batch, real_batch)

                # Accumulate metrics for the epoch
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value

            # Average metrics over steps
            epoch_metrics = {k: v / steps_per_epoch for k, v in epoch_metrics.items()}

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - {epoch_metrics}")
            

            
             # Call on_epoch_end for callbacks
            for cb in callbacks:
                cb.on_epoch_end(epoch, logs=epoch_metrics)

            # Reset Keras metrics
            for m in self.metrics:
                m.reset_state()


    def generator_save(self, path):
        """
        Saves the generator's weights to a specified file path in NumPy .npy format.

        Args:
            path (str): The file path where the generator weights will be saved.

        """
        weights_list = [w.numpy() for w in self.generator_weights]
        np.save(path, weights_list)
        print(f"Generator weights saved to {path}")


