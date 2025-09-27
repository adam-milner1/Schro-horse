# Schro-horse

This is a qiskit implementation of a QGAN market simulator in which quantum generators simulate market data for a number of tech stocks to be evaluated by a classical discriminator.  The tech stocks have been selected based on expected correlations - due to:
- historical correlations
- complementary and competing busiess offerings
- correlated customer-bases
- similar market sensitivies

The QGAN will be seeded from historical (Yahoo Finance) data and, for a given stock, will make use of the stock's open/close price, volume and ***

This historical data will be used to evaluate the QGAN's performance.

## Links

Research papers

[Prediction of Stocks Index Price using Quantum GANs](https://arxiv.org/html/2509.12286v1)

[Quantum generative modeling for financial time series with temporal correlations](https://arxiv.org/pdf/2507.22035)

[Towards Realistic Market Simulations: a Generative Adversarial Networks Approach](https://arxiv.org/pdf/2110.13287)

## Change log ##


from Jayden:
I think the model may be working now but I haven't got a way to test it yet so it could be not updating the weights or something idk- will investigate tomorrow.

Currently it only works with 1 ticker- also aiming to fix tomorrow.

Tomorrow morning i'll make some bits to load the data and params back into the qiskit circuit so everyone can use it.

theres probably loads of issues still and its been very painful process to get something that appears to train, but making progress!


## Dependencies

This projects depnds on the follow Python packages

`pip install qiskit==2.1.0`

`pip install git+https://github.com/PennyLaneAI/pennylane-qiskit.git#egg=pennylane-qiskit`

`pip install git+https://github.com/PennyLaneAI/pennylane.git#egg=pennylane`

**Penny Lane** is a QML library that makes it compatible with tensorflow training
