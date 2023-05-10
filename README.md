# Predictive maintenance

This is the experimental repo for code related to the predictive maintenance
project in collaboration with RUAG.
Its goal is to explore different versions of anomaly detection and how to make
the training as privacy-preserving as possible.

We looked at the following papers for inspiration:

- [Patchcore - Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/pdf/2106.08265)
- [PNI : Industrial Anomaly Detection using Position and Neighborhood Information](https://arxiv.org/pdf/2211.12634v3.pdf)

Currently, the following code is available:

- [patchcore](patchcore) - an implementation of a federated model of patchcore
- [MLP](resnet-mlp) - a first version of running MLP directly after resnet