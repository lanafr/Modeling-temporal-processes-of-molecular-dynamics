# Modeling temporal processes of molecular dynamics

This is the github repo for the code used in the Master thesis "Modeling temporal processes of molecular dynamics". It is the code for the Assisted GDE and Simple GDE networks for molecular dynamics. Specific versions can be accessed by changing augmentation, pretraining and architecture variables.

The networks aim to predict moelcular system configurations further into time. They are able to predict corect atomic distributions for up to 400fs, proved by the RDF graphs. They were trained and tested on the Lennard-Jones fluid.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- cuda
- pytorch
- jax

versions that match. The rest is in the requirements.txt file. They were all installed via pip. openmmtools cannot be installed with pip, so it was compiled from source.

The backbone for this code is GAMD (https://arxiv.org/abs/2112.03383, Li, Zijie and Meidani, Kazem and Yadav, Prakarsh and Barati Farimani, Amir, 2022.), and there is also a script for simulating an MD system with GAMD force-predictions.
