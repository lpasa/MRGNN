# Multiresolution Reservoir Graph Neural Network

Graph neural networks are receiving increasing
attention as state-of-the-art methods to process graph-structured
data. However, similar to other neural networks, they tend
to suffer from a high computational cost to perform training.
Reservoir computing (RC) is an effective way to define neural
networks that are very efficient to train, often obtaining compa-
rable predictive performance with respect to the fully trained
counterparts. Different proposals of reservoir graph neural
networks have been proposed in the literature. However, their
predictive performances are still slightly below the ones of fully
trained graph neural networks on many benchmark datasets,
arguably because of the oversmoothing problem that arises when
iterating over the graph structure in the reservoir computation.
In this work, we aim to reduce this gap defining a multiresolution
reservoir graph neural network (MRGNN) inspired by graph
spectral filtering. Instead of iterating on the nonlinearity in
the reservoir and using a shallow readout function, we aim to
generate an explicit k-hop unsupervised graph representation
amenable for further, possibly nonlinear, processing. Experiments
on several datasets from various application areas show that our
approach is extremely fast and it achieves in most of the cases
comparable or even higher results with respect to state-of-the-art
approaches.

Paper: https://ieeexplore.ieee.org/abstract/document/9476188

If you find this code useful, please cite the following:
> @article{pasa2021multiresolution,  
  title={Multiresolution Reservoir Graph Neural Network},  
  author={Pasa, Luca and Navarin, Nicol{\`o} and Sperduti, Alessandro},   
  journal={IEEE Transactions on Neural Networks and Learning Systems},  
  year={2021},  
  publisher={IEEE}  
}
