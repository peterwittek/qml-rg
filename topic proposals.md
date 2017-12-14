Topic proposals @ ICFO Fall Session 2017
=============================================

This file contains the topics the attendants of the Reading Group are interested with. Future weeks papers and exercises will be chosen in order to satisfy this list. Write down any proposal you have.

- **Topics**
  - Boltzmann Machines. (January) Troyers paper: https://arxiv.org/pdf/1606.02318.pdf, classical paper
  - BM and deep believ, BM and D-wave
  - Batchnormalization. https://arxiv.org/abs/1502.03167
  - Graphical probabilistic models.
  - resNet
  - Random Forest, XGBoost...
  - deep and cheap learning, universal approx theorem
  

- **Classical papers** (Dec 14)
  - [Dynamic Routing between capsules](https://arxiv.org/abs/1710.09829).  I have read in a few places already that this might be a very promising future direction for feedforward neural networks. The main idea is that different types of information (shapes, orientations...) are stored in different "capsules" in every layer, each independent of one another, the information in different capsules is sent/re-routed into different capsules in the following layer depending on the information itself. A less-technical description may be found [here](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc). It might be very interesting as well to dig in the [keras implementation](https://github.com/XifengGuo/CapsNet-Keras).
  
  - [Learning to act by predicting the future](https://arxiv.org/abs/1611.01779). A new proposal of mapping Reinforcement Learning to a supervised task, that allows for the definition of complex goals that change over time with the circumstances. In particular, this method crushed A3C in a Doom competition in 2016. For an introductive view, see [this blogpost](https://www.oreilly.com/ideas/reinforcement-learning-for-complex-goals-using-tensorflow). Also, a keras implementation can be found [here](https://github.com/flyyufelix/Direct-Future-Prediction-Keras).

- **Quantum papers**
  - [An efficient quantum algorithm for generative machine learning](https://arxiv.org/abs/1711.02038).  The title is quite self-explanatory. Indeed, the authors describe a quantum algorithm that allows for creating generative models in an efficient way. It might be interesting to discuss now that we know what a generative model is.
  - [A Quantum Extension of Variational Bayes Inference](https://arxiv.org/abs/1712.04709). The authors show a quantum method for variational Bayes inference, which is the method underlying variational autoencoders, claiming to outperform current methods.
