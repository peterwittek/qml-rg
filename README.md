Quantum Machine Learning Reading Group @ ICFO
=============================================

The reading group serves a dual purpose. On one hand, we would like to develop an understanding of statistical learning theory and how quantum resources can make a difference. On the other hand, we would like to develop skills that are marketable outside academia. To achieve this dual purpose, we structure the meetings along three topics:

1. A recent, but already important paper on classical machine learning.

2. A quantum machine learning paper.

3. Coding exercises that implement a learning algorithm or a simulation of a quantum protocol.

Papers will be announced a week in advance. Each week there will be a person responsible for a paper, but everybody is expected to read the two papers in advance and prepare with questions. Coding will be done collaboratively through this repository. The reading group requires commitment: apart from the 1.5-2 contact hours a week, at least another 2-3 hours must be dedicated to reading and coding. You are not expected to know machine learning or programming before joining the group, but you are expected to commit the time necessary to catch up and develop the relevant skills.

The language of choice is Python 3. MATLAB users will be shot on sight.
Julia is an upcoming language in which the bleeding-edge of machine learning and quantum simulation are easier to implement, and therefore it is an accepted alternative.

Resources
---------
The broader QML community is still taking shape. We are attempting to organize it through the website[quantummachinelearning.org](http://quantummachinelearning.org/), which is currently under revision. In any case, sign up for the mailing list there. Please also consider contributing to the recently rewritten [Wikipedia article on QML](https://en.wikipedia.org/wiki/Quantum_machine_learning). Apart from new content, stylistic and grammatical edits, figures, and translations are all welcome.

The best way to learn machine learning is by doing it. The book [Python Machine Learning](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning) is a good starter, along with its [GitHub repository](https://github.com/rasbt/python-machine-learning-book). [Kaggle](http://kaggle.com/) is a welcoming community of data scientists. It is not only about competitions: several hundred datasets are hosted on Kaggle, along with notebooks and scripts (collectively known as kernels) that do interesting stuff with the data. These provide perfect stepping stones for beginners. Find a dataset that is close to your personal interests and dive in. For a sufficiently engaging theoretical introduction to machine learning, the book [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://statweb.stanford.edu/~tibs/ElemStatLearn/) is highly recommended.

[Anaconda](https://www.continuum.io/downloads) is the recommended Python distribution if you are new to the language. It ships with most of the scientific and machine learning ecosystem around Python. It includes [Scikit-learn](http://scikit-learn.org/), which is excellent for prototyping machine learning models. For scalable deep learning, [Keras](https://keras.io/) is recommended: it can transparently change between TensorFlow and Theano as back-ends. [QuTiP](http://qutip.org/) is an excellent quantum simulation library, and with the latest version (4.0.2), it is [reasonably straightforward](http://qutip.org/docs/4.0.2/installation.html#platform-independent-installation) to install it in Anaconda with [conda-forge](https://conda-forge.github.io/). QuTiP is somewhat limited in scalability, so perhaps it is worth checking out other simulators, such as [ProjectQ](http://projectq.ch/).

As for [Julia](http://julialang.org/), the [JuliaML](https://github.com/JuliaML) project collects most of the machine learning efforts. It includes cool stuff like [Transformations](https://github.com/JuliaML/Transformations.jl), which allows you to define arbitrary computational graphs, not just the run-of-the-mill static ones implemented in TensorFlow and Theano. [Quantum simulators](https://juliaquantum.github.io/) lag behind a bit, but there are interesting initiatives for doing [tensor network calculations](https://github.com/Jutho/TensorOperations.jl).
