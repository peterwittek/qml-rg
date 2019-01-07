Quantum Machine Learning Reading Group @ ICFO Fall Session 2018
=============================================

This is the second edition of the QML Reading group at ICFO. The archive of the first edition is [here](https://github.com/peterwittek/qml-rg/tree/master/Archiv_Session_Spring_2017) and [here](https://github.com/peterwittek/qml-rg/tree/master/Archiv_Session_Spring_2018). We will face some restructuring and aim for more self initiative than in the last Session. We will go back to the roots of this RG and just discuss publications of interest that will be chosen by someone.

So for the first sessions we have the following structure in mind.

1. We will again define papers to read for each session. We are happy if people come forward with interesting topics or papers. This session will more be about keeping track of latest advances in ML, QML and ML assisted physics.

2. Most people attending will already be quite advanced in ML. Therefore we will not start with ML basics in this group. But we are aware that there are ML beginners who attend the RG and we are also happy to explain basics if needed.

3. We are not sure yet if there will be coding exercises. In the last session most people were too busy anyway. But we are always happy to discuss coding problems or suggestions.


Topics will be announced a week in advance.
Coding will be done collaboratively through this repository.

The reading group requires commitment: apart from the 1.5-2 contact hours a week, at least another 2-3 hours must be dedicated to reading and coding.
You are not expected to know machine learning or programming before joining the group, but you are expected to commit the time necessary to catch up and develop the relevant skills.

The language of choice is Python 3.

Resources
---------
The broader QML community is still taking shape.
We are attempting to organize it through the website [quantummachinelearning.org](http://quantummachinelearning.org/). You can also sign up for the mailing list there.
Please also consider contributing to the recently rewritten [Wikipedia article on QML](https://en.wikipedia.org/wiki/Quantum_machine_learning).
Apart from new content, stylistic and grammatical edits, figures, and translations are all welcome.

The best way to learn machine learning is by doing it.
The book [Python Machine Learning](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning) is a good starter, along with its [GitHub repository](https://github.com/rasbt/python-machine-learning-book). For the deep learning part, you can have look at the [github of the course](https://github.com/PatrickHuembeli/QML-Course-UPC-2018) we gave at the upc where we discuss convolutional neural networks, Boltzman machines and reinforcement learning.
[Kaggle](http://kaggle.com/) is a welcoming community of data scientists.
It is not only about competitions: several hundred datasets are hosted on Kaggle, along with notebooks and scripts (collectively known as kernels) that do interesting stuff with the data.
These provide perfect stepping stones for beginners.
Find a dataset that is close to your personal interests and dive in.
For a sufficiently engaging theoretical introduction to machine learning, the book [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://statweb.stanford.edu/~tibs/ElemStatLearn/) is highly recommended.

[Anaconda](https://www.continuum.io/downloads) is the recommended Python distribution if you are new to the language.
It ships with most of the scientific and machine learning ecosystem around Python.
It includes [Scikit-learn](http://scikit-learn.org/), which is excellent for prototyping machine learning models.
For scalable deep learning, [Keras](https://keras.io/) is easy to start with, bit it uses the more intransparent and complicated tensorflow backend. For simple implementations we recommend it, but as soon as someone wants to implement more advanced neural networks we recommend changing to pytorch.
[QuTiP](http://qutip.org/) is an excellent quantum simulation library, and with the latest version (4.1), it is [reasonably straightforward](http://qutip.org/docs/4.1/installation.html#platform-independent-installation) to install it in Anaconda with [conda-forge](https://conda-forge.github.io/).
QuTiP is somewhat limited in scalability, so perhaps it is worth checking out other simulators, such as [ProjectQ](http://projectq.ch/).

We will follow the good practices of [software carpentry](http://software-carpentry.org/), that is, elementary IT skills that every scientist should have.
In particular, we use [git](https://rogerdudler.github.io/git-guide/) as a version control system and host the repository right here on GitHub.
When editing text or [Markdown](https://guides.github.com/features/mastering-markdown/) documents like this one, please write every sentence in a new line to ensure that conflicts are efficiently resolved when several people edit the same file.

Meeting 1
---------
10.30-12.00, 18.October 2018, Aquarium Room (AMR) (280).

**Topic:**

- We will have a look at this [paper](https://www.semanticscholar.org/paper/Knowledge-graph-refinement%3A-A-survey-of-approaches-Paulheim/93b6329091e215b9ef007a85c07635f09e7b8adb). Knowledge graph refinement: A survey of approaches and evaluation methods.


Meeting 2
---------
10.30-12.00, 8.November 2018, Aquarium Room (AMR) (280).

**Topic:**

- We will go through the paper [Bayesian Deep Learning on a Quantum Computer](https://arxiv.org/abs/1806.11463). This will include a short introduction to supervised learning in feedforward neural networks for the newcomers, and notes on Bayesian learning.


Meeting 3
---------
10.30-12.00, 15.November 2018, Aquarium Room (AMR) (280).

**Topic:**

- Our journey through [Bayesian Deep Learning on a Quantum Computer](https://arxiv.org/abs/1806.11463) continues. We will also review equivalences between GP training and training of deep neural networks and quantum-assisted training of GPs.

Meeting 4
---------
09.30-11.00, 13.December 2018, Aquarium Room (AMR) (280).

**Topic:**

- In this session we will review the recent paper [[1]](https://arxiv.org/abs/1807.04271), where the author proposes a classical algorithm that mimics the quantum-algorithm for recommendation systems [[2]](http://drops.dagstuhl.de/opus/volltexte/2017/8154/pdf/LIPIcs-ITCS-2017-49.pdf) by using stochastic sampling [[3]](https://www.math.cmu.edu/~af1p/Texfiles/SVD.pdf). For preparation, it is recommended going over [1] and reading the nice introduction of [3]. Knowing [2]? Even better.

Meeting 5
---------
10.30-12.00, 10.January 2018, Aquarium Room (AMR) (280).

**Topic:**

- To kick-start the 2019 reading group, I will follow the recent trend of showing the loopholes of by presenting [this paper](https://arxiv.org/abs/1803.11173). 

The basic message is to argue that that there are major problems when trying to perform gradient descent on classically parametrised quantum circuits (i.e. 'quantum neural networks'), since the gradient will be essentially zero everywhere.




