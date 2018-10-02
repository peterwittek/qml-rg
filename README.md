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
The book [Python Machine Learning](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning) is a good starter, along with its [GitHub repository](https://github.com/rasbt/python-machine-learning-book).
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
10.00-11.30, 11. October 2017, Seminar Room (201).

**Topic:**

- ??

**Reading:**

This week we will read two papers about finding phase transitions in many body systems with neural networks.
- ['Learning phase transitions by confusion'](https://arxiv.org/abs/1610.02048), by Evert P.L. van Nieuwenburg,Ye-Hua Liu, Sebastian D. Huber.      
- ['Machine learning phases of matter'](https://arxiv.org/abs/1605.01735), by Juan Carrasquilla, Roger G. Melko.

**Coding exercises:**

- People who are new to the field we recommend to implement a fully connected neural network and a convolutional neural network and train it on a standard data set, like MNIST. Do it either in Keras or Pytorch.


