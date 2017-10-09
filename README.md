Quantum Machine Learning Reading Group @ ICFO Fall Session 2017
=============================================

This is the second edition of the QML Reading group at ICFO. The archive of the first edition is [here](https://github.com/peterwittek/qml-rg/tree/master/Archiv_Session_Spring_2017). We will face some restructuring and aim for much
more self organization than in the last Session. Accoring to the wishes of a lot of participants in the spring
session, we will focus a bit more on classical ML and try to repeat some of the topics that have been discussed
in spring and really try to understand these techniques. Some people also complained, that they got lost in the
coding exercises very early in the spring session. In this matter we want to slow down a bit and
restart with the basics.

So for the first sessions we have the following structure in mind.

1. We don't choose papers anymore, we will focus more on topics like (tSNE, Neural Networks, etc.)
   So there will be one topic each session and someone presents it. Some literature will be provided.

2. For the coding we think about simple tasks for the beginning. We for example have
https://grads.yazabi.com/ in mind. This is an introduction into ML and could also be done individually.


Topics will be announced a week in advance.
Coding will be done collaboratively through this repository.
Parallely for advanced programmer we can start another Kaggle competition.

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
For scalable deep learning, [Keras](https://keras.io/) is recommended: it can transparently change between TensorFlow and Theano as back-ends.
[QuTiP](http://qutip.org/) is an excellent quantum simulation library, and with the latest version (4.1), it is [reasonably straightforward](http://qutip.org/docs/4.1/installation.html#platform-independent-installation) to install it in Anaconda with [conda-forge](https://conda-forge.github.io/).
QuTiP is somewhat limited in scalability, so perhaps it is worth checking out other simulators, such as [ProjectQ](http://projectq.ch/).

We will follow the good practices of [software carpentry](http://software-carpentry.org/), that is, elementary IT skills that every scientist should have.
In particular, we use [git](https://rogerdudler.github.io/git-guide/) as a version control system and host the repository right here on GitHub.
When editing text or [Markdown](https://guides.github.com/features/mastering-markdown/) documents like this one, please write every sentence in a new line to ensure that conflicts are efficiently resolved when several people edit the same file.

Meeting 1
---------
10.00-11.30, October 2017, Seminar Room (201).

Papers:

- Deep Learning: Neural Networks, Convolutional Neural Networks, Autoencoder, VAE, DANN

Coding exercise:

- The first week does not have a coding exercise.
Instead, please ensure that your computational environment is up and running.
Python with the recommended libraries should be there, along with an editor.
Please make keras running in Python. On windows it is not so easy. Plan a day for this.
Either use your favourite text editor, or opt for Spyder, which is also bundled in Anaconda.
Ensure that you can open and run Jupyter notebooks.
Go through a git tutorial, like the one linked under Resources, fork this repository, clone it, and add the upstream repo to follow.
If you need help, we will be around.

Tutorial 1
----------
16.00-17.30, ???, Yellow Lecture Room (247).

This tutorial is about the importance of using a version control system. The focus is on git, starting with using it locally, for collaborating on LaTeX documents on Overleaf, and finishing with some examples using GitHub. The [notes](Tutorials/Git.md) are in the Tutorials folder.
