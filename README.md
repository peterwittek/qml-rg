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
10.00-11.30, 19. October 2017, Seminar Room (201).

**Topic:**

- Repetition Deep Learning: Neural Networks, Convolutional Neural Networks

**Reading:**

This week we will read two papers about finding phase transitions in many body systems with neural networks.
- ['Learning phase transitions by confusion'](https://arxiv.org/abs/1610.02048), by Evert P.L. van Nieuwenburg,Ye-Hua Liu, Sebastian D. Huber.      
- ['Machine learning phases of matter'](https://arxiv.org/abs/1605.01735), by Juan Carrasquilla, Roger G. Melko.

**Coding exercises:**

- The first week does not have a real coding exercise.
Instead, please ensure that your computational environment is up and running.
Python with the recommended libraries should be there, along with an editor.
Please make keras running in Python. On windows it is not so easy. Plan a day for this.
In the exercise folder is a python file 'week_01.py', which contains a simple CNN for the MNIST dataset.
Try to make this run and play a bit with it.
Either use your favourite text editor, or opt for Spyder, which is also bundled in Anaconda.
Ensure that you can open and run Jupyter notebooks.

- Go through the confusion code 'week1_confusion.py' and try to check if confusion is feasible for deep neural networks. For this add more layers to the NN and also try using more neurons each layer.

- Go through a git tutorial, like the one linked under Resources, fork this repository, clone it, 
and add the upstream repo to follow.

If you need help, we will be around.


Meeting 2
---------
10.00-11.30, 26. October 2017, Seminar Room (201).

**Topic:**

The topic of the session will be  **autoencoders**, their applications and limitations. We will follow part of the keras [tutorial](https://blog.keras.io/building-autoencoders-in-keras.html).

**Coding exercise:**
 
This weeks homework consists on three tasks:

1. Construct a simple autoencoder based on the MNIST database or, if you are brave, the CIFAR10 database.
2. Use an autoencoder to denoise images.
3. Build an autoencoder with a compressed representation of dimension 2 (this means 2 neurons in the central layer). Use the enconding part to extract features, plot them and compare the results to the Linear Discriminant Analysis (for an example, see this [link](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html)). This task is a simplification of the procedure explained in this week's classical paper.

**Reading:**

- G. E. Hinton and R. R. Salakhutdinov, [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/science.pdf), 2006.

- Jonathan Romero, Jonathan P. Olson, Alan Aspuru-Guzik, [Quantum autoencoders for efficient compression of quantum data](https://arxiv.org/abs/1612.02806), 2017.
 
Meeting 3
---------
10.00-11.30, 02. November 2017, Seminar Room (201).

**Topic:**

The topic of the session will be  **variational autoencoders (vAE)**

**Coding exercise:**

We propose something different for this week. We will have a main exercise, plus some extra ones that arose from the discussions. Feel free to choose any that you like. However, the main exercise is the recommended one for those who want to know the basics of ML.

- **Main exercise**: Build an autoencoder that colors images. Use the images in the CIFAR10 dataset and transform them to grayscale using the following function:

```python
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
```

- **Extra exercises**:
    - Take a classification network for MNIST (such as [this one](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)) and, after training, remove the classification layer and append a reversed copy of it to test its performance as an autoencoder.
    - Create a denoising autoencoder, where the noise model in during training is variable and chosen at random.
    - Explore pre-training autoencoders with RBMs, in the spirit of [Hinton's paper](https://www.cs.toronto.edu/~hinton/science.pdf).
    
**Reading:**

Since some people would like to have more introductory reading, we suggest:

- Michael Nielsens, http://neuralnetworksanddeeplearning.com/
- A very nice intuitive introduction into vAE http://kvfrans.com/variational-autoencoders-explained/
- A bit an advanced intro into vAE https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

Meeting 4
---------
10.00-11.30, 09. November 2017, Seminar Room (201).

**Topic:**

The topic of this week's session will be **Reinforcement Learning (RL)**. We will introduce the topic, talk about Q-learning algorithms and how to extend them to create deep Q-networks.

**Coding exercise:**

- **Main exercise**: Due to the complexity of the main homework of the previous week, the colourizing autoencoder, we will continue to work on it until next thursday.

- **Extra exercise**: You can start to explore the world of RL by creating a agent that can play to the Frozen Lake game. [Here](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) is a nice tutorial.

**Reading:**

- For an introduction on RL with coding examples, take a look on to the different parts of this [tutorial](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0), specially Parts 0, 1, 1.5 and 2.


Meeting 5
---------
10.00-11.30, 16. November 2017, Seminar Room (201).


**Topic:**

The topic for this week is **Convolutional autoencoders**. The discussion is centered around the Alexandre's code. You can find it in the codes folder.

**Reading:**

This nice [tutorial](http://tinyclouds.org/colorize/) explains how to apply a convolutional autoencoder to colorize images. It also introduces the network ResNet, which will be discussed in future sessions.


Meeting 6
---------
10.00-11.30, 23. November 2017, Aquarium Room (AMR) (280).

The topic of this week's session will be, again, **Reinforcement Learning (RL)**. We will continue from the basics we discussed on week 4 and introduce the concepts of Policy Network and Deep Q Networks.


**Reading:**

- We will take a look into the state of the art RL algorithms. One of the hottests one right now is the new [AlphaGo Zero](https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html), which beats with unsupervised learning the older, but still interesting, [AlphaGo](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html) algorithm.


**Coding exercise:**

- Create an actor-critic that can solve the [Cart-Pole game](https://gym.openai.com/envs/CartPole-v0/). A very enlightening implementation of the Actor-Critic model on the Frozen-Lake game can be found [here](http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html). An simple program to solve the Cart-Pole game can be found [here](https://github.com/GaetanJUVIN/Deep_QLearning_CartPole/blob/master/cartpole.py)


Meeting 7
---------
10.00-11.30, 30. November 2017, Aquarium Room (AMR) (280).

The topic of this week's session will be, again, **Reinforcement Learning (RL)**. We will finally try to find some time on the AlphaGo paper.


**Reading:**

- Same as last week

Meeting 8
---------
10.00-11.30, 14. December 2017, Aquarium Room (AMR) (280).

The topic of this week's session will be,  **Capsule Neural Networks** that might be a very promising future direction for feedforward neural networks. The main idea is that different types of information (shapes, orientations...) are stored in different "capsules" in every layer, each independent of one another, the information in different capsules is sent/re-routed into different capsules in the following layer depending on the information itself. A less-technical description may be found [here](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc). It might be very interesting as well to dig in the [keras implementation](https://github.com/XifengGuo/CapsNet-Keras)

**Reading:**
[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829), of Geoffrey E. Hinton, Google Brain, Toronto




Meeting 9
---------
10.00-11.30, 18. January 2018, Nest Yellow Meeting Room (NYMR) (226).


This week we will read [A Quantum Extension of Variational Bayes Inference](https://arxiv.org/abs/1712.04709). The authors show a quantum method for variational Bayes inference, which is the method underlying variational autoencoders, claiming to outperform current methods.


Meeting 10
---------
10.00-11.30, 25. January 2018, Nest Yellow Meeting Room (NYMR) (226).

The topic of this week's session will be,  **Boltzmann Machines**. We will first have a general introduction and then we will discuss a paper about finding ground states of many body Hamiltonians with Boltzmann machines. 

**Reading:**
[Solving the Quantum Many-Body Problem with Artificial Neural Networks](https://arxiv.org/abs/1606.02318), of Giuseppe Carleo and Matthias Troyer.


Meeting 11
---------
10.00-11.30, 1. February 2018, Nest Yellow Meeting Room (NYMR) (226).

Finally we will be able to discuss [A Quantum Extension of Variational Bayes Inference](https://arxiv.org/abs/1712.04709). Since most people don't seem to be very excited about this paper we will keep it short.

After that we will again focus on  **Boltzmann Machines**. There will not be anything to read. The idea for this session is more to pick up the discussion from last week and really try to understand Restricted Boltzmann Machines. How are they trained, what are they capable of, what not, etc., etc...

**Reading:**
Nothing specific. Do internet research about RBM and prepare questions we can discuss.

next Meetings
---------

- 8.2. Deep Believ (Gorka)
- 15.2. XGBoost, Decision Tree, Random Forest (Alex?)



Quantum Machine Learning
------------------------

Since last weeks we focused mostly on classical ML we would like to include some QML papers in our reading group. To at least have an overview, what is happening in the QML community, every week someone of the group should quickly go through the arxiv of the past week and give a quick update of the papers. Max 15 minutes. If there is time left it would also make sense to go to older QML publications.

We already agreed on an order, with whom will start:

Alexandre (start 18.1), Patrick,
Alejandro, Alex, Jessica, Mauri.

Tutorial 1
----------
16.00-17.30, ???, Yellow Lecture Room (247).

This tutorial is about the importance of using a version control system. The focus is on git, starting with using it locally, for collaborating on LaTeX documents on Overleaf, and finishing with some examples using GitHub. The [notes](Tutorials/Git.md) are in the Tutorials folder.
