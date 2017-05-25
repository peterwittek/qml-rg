Quantum Machine Learning Reading Group @ ICFO
=============================================

The reading group serves a dual purpose.
On one hand, we would like to develop an understanding of statistical learning theory and how quantum resources can make a difference.
On the other hand, we would like to develop skills that are marketable outside academia.
To achieve this dual purpose, we structured the first twelve meetings along three topics:

1. A recent, but already important paper on classical machine learning.

2. A quantum machine learning paper.

3. Coding exercises that implement a learning algorithm or a simulation of a quantum protocol.

Papers will be announced a week in advance.
Each week there will be a person responsible for a paper, but everybody is expected to read the two papers in advance and prepare with questions.
Coding will be done collaboratively through this repository.

After the first twelve meetings, papers are chosen by the person leading the session. A paper from the preceding six months is chosen. Sources of inspiration are [SciRate](https://scirate.com/) and [arXiv Sanity Preserver](http://www.arxiv-sanity.com/). Coding exercises are replaced by a Kaggle competition team.

The reading group requires commitment: apart from the 1.5-2 contact hours a week, at least another 2-3 hours must be dedicated to reading and coding.
You are not expected to know machine learning or programming before joining the group, but you are expected to commit the time necessary to catch up and develop the relevant skills.

The language of choice is Python 3.
MATLAB users will be shot on sight.
Julia is an upcoming language in which the bleeding-edge of machine learning and quantum simulation are easier to implement, and therefore it is an accepted alternative.

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

As for [Julia](http://julialang.org/), the [JuliaML](https://github.com/JuliaML) project collects most of the machine learning efforts.
It includes cool stuff like [Transformations](https://github.com/JuliaML/Transformations.jl), which allows you to define arbitrary computational graphs, not just the run-of-the-mill static ones implemented in TensorFlow and Theano.
[Quantum simulators](https://juliaquantum.github.io/) lag behind a bit, but there are interesting initiatives for doing [tensor network calculations](https://github.com/Jutho/TensorOperations.jl).

We will follow the good practices of [software carpentry](http://software-carpentry.org/), that is, elementary IT skills that every scientist should have.
In particular, we use [git](https://rogerdudler.github.io/git-guide/) as a version control system and host the repository right here on GitHub.
When editing text or [Markdown](https://guides.github.com/features/mastering-markdown/) documents like this one, please write every sentence in a new line to ensure that conflicts are efficiently resolved when several people edit the same file.

Meeting 1
---------
10.00-11.30, 16 February 2017, Seminar Room (201).

Papers:

- Kingma, D. & Ba, J. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). *Proceedings of ICLR-15, 3rd International Conference for Learning Representations*, 2014.
This paper introduces a derivative of stochastic gradient descent that is now widely used in training deep learning networks.
For instance, it is one of the [available optimizers](https://keras.io/optimizers/) in Keras.

- Wan, K. H.; Dahlsten, O.; Kristjánsson, H.; Gardner, R. & Kim, M. S. [Quantum generalisation of feedforward neural networks](https://arxiv.org/abs/1612.01045). *arXiv:1612.01045*, 2016.
Papers on quantum neural networks are typically written by crackpots, and this work is one of the refreshing exceptions.
It gives a twist on gradient descent to train neural networks of the quantum generalization of classical tasks. One of the examples is an autoencoder: a [related and equally interesting paper](https://arxiv.org/abs/1612.02806) came out a few days later, which is also worth reading.
This latter paper explicitly mentions that all simulations were implemented in Python with QuTiP.

Coding exercise:

- The first week does not have a coding exercise.
Instead, please ensure that your computational environment is up and running.
Python with the recommended libraries should be there, along with an editor.
Either use your favourite text editor, or opt for Spyder, which is also bundled in Anaconda.
Ensure that you can open and run Jupyter notebooks.
Go through a git tutorial, like the one linked under Resources, fork this repository, clone it, and add the upstream repo to follow.
If you need help, we will be around in 372 from 5pm on Wednesday, 15 Feb.

Tutorial 1
----------
16.00-17.30, 21 February 2017, Yellow Lecture Room (247).

This tutorial is about the importance of using a version control system. The focus is on git, starting with using it locally, for collaborating on LaTeX documents on Overleaf, and finishing with some examples using GitHub. The [notes](Tutorials/Git.md) are in the Tutorials folder.


Meeting 2
---------
10.00-11.30, 23 February 2017, Seminar Room (201).

Papers:

- Abadi, M.; Barham, P.; Chen, J.; Chen, Z.; Davis, A.; Dean, J.; Devin, M.; Ghemawat, S.; Irving, G.; Isard, M.; Kudlur, M.; Levenberg, J.; Monga, R.; Moore, S.; Murray, D. G.; Steiner, B.; Tucker, P.; Vasudevan, V.; Warden, P.; Wicke, M.; Yu, Y. & Zheng, X.
[TensorFlow: A System for Large-scale Machine Learning](https://arxiv.org/abs/1603.04467).
*Proceedings of the 12th USENIX Conference on Operating Systems Design and Implementation*, 2016, 265-283.
This is the only systems paper we will discuss.
Given the hype around it, its actual importance, and its relevance for getting a job in real life, it is worth looking at it.
An earlier open source effort, [Theano](http://deeplearning.net/software/theano/) implements the same idea of using a data flow graph as a computational abstraction; see the [matching paper](https://arxiv.org/abs/1605.02688). Notes:

  - Machine learning is mainly a bunch of linear algebra. This is what enables atomizing operations as TensorFlow does, and it also explains a large number of QML algorithms. You can also consider TensorFlow as an alternative to BLAS and LAPACK, and implement normal scientific workflows.

  - The complexity of TensorFlow is astonishing: it can be deployed on anything from FPGAs through ASICs to GPUs and CPUs. If this much software engineering went into quantum simulation libraries...

  - Finally a paper that says at least implicitly that MapReduce was the wrong paradigm, and Spark does not help much (see section on Batch dataflow systems).

  - Despite the claim that inference is expensive, it is actually quite cheap: 5 billion FLOPS = 5 GFLOPS. The Titan X consumer-grade GPU can do 6 TFLOPS, that is, 1200 times more. This is in sharp contrast to probabilistic graphical models, where inference is #P-complete.

  - The dataflow graph is deterministic. In Julia, [Transformations.jl](https://github.com/JuliaML/Transformations.jl) allow more freedom.

  - Check the datatypes: 32-bit representation wins out. Internally, Microsoft uses 3 bits for a single weight in a neural network. This is a far cry of the default 64-bit precision of contemporary CPUs, and it also explains why consumer-grade GPUs work better than GPUs designed for scientific workflows.

  - The data structures must be dense. This is primarily because of GPUs. To address sparse models, they have a method for sparse embedding (Section 4.2).


- Lau, H.-K.; Pooser, R.; Siopsis, G. & Weedbrook, C.
[Quantum machine learning over infinite dimensions](https://arxiv.org/abs/1603.06222). *arXiv:1603.06222*, 2016.
This paper is the only proposal so far for using continuous variable systems for doing machine learning. These systems are attractive for both their theoretical and experimental properties, and the paper also allows us to talk about building blocks of machine learning algorithms. Notes:

  - $a = {a_x: x=1,\ldots, N}$ is assumed to be normalized. This can be an expensive operation.

  - HHL in a CV setting. Many, if not most QML proposals use HHL. Among other things, HHL assumes that the matrix to be inverted is well-conditioned, which roughly means it is far from being singular.

  - See also [Issue #10](https://github.com/peterwittek/qml-rg/issues/10).


Coding exercises:

- Implement an autoenconder in TensorFlow, say, random ten dimensional vectors encoded on a six dimensional hidden layer.
Use Adam as the optimization routine.
Keras is a recommended abstraction layer around TensorFlow.

- Simulate a swap test.
The [circuit](https://en.wikipedia.org/wiki/Quantum_digital_signature#Public_Key_should_be_the_same_for_every_recipient_.28Swap_Test.29) is nothing but a Fredkin gate and a Hadamard gate, followed by a local measurement on the ancilla.
This task is trivial to do in QuTiP, but you can also try your hands on ProjectQ, or the [IBM Quantum Experience](https://www.research.ibm.com/quantum/).

Tutorial 2
----------
16.00-17.30, 07 March 2017, Yellow Lecture Room (247).

This is a Python introduction without talking about introductory stuff on Python. The goal was to give a general idea of what goes into designing a programming language and what kind of trade-offs have to be made, followed by some idiomatic expressions in Python and some caveats. The [corresponding notebook](Tutorials/Python_Introduction.ipynb) is in the Tutorials folder.

Meeting 3
---------
10.00-11.30, 09 March 2017, Seminar Room (201).

Papers:

- Silver, D.; Huang, A.; Maddison, C. J.; Guez, A.; Sifre, L.; van den Driessche, G.; Schrittwieser, J.; Antonoglou, I.; Panneershelvam, V.; Lanctot, M.; Dieleman, S.; Grewe, D.; Nham, J.; Kalchbrenner, N.; Sutskever, I.; Lillicrap, T.; Leach, M.; Kavukcuoglu, K.; Graepel, T. & Hassabis, D. [Mastering the game of *Go* with deep neural networks and tree search](http://doi.org/10.1038/nature16961). *Nature*, 2016, 529, 484-489. It is the state-of-the-art in reinforcement learning. The scheme combines deep learning with a heuristic search, which is a pattern that is seen over and over again since this paper came out. The simpler, but equally glamorous task of playing Atari games was published by the same group; [that paper](https://arxiv.org/abs/1312.5602) is also worth a look. Notes:

  - The self-playing aspect is essential to reduce overfitting in the RL of value networks.

  - Initial supervised training with deep learning massively reduces the search space.


- Dunjko, V.; Taylor, J. M. & Briegel, H. J. [Quantum-Enhanced Machine Learning](https://arxiv.org/abs/1610.08251). *Physical Review Letters*, 2016, 117, 130501. This paper takes a comprehensive look at what quantum agents can learn in a reinforcement learning scenario. It is worth looking at an [earlier and much longer version](https://arXiv.org/abs/1507.08482) of this paper. From Vedran:

  - In recent times, there has been remarkable progress in (quantum) machine learning, and specifically in the context of, arguably, data analysis: learning properties of (conditional) probability distributions, as is traditionally done using supervised, unsupervised and related modes of learning.

  - Reinforcement learning is often neglected. I grant that "big data'' applications, arguably, have more immediate value in the modern data-driven world. We are, however, driven by the (further out-of-reach) potential of AGI (artificial general intelligence), that is, human level intelligence. AGI naturally requires the capacities to generalize from examples and to identify structures or rules in data, just as is done in the majority of (Q)ML. However, these specialized aspects clearly do not suffice for AGI. We stick to the viewpoint the missing link between AGI of the future, and data-driven ML of today, can be formulated and investigated in the so called agent-environment paradigm for AI (see e.g. Russell and Norvig textbook on AI), which, as a (from a theory perspective) clean special case has reinforcement learning. We take this paradigm as a means to broach the questions of so-called whole agents (as opposed to specialized agents/devices).

  - From my perspective, the key conceptual contribution of the paper is to "quantize" this agent-environment paradigm -- more precisely, to provide one potential method to do so. It is worthwhile to note that, while our method is not the only possible, it does capture the settings of most of QML.
  From that point on, we focus on the less explored pure RL aspects, and we pluck the lowest of hanging fruits: identify the obviously impossible things (generic quantum speed up in a classical environment, for instance is impossible), where the main (only) contribution is a comparatively rigorous formalization of the otherwise clean concepts.

  - Once the "no-go's" are identified, we are well justified in relaxing many conditions in a quest for improvements in learning related (thus, not just computational) figures of merit.
  If task environments are appropriately "quantized" (which is impossible in most current applications of RL, but bear with me, there will be some saving grace) certain things can be done, and this justifies us to talk in terms of key buzz-words like "quantum enhancement".

  - The key technical contribution is to find a formal way to connect simple results from oracular quantum computation (e.g. Grover search) with anything to do with learning (note, in our perspective, it is important to conceptually separate raw search problems and "genuine learning problems", and this is discussed to some extent in the long paper).
  This is done by formally characterizing environments where fast searching indeed helps: we, essentially "quantize" the exploration phase of RL [think of it as broad exploration of (behaviour) functions performed to avoid local minima] which in turn provides for a more efficient exploitation phase.

  - Another way of viewing the above (not in paper) is to imagine a learning agent which can behave as any out of a parametrized family of learning agents. Each agent in such a family will be optimal for *some* environment (c.f. No Free Lunch Theorems), but in the classical case, finding out which one performs best is not-time efficient - it is better to stick with one agent and train that one. In the quantum case, we can identify indirect properties of the environment (an aspect of property testing) using quantum access faster. This allows us to choose which learning model to apply, which will perform better in a given setting. In some cases, this yields an overall improvement, and "luck favoring environments" are examples where this can be formally proven.

  - I promised to (try to) save (a bit of) grace: one of the visions we have in the group has to do with automated quantum experiments, i.e. nano-scale robots which interact with quantum environments on the quantum level. In such a scenario, quantum control we assume may become possible. Another possibility has to do with so-called model-based agents which use internal models of the environment to perform planning and similar. Since this is internal, this again allows for our results to be applied. This was, to some extent discussed in the long paper.

Coding exercises:

- Teach an agent to learn a reinforcement learning task. [OpenAI Gym](http://gym.openai.com) is a collection of environments in which you can benchmark reinforcement learning algorithms. It was officially announced at NIPS in December 2016. [This](https://github.com/claymcleod/dqn) works fine with Pacman, it uses Keras and Theano for implementing the agent, and it is barely a hundred lines of code.

- Optional: The classical first-person shooter Doom is [one of the possible environments](https://gym.openai.com/envs#doom). Installing the Doom environment is fairly intricate as it is not included by default. Follow [these](https://github.com/peterwittek/qml-rg/issues/8#issuecomment-282140692) instructions to get it right.

Meeting 4
---------
10.00-11.30, 16 March 2017, Seminar Room (201).

Papers:

- LeCun, Y.; Bengio, Y. & Hinton, G. [Deep learning](http://doi.org/10.1038/nature14539). *Nature*, 2015, 521, 436-444. This is a review paper by the three greatest giants in deep learning. The paper gives you an idea of the most important neural network architectures that we use today. The current flood of deep learning was unleashed on us by [this paper on convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Geoff Hinton. It is also well worth reading: you seldom come across a paper that accumulated over 10,000 citations in five years. [This blog post](http://www.asimovinstitute.org/neural-network-zoo/) gives an entertaining overview of neural network architectures. Notes:

  - It would be nice to know what pre-training is (page 439), and how it differs from training.

  - ReLU is typically better in networks with many layers.

  - The idea of the hidden layers doing non-linear transformations to the input so as to be able to do linear separations is (at least for me) very enlightening.

- Wiebe, N.; Kapoor, A. & Svore, K. M. [Quantum Deep Learning](http://arxiv.org/abs/1412.3489). *arXiv:1412.3489*, 2014. This is an insightful paper on stacked Boltzmann machines that highlights many possibilities and limitations of using quantum protocols for learning. It was also one of the first papers to consider Boltzmann machines for quantum-enhanced learning -- since then, this line of research took off and now there are N+1 papers on it.

  - GEQS does "only" present a polynomial speedup with respect to classical algorithms. The number of operations to compute gradients does not scale with the number of layers, while in classical algorithms it scales linearly.

  - GEQAE is additionally optimized using amplitude amplification (see Appendix B)

  - The steps are:

  1. Classical mean-field approximation of Gibbs state and efficient preparation of this state.

  2. Quantum protocol to transform a state closer to the true Gibbs state.

  3. Sampling.

  - The two variants of the algorithms are very similar. The second one assumes that the classical data can be accessed in superposition (either via an oracle or a QRAM).

  - It is unclear when this mean-field approximation is good. It *does* depend on the topology, despite what the conclusions claim. For instance, in Appendix E.5, the authors mention that the overlap with the true Gibbs state is worse.

  - The temperature is defined arbitrarily (p.1, before Eq.(1)). This is in contrast with [our work](https://arxiv.org/abs/1611.08104) on a similar sampling-based approach.

  - The success probability in Lemma 2 approaches 1 if there is no model at all (Corollary 3 on p.30). That looks like a steep assumption.

  - I really don't get how a single ancilla qubit can store a probability or an energy value. See the paragraph before (A7) on p.10 or the details of Algorithm 1.

  - Gate complexity is not discussed, but it probably grows fast for the controlled unitary as the size of the Hilbert space increases.

  - See also [Issue #14](https://github.com/peterwittek/qml-rg/issues/14).


Coding exercises:

- Implement Q-learning from scratch for a [simple toy example](Exercises/04_rl_toy_example.py). The state space is trivial and the agent has full access to it, there is no adversary and the distribution does not change depending on the actions of the agent. After the classical agent, try it with simulated quantum agents. See, for instance, [this paper](https://arxiv.org/abs/1401.4997) or [this one](https://arxiv.org/abs/1601.07358) for clues. The trivial solution is to replace the search by Grover's.

- Optional: Do the same thing with tic-tac-toe against a perfect AI (i.e., you cannot win). See the instructions in the [corresponding file](Exercises/04_tictactoe.py). Here the state and the action space might prove too large for a classical simulation of a quantum agent, so you might want to introduce heuristics to reduce it.

Tutorial 3
----------
16.00-17.30, 21 March 2017, Yellow Lecture Room (247).

The tutorial will be on Python and the scientific ecosystem: using Python for science and machine learning, plotting and visualization, how to write beautiful scientific code, and the best practices of providing a computational appendix to your papers.


Meeting 5
---------
10.00-11.30, 23 March 2017, Seminar Room (201).

Papers:

- Chen, T. & Guestrin, C. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). *Proceedings of KDD-16,  22nd International Conference on Knowledge Discovery and Data Mining*. 2016, 785-794. XGBoost is a simple boosting algorithm for a class of ensemble methods and it has been winning Kaggle competitions. The popularity is not yet evidenced in the citation record, but it is in the [matching GitHub repo](https://github.com/dmlc/xgboost). Boosting is an ancient method, the most well-known example being [AdaBoost](https://link.springer.com/chapter/10.1007/3-540-59119-2_166). Pay attention to how regularization is done. There is a nice introduction on [this page](http://xgboost.readthedocs.io/en/latest/model.html).

- Neven, H.; Denchev, V. S.; Drew-Brook, M.; Zhang, J.; Macready, W. G. & Rose, G. [Binary classification using hardware implementation of quantum annealing](https://www.google.com/googleblogs/pdfs/nips_demoreport_120709_research.pdf). *Demonstrations at NIPS-09, 24th Annual Conference on Neural Information Processing Systems*, 2009, 1-17. Perhaps the earliest implementation of a quantum machine learning algorithm. It relies on one of D-Wave's early annealing chips and exploits nonconvex optimization for a better regularized boosting algorithm.

Coding exercise:

- Crack the annoying APS captcha. A cleaned up data set is available as a [zip](Exercises/aps_captcha_images.zip), along with a [Python file to load the images](Exercises/tools.py). Use a convolutional neural network like [LeNet in Keras](http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/). You definitely do not need [Inception](https://arxiv.org/abs/1602.07261) to crack this. The real-life images contain sheared examples: once you are done with the basic example, turn to this set as testing examples. The labels are given in a text file. You can solve this two ways. 1) Use a hack: APS was stupid enough to include enough information in the images to de-shear them. A function `deshear` is included in the image loader to help you. 2) Do it the deep learning way and [use data augmentation](http://ankivil.com/kaggle-first-steps-with-julia-chars74k-first-place-using-convolutional-neural-networks/). This is a crucially important technique in data science.

Meeting 6
---------
10.00-11.30, 30 March 2017, Seminar Room (201).

On this meeting, we will only discuss one paper. Then we will spend half an hour discussing what we learned over the first six weeks to consolidate our knowledge.

Paper:

- Rebentrost, P.; Mohseni, M. & Lloyd, S. [Quantum Support Vector Machine for Big Data Classification](https://arxiv.org/abs/1307.0471). *Physical Review Letters*, 2014, 113, 130503. Take a look at the [experimental demonstration](https://arxiv.org/abs/1410.1054) too.

Coding exercise:

- Continue working on the APS captcha collection. Do the same thing as the week before, but replacing the neural network. Try the following three algorithms: [XGBoost](https://github.com/dmlc/xgboost), [random forests](https://en.wikipedia.org/wiki/Random_forest), and [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine). The former two are still competitive in Kaggle challenges even in the face of deep neural networks, whereas support vector machines ruled the machine learning landscape for a decade between about 1995 and 2005. [Random forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [support vector machines](http://scikit-learn.org/stable/modules/svm.html) are available as part of Scikit-learn, and XGBoost also [plays along nicely](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py) with the package. Visualize the decision boundaries of the algorithms, along with different choices of parameters and regularization.

Meeting 7
---------
10.00-11.30, 6 April 2017, Seminar Room (201).

Papers:

- Srivastava, N.; Hinton, G. E.; Krizhevsky, A.; Sutskever, I. & Salakhutdinov, R. [Dropout: a simple way to prevent neural networks from overfitting](http://jmlr.org/papers/v15/srivastava14a.html). *Journal of Machine Learning Research*, 2014, 15, 1929-1958. This is a great example of how regularization is done in deep learning. For a prehistoric paper on regularizing neural networks, read [Optimal Brain Damage](https://papers.nips.cc/paper/250-optimal-brain-damage) from 1989.

  - A new 'thinned' network (with some units dropped out) is used for each training case in a mini-batch, and the gradients are averaged over all the training cases in each mini-batch. This makes the training time per mini-batch approximately the same as without dropout.
  - But still training a network with dropout need more time to train (2-3 times the time needed for a regular network without dropout).
  - Also, the size of a dropout network is bigger than a regular network doing the same task. If a regular network has *n* nodes, a good dropout network for the same task should have *~p·n* nodes, where *p* is the probability of dropping out one node.
  - The difference in size is enormous when comparing with Bayesian networks (1000s vs. 10s of nodes)

- Amin, M. H.; Andriyash, E.; Rolfe, J.; Kulchytskyy, B. & Melko, R. [Quantum Boltzmann Machine](https://arxiv.org/abs/1601.02036). *arXiv:1601.02036*, 2016. This paper uses the D-Wave machine for Gibbs sampling to train Boltzmann machines. Unlike some other proposals that suggest using this hardware-based sampling for increasing connectivity (and thus complexity), the authors used an actually quantum Hamiltonian and analyzed the outcome.

Coding exercise:

- Simulate a full quantum support vector machine. It has many components, so you might want to consider distributing the workload across the programming groups. For an example on the HHL, look at [this notebook](https://github.com/mariaschuld/phdthesis/blob/master/QLSE%20algorithm.ipynb). Once you have the simulation ready, downsample the APS captcha collection to a ridiculously low resolution (say, 2x2), and train your QSVM on the collection. You could also use a classical autoencoder instead of raw downsampling. Or a quantum one.

Tutorial 4
----------
16.00-17.30, 18 April 2017, Yellow Lecture Room (247).

The tutorial will be on advanced data science, covering data collection, filtering, cleaning, and visual analysis. We will study whether arXiv metadata alone is predictive enough to tell the impact factor of the journal where the manuscript will be published.

Meeting 8
---------
10.00-11.30, 20 April 2017, Seminar Room (201).

Papers:

- Sutskever, I.; Vinyals, O. & Le, Q. V. [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks). *Advances in Neural Information Processing Systems*, 2014, 27, 3104-3112. [Long short-term memory](https://dx.doi.org/10.1162%2Fneco.1997.9.8.1735) has been used for two decades for sequence learning, and this paper makes it deep. Here is a [decent explanation](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), and here is a [decent implementation](https://github.com/farizrahman4u/seq2seq).

- Schuld, M.; Fingerhuth, M. & Petruccione, F. [Quantum machine learning with small-scale devices: Implementing a distance-based classifier with a quantum interference circuit](https://arxiv.org/abs/1703.10793). *arXiv:1703.10793*, 2017. There is no connection to the classical paper, but it just came out and it is a really fun paper. It flips the perspective: instead of trying to come up with an abstract formulation of a quantum-enhanced learning protocol that needs a million qubits, a universal quantum computer, a QRAM, plus an oracle just in case, this manuscript takes the IBM Quantum Experience as the starting point and looks at what kind of learning can be done with it.

Coding exercise:

- Do what everyone who learns machine learning coming from science background does first: predict stock prices. Grab a data set (e.g. [Dow Jones Index](https://archive.ics.uci.edu/ml/datasets/Dow+Jones+Index)) and unleash a [phased LSTM](https://github.com/dannyneil/public_plstm) on it. The [paper on phased LSTMs](https://arxiv.org/abs/1610.09513) was published on NIPS last year, trying to address mid-term memory. Ordinary LSTMs are routinely deployed on stock prices, for instance, see [this kernel](https://www.kaggle.com/pablocastilla/d/dgawlik/nyse/predict-stock-prices-with-lstm) on Kaggle. I take a 10% cut if you make money on this.

Meeting 9
---------
10.00-11.30, 27 April 2017, Seminar Room (201).

Paper:

- Wattenberg, M.; Viégas, F. & Johnson, I. [How to Use t-SNE Effectively](https://doi.org/10.23915/distill.00002). *Distill*, 2016. Manifold learning as it is known, took off with [Isomap](https://doi.org/10.1126/science.290.5500.2319), although there were some precursors to it, like [self-organizing maps](https://en.wikipedia.org/wiki/Self-organizing_map), that used a two-dimensional grid of neurons to do an embedding. The original [t-SNE paper](http://www.jmlr.org/papers/v9/vandermaaten08a.html) appeared in 2008, and it became the most popular manifold learning method. It is, however, not easy to get it right, and this interactive paper gives insights on the inner workings of the algorithm. [Add the Jonker-Volgenant algorithm](https://blog.sourced.tech/post/lapjv/), and you have visualizing superpowers. Submitting to Distill means send a pull request on GitHub, which also means that [this paper is on GitHub](https://github.com/distillpub/post--misread-tsne). Got questions? [Open an issue](https://github.com/distillpub/post--misread-tsne/issues). Comments during the presentation:

  - Global features are typically useless, so t-SNE and other visualization methods minimize cost functions that have big penalty for mapping close points to distant points, but not for mapping distant points to close points.

  - t-SNE is useful for getting an intuition on the raw, unlabeled data, as well as for analyzing the representations that NN create.


Coding exercise:

- Assume that cats and dogs lie on a high-dimensional manifold. Get the images from the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) data set. Embed the manifold in two-dimensions with a globally optimal method (SVD or MDS), and three local methods (Isomap, spectral embedding, and t-SNE). Plot sample images along with the actual points. Scikit-learn has a [handy tutorial](http://scikit-learn.org/stable/modules/manifold.html) on this. There is another [awesome explanation](https://colah.github.io/posts/2014-10-Visualizing-MNIST/) in 2 and 3D. Then do the same thing, but first train a CNN on the images, and visualize the last representation layer before the ordinary FNN part. [Here is](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html) a tutorial on the raw output, and [here is](https://colah.github.io/posts/2015-01-Visualizing-Representations/) a blog post that uses manifold learning to visualize the abstract representation. Following [this comment](https://github.com/peterwittek/qml-rg/commit/94af3599969d04c63a0bbec2a3ab8f40c40f1ab6#commitcomment-21929565), it is a good idea to pull off a pre-trained model from [keras.applications](https://keras.io/applications/).

Meeting 10
---------
11.00-13.00, 04 May 2017, Yellow Lecture Room (247).

Papers:

- Zheng, S.; Jayasumana, S.; Romera-Paredes, B.; Vineet, V.; Su, Z.; Du, D.; Huang, C. & Torr, P. H. S. [Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/abs/1502.03240). *Proceedings of ICCV-15, International Conference on Computer Vision*, 2015, 1529-1537. This is an important paper that draws a parallel between probabilistic graphical models (here Markov networks and conditional random fields) and neural networks.

- Benedetti, M.; Realpe-Gómez, J.; Biswas, R. & Perdomo-Ortiz, A. [Quantum-assisted learning of graphical models with arbitrary pairwise connectivity](https://arxiv.org/abs/1609.02542). *arXiv:1609.02542*, 2016. In probabilistic graphical models, both learning and inference are computationally expensive. This paper looks at how we can embed arbitrary graphs in a contemporary quantum annealing architecture to learn the structure of a model efficiently.

Coding exercise:

- Take the image of Einstein (or Marie Curie) from the APS Captcha collection. Train a simple Markov random field to reproduce the image based on the gradients described in Benedetti et al., 2016. Then use MCMC Gibbs sampling and simulated thermal state sampling to infer the full image based on a partial input. For the former, you can use [this educational package](https://github.com/tbabej/gibbs).

Tutorial 5
----------
16.00-17.30, 09 May 2017, Yellow Lecture Room (247).

We will go through the different functions of Kaggle, promoting you from [Novice to Contributor](https://www.kaggle.com/progression). It is assumed that you are able to solve the exercises given in the first four tutorials.

Meeting 11
----------
11.00-13.00, 11 May 2017, Yellow Lecture Room (247).

Paper:

- Kerenidis, I. & Prakash, A. [Quantum Recommendation Systems](https://arxiv.org/abs/1603.08675). *arXiv:1603.08675*, 2016. Recommendation systems go back to a sparse matrix completion problem, for which this is a fun quantum protocol.

Meeting 12
----------
11.00-13.00, 18 May 2017, Yellow Lecture Room (247).

Paper:

- Goodfellow, I. J.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A. & Bengio, Y. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). *arXiv:1406.2661*, 2014. This is the hottest topic in ML today. See also [a tutorial](https://arxiv.org/abs/1701.00160) by I. Goodfellow.

Meeting 13
----------
11.00-13.00, 25 May 2017, Aquarium (280).

Paper:

- Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever, [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864). *arXiv:1703.03864*, 2017. This paper shows how an old idea, called Evolution Strategies, allows for strong improvement compared to standard Reinforcement Learning methods. Points noted:
  - It works well for continuous action spaces and delayed rewards.
  - The objective function F includes a layer of stochasticity by evaluating a perturbed set of parameters over a single (random) episode.
  - There is no global heuristics involved, the optimization is based on SGD.
  - Scales linearly with the number of cores involved because the communication overhead is minimal (contrast to backprop).
  - Backprop would focus on the sequence of actions (and estimating the value function) as opposed to the parameters of the policy.
  - See also the [popular science write-up](https://blog.openai.com/evolution-strategies/).
  - A reference implementation in TensorFlow is [here](https://github.com/openai/evolution-strategies-starter).

Meeting 14
----------
11.00-13.00, 01 June 2017, Yellow Lecture Room (247).

Paper:

- Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, [Mask R-CNN](https://arxiv.org/abs/1703.06870). *arXiv:1703.06870*, 2017.
