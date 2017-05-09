Introduction
============
Kaggle made learning data science less boring. Its sole purpose used to be hosting competitions, but it evolved into an engaging platform to learn about data science, compete for jobs, it provides a great way to interact with data scientists who actually know it better (and also with ones who think they know it better). If you have a shortage of opportunities to interact with complete morons, Kaggle is also here to help. Kaggle gamified the learning process via a progression system, so you have yet another pointless indicator to track, apart from your i10 and h5 indices.

Today Kaggle has a documentation problem: they rolled out features fast, and the system is confusing for beginners. There is a [wiki](https://www.kaggle.com/wiki/Home) and a blog aptly named [No Free Hunch](http://blog.kaggle.com/), but they are not especially well structured for newcomers. In this tutorial, we will go through the various features, and promote you from rookie to less rookie.

A new problem
=============
Kaggle [joined](http://blog.kaggle.com/2017/03/08/kaggle-joins-google-cloud/) Google Cloud. Since then, many things broke and I conjecture that even more data is stolen about your self. The most notable new annoyance is that the Kaggle interface now often freezes Firefox (at least with version 53, but possibly with others too). Use an alternative browser if you can.

Progression system
==================
The Kaggle [progression system](https://www.kaggle.com/progression) makes you feel like a karate champion. Registration earns you a Novice rank, but Contibutor is just a few clicks away. Unfortunately, [it shows](https://www.kaggle.com/rankings): about 78% of the Kagglers are contributors, and the next level is definitely more than a few clicks away. Above Contributor, your progress is divided into categories, and the overall tier is your maximum among all categories.

Let's go through the easy steps:

- Add your bio: slave traders sold you to a quantum theory group.

- Add your location: the boondocks.

- Add your occupation: quantum whateveratician.

- Add your organization: ICFO-光子科学研究所.

- SMS verify your account: this is needed because real money is involved in the competitions, and because of the importance of the rankings in seeking a job. Kaggle rules say there should be one and only one human per account.

- Cast 1 upvote: click on anything you fancy.

Datasets
========
Kaggle hosts a mesmerizing array of small and medium-sized datasets, mainly contributed by users. There is everything from [food facts](https://www.kaggle.com/openfoodfacts/world-food-facts) through [song lyrics](https://www.kaggle.com/mousehead/songlyrics) to [Game of Thrones](https://www.kaggle.com/mylesoneill/game-of-thrones). At least four Pokémon datasets also await your data science prowess. If you look hard enough, there is also a [collection](https://www.kaggle.com/peterwittek/scirate-quant-ph) on papers published in quant-ph.

A common problem with the datasets is that somehow it is always the part missing that you want to investigate. A good example is the IMDB collection: there isn't a single entry with Kevin Bacon. How are you going to investigate the impact of the Bacon number on box office performance then? These datasets are mainly for fooling around and demonstrating algorithms. For any non-trivial work, you probably want to scrape your own data or enter a competition, where datasets were compiled by teams of experts.

You can do two things with a dataset: (i) download it and run analysis it on locally; or (ii) run a kernel on it. The latter is more interesting, so we are discussing this next.

The many meanings of the word 'kernel'
======================================
The word 'kernel' is gruesomely overloaded in maths and computer science. The basic uses that are most relevant to us are as follows:

- The natural meaning: a kernel defines an integral transformation $(Tf)(x)=\int_{y{\in}Y}K(y,x)f(y)dy$.

- Kernel methods like support vector machines, kernel PCA, or kernel k-means, but also Gaussian processes and kernel density estimators: the dot product between data points (or the correlations) can be replaced by a positive semidefinite integral kernel. The connection is via reproducing kernel Hilbert spaces.

- Kernel of a linear map.

- Computational kernel: part of a numerical algorithm that performs the computationally intensive part.

- OS kernel: (i) the stuff that produces the blue screen of death on Windows; (ii) the thing that makes other operating systems work.

- Jupyter kernel: language back-end for notebooks, e.g. Python, R, and Julia kernels.

- Kernel on Kaggle: a script or a notebook run on the cloud infrastructure of Kaggle. They typically run associated with one or more datasets, but there are ones independent of actual data. The name stems from the use of the term in Jupyter.

The cloud infrastructure for running kernels is pretty convenient since it lets you use an extensive set of Python libraries without having to install anything locally on their computer (e.g. Windows users can use XGBoost this way). On the other hand, the cloud instances are [severely limited](https://www.kaggle.com/wiki/Scripts): maximum execution time is ten minutes (forget deep learning), Internet access is fully blocked so you are confined to data and code that are already there, and the RAM is only 8GB.

The next task is to run a kernel. The two straightforward ways to do it:

- The easy way: go to kernels, pick one you like, fork it and run it.

- The less easy way: hit "New Kernel" either under at a dataset or under Kernels, and start coding on your own.

Competitions
============
It is surprisingly easy to enter a competition and get a decent ranking without doing much. The process is as follows:

1. Pick a competition with a low number of participants.

2. Look at the most upvoted discussion. It probably refers to code hosted somewhere with a high-scoring solution.

3. Download the code, run it on the competition files, and submit the results.

I discovered this accidentally when the non-improved variant of a solution got me in the top 40% of the Julia competition (which nobody cared about).

In any case, keep in mind that (i) the network in ICFO cannot handle files larger than zero bytes; and (ii) your computer will eventually process the data. Therefore competitions involving videos or images are not the best first targets, so don't go for the 95GB sea lion collection. The Russian Housing Market is more like it. This will also run on the cloud infrastructure: fork a kernel and run it there.

When you first download the competition dataset, you must accept the conditions of the competition sponsor. This always includes sharing the code of the winning entry with at least the sponsor, but quite possibly an open source licence that permits commercial use. MIT license seems to be a safe bet.
