Alex's comments
- A new 'thinned' network (with some units dropped out) is used for each training case in a mini-batch, and the gradients are averaged over all the training cases in each mini-batch. This makes the training time per mini-batch approximately the same as without dropout.
- But still training a network with dropout need more time to train (2-3 times the time needed for a regular network without dropout).
- Also, the size of a dropout network is bigger than a regular network doing the same task. If a regular network has *n* nodes, a good dropout network for the same task should have *~p·n* nodes, where *p* is the probability of dropping out one node.
- The difference in size is enormous when comparing with Bayesian networks (1000s vs. 10s of nodes)
