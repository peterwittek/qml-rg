See also [Issue #14](https://github.com/peterwittek/qml-rg/issues/14).

Alex's comments

- GEQS does "only" present a polynomial speedup with respect to classical algorithms. The number of operations to compute gradients does not scale with the number of layers, while in classical algorithms it scales linearly.

- GEQAE is additionally optimized using amplitude amplification (see Appendix B)

Peter's comments:

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
