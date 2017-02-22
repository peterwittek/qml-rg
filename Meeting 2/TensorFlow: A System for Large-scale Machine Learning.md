Some remarks:

- Machine learning is mainly a bunch of linear algebra. This is what enables atomizing operations as TensorFlow does, and it also explains a large number of QML algorithms. You can also consider TensorFlow as an alternative to BLAS and LAPACK, and implement normal scientific workflows.

- The complexity of TensorFlow is astonishing: it can be deployed on anything from FPGAs through ASICs to GPUs and CPUs. If this much software engineering went into quantum simulation libraries...

- Finally a paper that says at least implicitly that MapReduce was the wrong paradigm, and Spark does not help much (see section on Batch dataflow systems).

- Despite the claim that inference is expensive, it is actually quite cheap: 5 billion FLOPS = 5 GFLOPS. The Titan X consumer-grade GPU can do 6 TFLOPS, that is, 1200 times more. This is in sharp contrast to probabilistic graphical models, where inference is #P-complete.

- The dataflow graph is deterministic. In Julia, [Transformations.jl](https://github.com/JuliaML/Transformations.jl) allow more freedom.

- Check the datatypes: 32-bit representation wins out. Internally, Microsoft uses 3 bits for a single weight in a neural network. This is a far cry of the default 64-bit precision of contemporary CPUs, and it also explains why consumer-grade GPUs work better than GPUs designed for scientific workflows.

- The data structures must be dense. This is primarily because of GPUs. To address sparse models, they have a method for sparse embedding (Section 4.2).

