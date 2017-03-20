See also [Issue #10](https://github.com/peterwittek/qml-rg/issues/10).

- $a = {a_x: x=1,\ldots, N}$ is assumed to be normalized. This can be an expensive operation.

- HHL in a CV setting. Many, if not most QML proposals use HHL. Among other things, HHL assumes that the matrix to be inverted is well-conditioned, which roughly means it is far from being singular.
