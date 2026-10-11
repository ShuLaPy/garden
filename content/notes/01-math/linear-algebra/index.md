---
date: 2026-09-29
status: growing
title: Linear Algebra
weight: 2
tags: [math]
---

Data in ML is stored as vectors and matrices, so linear algebra is the grammar of the whole subject.

## Vectors

A vector $\mathbf{x} \in \mathbb{R}^n$ is an ordered list of $n$ numbers. The **dot product** measures how aligned two vectors are:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert \cos\theta
$$

## Matrices

A dataset with $m$ examples and $n$ features is a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$. Multiplying it by a weight vector gives every prediction at once:

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}
$$

> Matrix multiplication is only defined when the inner dimensions match: $(m \times n)(n \times 1) = (m \times 1)$.

## In NumPy

```python
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 examples, 2 features
w = np.array([0.5, -1.0])

y_hat = X @ w   # array([-1.5, -2.5, -3.5])
```
