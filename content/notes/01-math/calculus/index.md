---
date: 2026-09-22
status: evergreen
title: Calculus
weight: 1
tags: [math, optimisation]
---

Calculus tells us **how a function changes**. In ML, that is exactly what we need to tune parameters: nudge each weight in the direction that lowers the loss.

## Derivatives

The derivative of $f$ at $x$ is the slope of the tangent line:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

A few rules worth memorising:

| Rule | Formula |
|------|---------|
| Power | $\frac{d}{dx} x^n = n x^{n-1}$ |
| Product | $(fg)' = f'g + fg'$ |
| Chain | $\frac{d}{dx} f(g(x)) = f'(g(x))\, g'(x)$ |

{{< callout type="info" >}}
The **chain rule** is the whole idea behind backpropagation in neural networks.
{{< /callout >}}

## Partial derivatives and the gradient

For a function of many variables, the gradient collects every partial derivative into one vector:

$$
\nabla f(\mathbf{w}) = \left[ \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_n} \right]^\top
$$

The gradient points in the direction of **steepest increase**, so we step the opposite way.

## Gradient descent

$$
\mathbf{w} \leftarrow \mathbf{w} - \alpha \, \nabla J(\mathbf{w})
$$

where $\alpha$ is the learning rate.

```python
def gradient_descent(grad, w, lr=0.1, steps=100):
    for _ in range(steps):
        w = w - lr * grad(w)
    return w
```
