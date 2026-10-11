---
date: 2026-10-11
status: seedling
title: Simple Linear Regression
weight: 3
tags: [ml, regression]
---

Simple linear regression models the relationship between **one input** $x$ and a continuous output $y$ with a straight line.

## The model

$$
\hat{y} = \theta_0 + \theta_1 x
$$

- $\theta_0$ — the intercept (value of $\hat{y}$ when $x = 0$)
- $\theta_1$ — the slope (how much $\hat{y}$ changes per unit of $x$)

## Cost function

We measure how wrong the line is with the **mean squared error**:

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

The $\frac{1}{2}$ is only there to make the derivative cleaner.

## Fitting the line

### Closed form

Setting the derivatives to zero gives:

$$
\theta_1 = \frac{\sum_i (x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_i (x^{(i)} - \bar{x})^2}
\qquad
\theta_0 = \bar{y} - \theta_1 \bar{x}
$$

### Gradient descent

Repeat until convergence:

$$
\theta_0 \leftarrow \theta_0 - \alpha \frac{1}{m} \sum_{i} \left( \hat{y}^{(i)} - y^{(i)} \right)
\qquad
\theta_1 \leftarrow \theta_1 - \alpha \frac{1}{m} \sum_{i} \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}
$$

## In code

```python
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

model = LinearRegression().fit(x, y)
print(model.intercept_, model.coef_)   # ≈ 0.05, [1.99]
```

## Evaluating the fit

| Metric | Formula | Read as |
|--------|---------|---------|
| MSE | $\frac{1}{m}\sum (\hat{y} - y)^2$ | Average squared error |
| RMSE | $\sqrt{\text{MSE}}$ | Error in the units of $y$ |
| $R^2$ | $1 - \frac{SS_{res}}{SS_{tot}}$ | Share of variance explained |

{{< callout type="info" >}}
**Next up:** multiple linear regression, where $x$ becomes a vector of features.
{{< /callout >}}
