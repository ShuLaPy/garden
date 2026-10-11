---
date: 2026-10-06
status: seedling
title: Probability & Statistics
weight: 3
tags: [math, statistics]
---

Models never see the "true" world, only noisy samples of it. Probability gives us the tools to reason about that uncertainty.

## Expectation and variance

For a random variable $X$:

$$
\mathbb{E}[X] = \sum_x x \, P(X = x) \qquad \operatorname{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]
$$

## Bayes' theorem

$$
P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}
$$

- $P(A)$ — the **prior**, what we believed before seeing data
- $P(B \mid A)$ — the **likelihood**
- $P(A \mid B)$ — the **posterior**, our updated belief

## The normal distribution

$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

{{< callout type="warning" >}}
Many methods *assume* normally distributed errors. Always check the residuals before trusting that assumption.
{{< /callout >}}
