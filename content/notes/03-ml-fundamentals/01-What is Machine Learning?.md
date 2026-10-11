---
date: 2026-10-09
status: growing
title: What is Machine Learning?
weight: 2
tags: [ml, intro]
math: true  # math appears inside tabs
---

> A computer program is said to learn from experience *E* with respect to some task *T* and performance measure *P*, if its performance on *T*, as measured by *P*, improves with experience *E*.
>
> — Tom Mitchell, 1997

Instead of writing rules by hand, we show the computer examples and let it **find the rules itself**.

## Types of learning

{{< tabs >}}
  {{< tab name="Supervised" >}}
  Every example comes with a label. The model learns a mapping $f: \mathbf{x} \mapsto y$.

  *Examples:* house-price prediction (regression), spam detection (classification).
  {{< /tab >}}
  {{< tab name="Unsupervised" >}}
  No labels — the model looks for structure on its own.

  *Examples:* customer segmentation (clustering), dimensionality reduction (PCA).
  {{< /tab >}}
  {{< tab name="Reinforcement" >}}
  An agent acts in an environment and learns from rewards.

  *Examples:* game-playing agents, robotics.
  {{< /tab >}}
{{< /tabs >}}

## The ML workflow

{{% steps %}}

### Define the problem

What are we predicting, and how will we measure success?

### Collect and clean data

Usually the longest step. Handle missing values, outliers and inconsistent formats.

### Train a model

Pick an algorithm and fit it on the training set.

### Evaluate

Measure performance on data the model has **never seen**.

### Deploy and monitor

Real-world data drifts, so keep an eye on performance over time.

{{% /steps %}}

{{< callout type="important" >}}
Never evaluate on your training data — it tells you how well the model *memorised*, not how well it *generalises*.
{{< /callout >}}
