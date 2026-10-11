---
title: "Multiple Linear Regression"
description: "Extending linear regression to many input features at once."
date: 2026-10-11
weight: 8
status: "growing"
tags: [ml, regression]
---

## 1. Introduction

Suppose you're trying to guess the price of a house before it's listed. If you only knew its **size**, you could make a rough guess — bigger house, higher price. But you'd be wrong a lot. Two houses of the same size can sell for very different prices depending on the **number of bedrooms**, the **age of the house**, and **how good the neighborhood is**.

A single input isn't enough. You want to combine several pieces of information at once and let each one pull the prediction up or down by the right amount. That's exactly the problem **Multiple Linear Regression (MLR)** solves.

**Formal definition:** Multiple Linear Regression is a method that predicts one **dependent variable** (the thing you want to know — here, price) using two or more **independent variables** (the things you already know — size, bedrooms, age), assuming the relationship between them is a straight-line (linear) one: each input contributes a fixed amount per unit, and the contributions just add up.

If you've heard of **Simple Linear Regression** (one input, one output, a straight line through a scatter plot), MLR is the natural extension: same idea, more inputs.

---

## 2. History

The story of MLR starts not with houses, but with **planets**.

In the early 1800s, astronomers were trying to predict the orbits of comets and asteroids from telescope measurements. The problem was that every measurement had a small amount of error — the telescope, the observer, the atmosphere, all introduced noise. Given several noisy, slightly-inconsistent observations, how do you find the _single best_ orbit that explains them all?

In 1805, the French mathematician **Adrien-Marie Legendre** published a technique called the **method of least squares** — choose the curve that minimizes the total squared distance between the predicted and observed values. Shortly after, **Carl Friedrich Gauss** claimed he had been using the same method privately since 1795, when he used it to correctly predict the position of the dwarf planet **Ceres** after it had been lost from view. Their priority dispute is famous in the history of math, but the important part is this: least squares was born out of a very practical need to fit a line (or curve) through noisy data — which is precisely what regression still does today.

At this stage, though, the method was mostly used with **one predictor at a time**. The real shift came later in the 19th century with **Sir Francis Galton**, a scientist studying how children's heights related to their parents' heights. He noticed that unusually tall parents tended to have children who were tall, but usually _not as tall_ — their heights "regressed" back toward the average. He called this phenomenon **regression toward the mean**, and the name stuck to the whole family of techniques, even though modern regression is about much more than this one observation.

Around the same time, statisticians realized that most real outcomes don't depend on just one cause. A crop's yield depends on rainfall _and_ fertilizer _and_ temperature, not any single one of them. **Karl Pearson** and later **Ronald Fisher**, working in the early 1900s, formalized the mathematics of correlation and regression with multiple variables, expressing everything in terms of matrix algebra — which made it possible, at least in principle, to fit a line using any number of inputs at once.

The "in principle" was the catch: solving these matrix equations by hand for more than two or three variables was brutally tedious. It wasn't until **digital computers** became available in the mid-20th century that Multiple Linear Regression became a practical, everyday tool — first in statistics and economics, and later, as computing kept getting cheaper, as one of the very first algorithms taught in machine learning, because it's simple, fast, and a natural stepping stone toward more complex models like neural networks.

---

## 3. Core Concepts

### 3.1 The Building Blocks

|Term|Plain-English Meaning|
|---|---|
|**Dependent variable** (target / label)|The thing you're trying to predict. Usually written $y$.|
|**Independent variable** (feature / predictor)|An input you use to make the prediction. Usually written $x$.|
|**Coefficient** (weight)|A number that says how much a feature pulls the prediction up or down per unit.|
|**Intercept** (bias)|The baseline prediction when all features are zero.|

### 3.2 From One Feature to Many

**Simple Linear Regression** (one feature):

$$y = mx + b$$

This draws a single straight line through a 2D scatter plot of $x$ vs $y$.

**Multiple Linear Regression** (many features) simply extends this pattern — one more term per feature:

$$y = b_0 + b_1x_1 + b_2x_2 + \dots + b_px_p$$

Nothing new conceptually happens when you go from 1 feature to 10 — you're just adding more "$\text{weight} \times \text{feature}$" terms before summing them up.

||Simple Linear Regression|Multiple Linear Regression|
|---|---|---|
|Number of features|1|2 or more|
|Shape of the model|A straight line|A flat plane (2 features) or a "hyperplane" (3+ features)|
|Equation|$y = mx + b$|$y = b_0 + b_1x_1 + \dots + b_px_p$|
|Example use case|Predict price from size alone|Predict price from size, bedrooms, age, location|

![Simple vs multiple](/images/notes/simple_vs_multiple.png)

> Simple regression (size only) vs multiple regression (size + bedrooms) on the same houses

### 3.3 The Geometric Picture

With **1 feature**, you're fitting a line through points on a 2D graph (x, y).

With **2 features**, you're fitting a flat plane through points floating in 3D space (x1, x2, y):

```
        y (price)
        |
        |        *
        |     *
        |   ______________  <- the fitted plane
        | *
        |________________________ x1 (size)
       /
      / x2 (bedrooms)
```

![MLR 3D plane fit](/images/notes/mlr_3d_plane_fit.png)

> The four houses plotted in 3D with the fitted regression plane

With **3 or more features**, you can no longer draw it, but the math works exactly the same way — statisticians call this generalized flat surface a **hyperplane**. "Hyper" just means "in more than 3 dimensions, which we can't visualize, but can still compute."

### 3.4 The Assumptions Behind MLR

MLR isn't magic — it works well only when certain conditions roughly hold. These are the classic assumptions:

|Assumption|Plain-English Meaning|
|---|---|
|**Linearity**|The true relationship really is roughly a straight-line combination of the features, not curved.|
|**Independence of errors**|One prediction's mistake doesn't influence another's (e.g., no hidden time pattern).|
|**Homoscedasticity**|The size of typical prediction error stays roughly constant across the range of inputs, instead of growing for larger values.|
|**Normality of residuals**|The leftover errors (actual − predicted) are roughly bell-curve shaped, useful mainly for statistical significance testing.|
|**No severe multicollinearity**|The features shouldn't be near-duplicates of each other (more on why in Section 6).|

You don't need to memorize these — just know that when a MLR model behaves strangely, one of these assumptions being violated is usually the reason.

---

## 4. The Math, Built Up Slowly

### 4.1 Symbol Table

|Symbol|Meaning|
|---|---|
|$y$|The actual observed value of the target (e.g., real price of a house)|
|$\hat{y}$ ("y-hat")|The model's **predicted** value|
|$x_1, x_2, \dots, x_p$|The $p$ input features|
|$b_0$|The intercept (baseline prediction)|
|$b_1, \dots, b_p$|The coefficients (weights) for each feature|
|$\beta$ ("beta")|Shorthand vector containing all coefficients, $[b_0, b_1, \dots, b_p]$|
|$X$|A matrix holding all the feature values for every data point|
|$Y$|A vector holding all the actual target values|
|$n$|Number of data points (rows)|
|$p$|Number of features (not counting the intercept)|
|$\varepsilon$ ("epsilon")|The error — the gap between what the model predicts and reality|

### 4.2 One Feature, Then Many

Start with one feature:

$$\hat{y} = b_0 + b_1x_1$$

Add a second:

$$\hat{y} = b_0 + b_1x_1 + b_2x_2$$

Generalize to $p$ features:

$$\hat{y} = b_0 + b_1x_1 + b_2x_2 + \dots + b_px_p$$

This is really just a **dot product** plus a constant. Quick refresher: the dot product of two lists of numbers means "multiply matching positions together, then add up the results." So $[b_1, b_2]\cdot[x_1,x_2] = b_1x_1 + b_2x_2$. The whole regression equation is just "dot product of weights and features, then add the intercept."

### 4.3 Packing Everything Into Matrices

Writing this equation out separately for every row of data gets repetitive fast. Instead, statisticians stack **all** the data into a matrix $X$ (adding a column of 1's so the intercept fits the same pattern as the other weights) and all the answers into a vector $Y$:

$$Y = X\beta + \varepsilon$$

This single line represents _every row of your dataset at once_: "the actual values equal the matrix of features times the weight vector, plus whatever error is left over."

### 4.4 How Do We Choose the Best Weights?

We want the $\beta$ that makes the predictions $\hat{y}$ as close as possible to the actual $y$ values, across **all** rows at once. "Close" is measured using the **Sum of Squared Errors (SSE)**:

$$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Why squared?** Two reasons: (1) it turns negative errors (predicted too low) and positive errors (predicted too high) into equally "bad" positive numbers, so they don't cancel out, and (2) it punishes big mistakes much more than small ones — being off by 10 is worse than being off by 1, ten times over even before squaring, and 100 times worse after squaring.

![Residuals SSE](/images/notes/residuals_sse.png)

> Residuals: the vertical gaps between actual and predicted price that SSE squares and sums

### 4.5 The Normal Equation

Calculus tells us that a smooth curve is at its lowest point exactly where its slope is zero. The "slope" here is with respect to _each_ coefficient — so finding the best $\beta$ means solving the whole system of equations where every one of those slopes hits zero at once. Doing that algebra (we won't derive it line-by-line here, but it's a standard calculus result) for all coefficients simultaneously produces this closed-form formula, called the **normal equation**:

$$\hat{\beta} = (X^TX)^{-1}X^TY$$

Piece by piece:

- $X^T$ means the **transpose** of $X$ — flip its rows and columns.
- $X^TX$ multiplies the transposed matrix by itself — this produces a small square matrix summarizing how every feature relates to every other feature (including itself).
- $(X^TX)^{-1}$ is the **matrix inverse** — the matrix equivalent of "dividing by a number." Just like $5 \times \frac{1}{5} = 1$, a matrix times its inverse gives the identity matrix (matrix version of "1").
- $X^TY$ summarizes how each feature relates to the actual target values.
- Multiplying it all together gives you the exact set of weights that minimizes the SSE.

### 4.6 Plugging In Real Numbers

Here's a tiny made-up dataset, predicting house **price** (in lakhs) from **size** (in units of 100 sq ft) and **bedrooms**:

|House|Size ($x_1$)|Bedrooms ($x_2$)|Price ($y$)|
|---|---|---|---|
|1|10|2|300|
|2|15|2|350|
|3|15|3|400|
|4|20|3|450|

Adding a column of 1's for the intercept, our matrix $X$ and vector $Y$ are:

$$X = \begin{bmatrix} 1 & 10 & 2 \ 1 & 15 & 2 \ 1 & 15 & 3 \ 1 & 20 & 3 \end{bmatrix} \qquad Y = \begin{bmatrix} 300 \ 350 \ 400 \ 450 \end{bmatrix}$$

Computing $X^TX$ (each entry is a sum of products across the 4 rows):

$$X^TX = \begin{bmatrix} 4 & 60 & 10 \ 60 & 950 & 155 \ 10 & 155 & 26 \end{bmatrix}$$

Computing $X^TY$:

$$X^TY = \begin{bmatrix} 1500 \ 23250 \ 3850 \end{bmatrix}$$

Solving $(X^TX)\beta = X^TY$ gives:

$$b_0 = 100, \quad b_1 = 10, \quad b_2 = 50$$

**You can hand-verify this** without inverting anything — just plug these numbers into the three equations that $(X^TX)\beta = X^TY$ represents, and check both sides match:

- Row 1: $4(100) + 60(10) + 10(50) = 400 + 600 + 500 = 1500$ ✓ (matches $X^TY$ row 1)
- Row 2: $60(100) + 950(10) + 155(50) = 6000 + 9500 + 7750 = 23250$ ✓
- Row 3: $10(100) + 155(10) + 26(50) = 1000 + 1550 + 1300 = 3850$ ✓

All three check out, so $\beta = [100, 10, 50]$ is confirmed correct. Now plug it back into the original equation to double check the predictions match the actual prices exactly:

$$\hat{y} = 100 + 10x_1 + 50x_2$$

- House 1: $100 + 10(10) + 50(2) = 100+100+100 = 300$ ✓
- House 2: $100 + 10(15) + 50(2) = 100+150+100 = 350$ ✓
- House 3: $100 + 10(15) + 50(3) = 100+150+150 = 400$ ✓
- House 4: $100 + 10(20) + 50(3) = 100+200+150 = 450$ ✓

Every prediction matches perfectly — because this toy data was built from an exact linear rule with no noise. Real data will almost never fit this perfectly; there will usually be leftover error ($\varepsilon$) even with the best possible $\beta$.

---

![MLR 3D plane fit 1](/images/notes/mlr_3d_plane_fit%201.png)

> The exact fitted plane from the worked example|361

## 5. What The Math Is Actually Doing

Underneath all the matrix notation, the normal equation is just doing something very intuitive: it's asking "for every combination of weights I could choose, how far off would my total predictions be, squared and added up — and which single combination makes that total as small as possible?" The matrix algebra is a shortcut that jumps straight to that best combination instead of guessing-and-checking millions of possibilities. In the house example, the model effectively learned that every extra "100 sq ft unit" of size is worth 10 lakhs, every extra bedroom is worth 50 lakhs, and even a house with zero size and zero bedrooms (a nonsensical but mathematically necessary baseline) would start at 100 lakhs — and it found those exact numbers by minimizing total squared error across every house at once, not by looking at any single house in isolation.

---

## 6. Types / Variants

Plain Multiple Linear Regression is the starting point, but it has well-known cousins built to fix specific weaknesses:

|Variant|Analogy|What It Changes|
|---|---|---|
|**Ordinary MLR**|A straight ruler laid across your data points|Minimizes SSE only, no extra constraints|
|**Polynomial Regression**|Bending that ruler into a curve|Adds squared/cubed versions of features ($x^2, x^3$) — still "linear" in the _weights_, just curved in the _features_|
|**Ridge Regression**|A ruler with a rubber band pulling all weights gently toward zero|Adds a penalty for large coefficients (L2 penalty) — reduces instability from multicollinearity|
|**Lasso Regression**|A ruler that will happily snap some weights to exactly zero|Adds a penalty that can eliminate less useful features entirely (L1 penalty) — doubles as feature selection|
|**Elastic Net**|A ruler with both a rubber band and a snapping mechanism|Combines Ridge's and Lasso's penalties together|

---

![Regularization shrinkage](/images/notes/regularization_shrinkage.png)

> Ridge shrinks coefficients toward zero; Lasso can zero them out entirely

## 7. Limitations — Worked Examples

### 7.1 Multicollinearity: When Features Copy Each Other

![Multicollinearity](/images/notes/multicollinearity.png)

> Left: two perfectly correlated features. Right: infinitely many (b1, b2) fit equally well

Suppose two features are near-duplicates. Take this tiny dataset where $x_2$ is always exactly $2 \times x_1$:

|Row|$x_1$|$x_2$|$y$|
|---|---|---|---|
|1|1|2|3|
|2|2|4|6|
|3|3|6|9|

The true rule could be $y = 3x_1$, or equally, $y = 1.5x_2$, or a mix like $y = x_1 + x_2$ (check: $1+2=3$ ✓, $2+4=6$ ✓, $3+6=9$ ✓). All of these fit the data **perfectly** — meaning there is no single "correct" answer for how much credit belongs to $x_1$ versus $x_2$. The math has infinitely many valid solutions, and small amounts of noise can flip the coefficients from strongly positive to strongly negative between two runs of the same model on slightly different data. This is what multicollinearity does in practice: the model's _predictions_ stay accurate, but the _individual coefficients_ become meaningless and unstable — dangerous if you're trying to interpret "how much does bedroom count matter," rather than just predict price.

**Fix:** Ridge regression's penalty discourages any one coefficient from growing huge to compensate for another, which stabilizes this exact situation. Alternatively, you can just drop one of the redundant features.

### 7.2 Overfitting: Too Many Features, Too Little Data

![Overfitting](/images/notes/overfitting.png)

> A simple fit that generalizes vs a high-degree fit that overfits the training points

If you have, say, 3 data points but use 3 features (plus an intercept — 4 total parameters), the model has enough flexibility to fit those 3 points _exactly_, even if the points were pure random noise with no real underlying pattern. It isn't "learning" anything general — it's just memorizing. On brand-new houses it hasn't seen, its predictions can be wildly wrong despite a perfect fit on the training data.

**Fix:** Collect more data relative to the number of features, use regularization (Ridge/Lasso), or use fewer, more meaningful features.

### 7.3 Other Limits

- MLR assumes a straight-line relationship — if price actually rises steeply then plateaus, MLR will fit it poorly unless you add polynomial terms.
- A few extreme outliers can drag the whole fitted plane toward them, since squared error punishes big misses heavily.
- Extrapolating far outside the range of your training data (predicting the price of a 10,000 sq ft mansion when your data tops out at 2,000 sq ft) is unreliable — the straight-line relationship may not hold that far out.

---

## 8. Connecting This To The Bigger Picture

Even without having covered other topics yet in this chat, it's worth knowing where MLR sits in the wider landscape you're heading toward:

- A **perceptron** (the basic unit of a neural network, loosely inspired by a biological neuron) computes exactly the same weighted-sum-plus-bias formula as MLR — $b_0 + b_1x_1 + \dots + b_px_p$ — and then usually passes it through an "activation function." If you remove the activation function (or use the simplest possible one, the identity function), a single perceptron _is_ Multiple Linear Regression.
- A **neural network** is, at its core, many of these weighted sums stacked and chained together with nonlinear activation functions in between, which is what lets it model curved, complex relationships that plain MLR cannot.
- The normal equation gives an exact, one-shot solution for MLR because the SSE cost function has a nice bowl shape (mathematically: it's _convex_). Neural networks don't have this luxury — their cost functions are far more complicated, so instead of solving directly, they use **gradient descent**, an iterative "take small steps downhill" method. Learning gradient descent next will make much more sense once you've seen the "downhill" shape MLR's own cost function has.

In short: MLR is the simplest possible neural network, and the least squares problem it solves is the friendliest possible version of the optimization problem every deep learning model is ultimately trying to solve.

---

## 9. Quick Recap Table

|Concept|Meaning|
|---|---|
|Dependent variable ($y$)|What you're predicting|
|Independent variable ($x$)|An input used to predict $y$|
|Coefficient ($b_1, b_2, \dots$)|How much each feature moves the prediction|
|Intercept ($b_0$)|Baseline prediction when all features are 0|
|$X$ matrix|All feature values for all rows, stacked together|
|$Y$ vector|All actual target values|
|Sum of Squared Errors (SSE)|Total squared gap between predictions and reality|
|Normal equation|$\hat{\beta} = (X^TX)^{-1}X^TY$ — the exact formula for the best-fit weights|
|Multicollinearity|Features that are near-duplicates of each other, causing unstable coefficients|
|Overfitting|Model memorizes training data instead of learning a general pattern|
|Ridge / Lasso / Elastic Net|Regularized versions of MLR that penalize large or unnecessary coefficients|

---

## 10. Key Takeaway

> Multiple Linear Regression predicts an outcome by combining several inputs, each weighted by how much it matters, into one straight-line (or flat-plane, or hyperplane) equation — and it finds the _single best_ set of weights by minimizing total squared prediction error across all the data at once, using the closed-form normal equation. It's simple, interpretable, and exact when its assumptions hold, but it becomes shaky when features overlap with each other (multicollinearity) or when there isn't enough data to support the number of features used (overfitting). Most importantly, it's not just a standalone statistics tool — it's the mathematical seed from which perceptrons, and eventually entire neural networks, grow.