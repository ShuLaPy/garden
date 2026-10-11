---
title: "Convergence Algorithm and the Learning Rate (α)"
linkTitle: "Convergence and Learning Rate"
slug: "04-convergence-algorithm-and-the-learning-rate"
description: "How big each gradient descent step should be, and how to know when to stop."
date: 2026-10-11
weight: 5
status: "growing"
tags: [ml, optimisation, gradient-descent]
---

## 1. Introduction

Imagine you're hiking down a mountain in thick fog. You can't see the bottom, but you can feel the slope of the ground under your feet — so your strategy is simple: at every point, take a step in whichever direction feels steepest downhill. Sooner or later, the ground under you starts feeling flat. That's your signal: you've probably reached the valley floor, and there's no point taking more steps. You stop.

This is _exactly_ what gradient descent does when training a model like Simple Linear Regression — except instead of a mountain, it's walking downhill on the cost surface $J(\theta_0, \theta_1)$ from your earlier notes. Two questions naturally follow: **how big should each step be?** and **how do you know when to stop walking?**

The first question is answered by the **learning rate ($\alpha$)** — a number that controls the size of every step gradient descent takes. The second question is answered by a **convergence algorithm** — a rule that tells you when the parameters have settled close enough to the minimum that further steps aren't worth taking.

Formally: **A convergence algorithm is a stopping rule used alongside an iterative optimization method (like gradient descent) to determine when the model's parameters have stabilized near the minimum of the cost function, so training can stop.**

---

## 2. History

The story of convergence rules is really the second half of the gradient descent story from your Cost Function notes — Cauchy described the "walk downhill" idea back in 1847, but for over a century afterward, nobody had a rigorous, mathematical guarantee that this simple step-by-step process would actually _reach_ the bottom, rather than wandering forever or overshooting endlessly.

That changed in 1951, when two statisticians, **Herbert Robbins and Sutton Monro**, published a landmark paper introducing what's now called the **Robbins–Monro algorithm** — widely considered the first rigorous proof that an iterative, step-taking process (much like gradient descent) could be _guaranteed_ to converge to the correct answer, provided the step sizes shrank in a carefully controlled way over time. This was a turning point: it transformed "gradient descent seems to work in practice" into "gradient descent is mathematically guaranteed to work under these specific conditions." Their insight — that the _sequence_ of step sizes matters, not just their initial value — laid the groundwork for how we think about learning rates today.

For decades, this remained mostly a statistical curiosity, applied to small optimization problems. But once neural networks and backpropagation became practical in the 1980s and 90s (as covered in your Cost Function notes), training set sizes and parameter counts exploded — and a single fixed learning rate started causing real, practical headaches: too large, and huge networks would diverge instead of learn; too small, and training could take days or weeks longer than necessary.

This drove a wave of research into smarter, _adaptive_ convergence strategies. In 2011, researchers introduced **Adagrad**, which automatically shrinks the learning rate for parameters that have already changed a lot, and grows it for parameters that haven't moved much. In 2012, **RMSprop** refined this idea to work better on the messy, non-convex cost surfaces typical of deep learning. Then in 2014, **Adam (Adaptive Moment Estimation)** combined the best ideas from both, and quickly became — and largely remains — the default choice for training modern deep neural networks, including the very large models used in AI today. Each of these was, at its core, still solving Robbins and Monro's original problem from 1951: how do you choose step sizes that reliably lead to convergence, without wasting time or overshooting the target?

---

## 3. Core Concepts

### 3.1 The Building Blocks

| Term                      | Meaning                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Iteration**             | One full pass of the gradient descent update — compute the gradient, take one step                      |
| **Learning rate (α)**     | A number controlling how big each step is                                                               |
| **Convergence**           | The point where further iterations no longer meaningfully reduce the cost                               |
| **Convergence algorithm** | The rule used to decide when to stop iterating                                                          |
| **Epsilon (ε)**           | A small threshold value used to define "no longer meaningfully improving"                               |
| **Divergence**            | When, instead of settling toward a minimum, the parameters spiral out of control and the cost increases |
| **Oscillation**           | When updates repeatedly overshoot the minimum, bouncing back and forth across it                        |

### 3.2 What Learning Rate (α) Actually Does

Recall the gradient descent update rule from your Cost Function notes:

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) $$

The gradient $\frac{\partial}{\partial \theta_j} J$ tells you the _direction_ of steepest increase in cost (so you move opposite to it). But it doesn't tell you _how far_ to move — that's entirely $\alpha$'s job. Think of the gradient as a compass needle pointing "uphill," and $\alpha$ as the length of your stride in the opposite direction. A tiny stride gets you there safely, but slowly. A huge stride might carry you clean over the valley and up the opposite slope.

> *In above function why we have $\theta_0$, $\theta_1$ passing to cost function?*

Because the **cost function depends on all the model parameters**, not just the parameter you are currently updating.

For a simple linear regression:
$$\hat y = \theta_0 + \theta_1 x$$
- ($\theta_0$) = intercept
- ($\theta_1$) = slope

The cost function, for example Mean Squared Error, is:
$$J(\theta_0,\theta_1)
\frac{1}{m}\sum_{i=1}^{m}  
\left(\theta_0+\theta_1x^{(i)}-y^{(i)}\right)^2  
$$
So (J) needs **both (\theta_0) and (\theta_1)** to calculate the current error.
### Then what does this mean?

Your update rule is:
$$
\theta_j :=  
\theta_j -  
\alpha  
\frac{\partial}{\partial\theta_j}  
J(\theta_0,\theta_1)  
$$
Suppose (j=1):
$$
\theta_1 :=  
\theta_1 -  
\alpha  
\frac{\partial}{\partial\theta_1}  
J(\theta_0,\theta_1)  
$$
Here, you're asking:

> **"If I change (\theta_1), how does the cost change, while considering the current value of (\theta_0)?"**

Similarly, for (j=0):
$$
\theta_0 :=  
\theta_0 -  
\alpha  
\frac{\partial}{\partial\theta_0}  
J(\theta_0,\theta_1)  
$$
You're asking:

> **"If I change (\theta_0), how does the cost change, while considering the current value of (\theta_1)?"**

### Think of it this way

Imagine the cost function as:
$$J(\theta_0,\theta_1)$$
It's like a **3D landscape** where:

- x-axis → ($\theta_0$)
- y-axis → ($\theta_1$)
- z-axis → cost (J)

When updating ($\theta_1$), you need to know **where you currently are on the ($\theta_0$,$\theta_1$) landscape**.

That's why the cost function is written as:

$$\boxed{J(\theta_0,\theta_1)}$$

even though you're taking the derivative with respect to only one parameter:

$$\frac{\partial J}{\partial\theta_1}$$

The key distinction is:

> **($J(\theta_0,\theta_1)$) tells you what the cost depends on.**  
> **($\frac{\partial J}{\partial\theta_1}$) tells you how the cost changes when ($\theta_1$) changes.**

So **all parameters go into the cost function**, while the **partial derivative chooses which parameter you're measuring the effect of**.

### 3.3 Convergence in Plain Terms

A model has "converged" when taking another gradient descent step barely changes $J(\theta_0,\theta_1)$ anymore — meaning the parameters are effectively sitting at (or extremely close to) the bottom of the bowl-shaped cost surface from your earlier notes. Convergence isn't usually an exact, precise stop — it's a practical judgment call: "close enough that more effort isn't worth it."

---

## 4. The Math — Building It Step by Step

### 4.1 Step 1: Why Step Size Matters — Overshoot Intuition

Picture a simple 1D cost curve (like a parabola). At any current position $\theta$, the gradient descent step is:

$$ \theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \frac{\partial J}{\partial \theta} $$

If $\alpha$ is small, $\theta_{\text{new}}$ lands just a little closer to the minimum than $\theta_{\text{old}}$ — safe, but slow. If $\alpha$ is large, the step can carry $\theta_{\text{new}}$ _past_ the minimum entirely, landing on the opposite side of the bowl — and if $\alpha$ is large enough, each subsequent step overshoots even further than the last, and the cost actually starts _increasing_ instead of decreasing. This is divergence.

### 4.2 Step 2: A Rule of Thumb for a "Safe" Learning Rate

For a simple bowl-shaped (convex) cost function, there's a mathematical boundary: if the curvature of the cost function is $L$ (roughly, how sharply it bends), gradient descent is guaranteed not to diverge as long as:

$$ \alpha < \frac{2}{L} $$

**Why this makes sense in plain words:** a sharply curved bowl (large $L$) punishes overshooting more severely, so you need a smaller stride to stay safe. A gently curved, shallow bowl (small $L$) is more forgiving, so a larger stride is still safe. This is why the "right" learning rate isn't a universal constant — it depends on the shape of the specific cost function you're minimizing.

### 4.3 Step 3: Defining the Convergence Test Mathematically

The most common convergence rule compares how much the cost changed between two consecutive iterations:

$$ \left| J^{(t-1)} - J^{(t)} \right| < \epsilon $$

Where $J^{(t)}$ is the cost after iteration $t$, and $\epsilon$ is a small number you choose in advance (e.g. $0.001$). **In plain words:** "if the last step barely improved anything, assume we've arrived, and stop."

Some implementations instead check the size of the gradient itself:

$$ \left| \frac{\partial J}{\partial \theta} \right| < \epsilon $$

**Why this alternative also makes sense:** at the true minimum, the slope of the cost surface is exactly zero (it's flat at the bottom of the bowl) — so a gradient that's shrunk very close to zero is itself strong evidence you've arrived, even before checking how much the cost value changed.

### 4.4 Step 4: Putting It Together — The Full Convergence Algorithm

1. Initialize $\theta_0, \theta_1$ (often randomly, or at zero)
2. Repeat:
    - Compute the gradient of $J$ at the current $\theta$ values
    - Update: $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$
    - Check: has $|J^{(t-1)} - J^{(t)}| < \epsilon$ (or has a maximum iteration limit been reached)?
3. If yes, stop and report the current $\theta$ values as the learned parameters. If no, go back to step 2.

### 4.5 Geometric Interpretation

![Line chart comparing cost J against iteration number for three learning rates: a good rate that smoothly flattens, a too small rate that decreases very slowly, and a too large rate that diverges upward](/images/notes/diagram_alpha_cost_vs_iteration.png)

_The green curve (good $\alpha$) drops quickly and flattens — clear convergence. The blue curve (too small $\alpha$) is still visibly falling at iteration 30 — it will converge eventually, just wastefully slowly. The red curve (too large $\alpha$) doesn't settle at all — it's overshooting the minimum on every step, and the cost is actually increasing over time. This is the exact plot a convergence algorithm is "reading" automatically, instead of a human eyeballing a graph._

![Contour map of the cost surface viewed from above, showing three gradient descent paths: a smooth green path converging to the center, a slow blue path barely moving, and a zigzagging red path bouncing between the walls of the valley](/images/notes/diagram_alpha_contour_paths.png)

_Viewed from directly above the bowl-shaped cost surface (like a topographic map), each ring is a contour of equal cost — rings closer together mean the surface is steeper there. The green path takes efficient, moderate steps straight toward the center (the global minimum, marked with a star). The blue path creeps along, barely making progress with each tiny step. The red path zigzags wildly back and forth across the valley, each step overshooting the previous one — a textbook picture of a learning rate that's too large._

![Bar chart showing the improvement in cost at each iteration, shrinking over time, with a red dashed threshold line marking epsilon, and a green marker showing where the algorithm detects convergence](/images/notes/diagram_convergence_epsilon.png)

_Each purple bar is the improvement made by one gradient descent step — how much smaller the cost got compared to the step before. Early on, improvements are large. As the algorithm nears the minimum, each step improves things less and less. Once the improvement drops below the red $\epsilon$ threshold line, the convergence algorithm declares "close enough" and stops — marked by the green dotted line._

---

## 5. Types / Variants

### 5.1 Convergence Criteria (Ways to Decide "We're Done")

- **Fixed iteration count** is like baking a cake for exactly 30 minutes regardless of how it looks — simple and predictable, but might stop too early or run needlessly long.
- **Cost-change threshold** is like tasting soup every few minutes and stopping once it stops tasting noticeably different from one taste to the next — adaptive, but needs a sensibly chosen $\epsilon$.
- **Gradient-norm threshold** is like checking if the ground has gone flat under your feet while hiking — a direct check on "are we near the bottom," rather than an indirect one based on the cost value.

### 5.2 Learning Rate Schedules (Ways to Choose α)

- **Fixed learning rate** is like driving at a single constant speed the entire trip, regardless of traffic or road conditions — simple, but not adaptive to circumstances.
- **Decaying learning rate** is like slowing your car down automatically the closer you get to your destination, so you don't overshoot the parking spot — the step size shrinks over time on a schedule.
- **Adaptive learning rate (Adagrad, RMSprop, Adam)** is like a smart cruise control that adjusts speed per-lane, in real time, based on how bumpy each specific lane has been recently — different parameters can get different, automatically tuned step sizes.

|Approach|How α behaves|Best suited for|Analogy|
|---|---|---|---|
|**Fixed learning rate**|Stays constant throughout training|Simple models like SLR, small/convex cost surfaces|Constant driving speed|
|**Decaying learning rate**|Shrinks on a fixed schedule (e.g. halved every N iterations)|Longer training runs where fine-tuning near the end matters|Slowing down as you approach your parking spot|
|**Adagrad**|Shrinks automatically per-parameter, based on how much that parameter has already changed|Sparse data, where some parameters update rarely|Cruise control that remembers which lanes were bumpy|
|**RMSprop**|Like Adagrad, but prevents the rate from shrinking too aggressively over long training|Deep neural networks, non-convex surfaces|A cruise control with a "recent memory," not a lifetime one|
|**Adam**|Combines RMSprop-style adaptive rates with momentum (remembering recent step directions)|Most modern deep learning — the default choice|Cruise control that also remembers _which direction_ you've been drifting|

---

## 6. Limitations and Trade-offs

### 6.1 Choosing α Badly Wastes Time or Breaks Training

**Worked example:** Suppose you're training your ice-cream sales model (from the SLR notes) with $\alpha = 0.9$ when a safe value would have been closer to $\alpha = 0.1$. Each gradient step overshoots the minimum by a wide margin, landing on the opposite wall of the cost bowl — and because the bowl is roughly symmetric, the next step overshoots back in the other direction, even further out. Within a handful of iterations, $\theta_1$ might swing from $2.5$ to $-8$ to $19$ to $-51$ — the model doesn't just fail to learn, it actively gets _worse_ with more training, which is a confusing and frustrating failure mode if you don't know to check the cost-vs-iteration plot.

**Fix that came later:** this is exactly why decaying and adaptive learning rates (Adagrad, RMSprop, Adam) were developed — to remove the burden of hand-picking one "magic" fixed $\alpha$ value that works for the entire training run.

### 6.2 A Fixed ε Doesn't Generalize Across Problems

**Worked example:** Say you set $\epsilon = 0.001$ for your ice-cream sales model, where the cost values naturally range in the hundreds. That threshold works well. But now apply that _same_ $\epsilon = 0.001$ to a much bigger dataset where costs range in the millions — a change of $0.001$ is now so relatively tiny that your model will "converge" after just one or two iterations, long before it's actually found a good minimum.

**Fix that came later:** many implementations now use a _relative_ threshold (e.g., "stop when the cost improves by less than 0.01% of its current value") instead of an absolute one, so the same relative logic works regardless of the scale of the problem.

### 6.3 Non-Convex Surfaces Complicate Convergence Checks

As mentioned in your Cost Function notes, deep neural networks have cost surfaces with multiple dips (non-convex), not a single smooth bowl. A convergence algorithm can trigger on a **local minimum** or even a flat "saddle" region that isn't the true best solution — the cost genuinely stops changing much, but not because you've found the best possible parameters.

**Fix that came later:** momentum-based methods (like Adam) help the optimizer "roll through" small flat regions and shallow local minima rather than stopping prematurely, by carrying some of its previous direction and speed forward into the next step.

---

## 7. Connecting to What You Already Know

- **The gradient descent / cost function connection:** convergence is the missing "when to stop" half of the gradient descent story from your Cost Function notes — the update rule $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J$ tells you _how_ to move, and the convergence algorithm tells you _when to stop moving_.
    
- **The SLR connection:** for Simple Linear Regression, the closed-form $\beta_0, \beta_1$ formulas skip the entire convergence question — they jump straight to the answer with algebra. Convergence checking only becomes necessary once you switch to the iterative $\theta$-based approach, which is exactly why your θ vs. β comparison section noted that SLR doesn't _need_ gradient descent, but learning it there matters for later, bigger models.
    
- **The perceptron connection:** early perceptron training also used a learning-rate-like parameter to control how much weights adjusted after each misclassified example — too large a rate could cause the perceptron's decision boundary to swing wildly between training examples, never settling; too small a rate could take an impractically long time to separate the classes.
    
- **The deep learning connection:** modern deep networks are trained for many **epochs** (full passes over the training data), and adaptive optimizers like Adam are essentially running a smart, automatically-tuned convergence process across millions of parameters simultaneously — the same fundamental question ("how big a step, and when to stop") just scaled up enormously from the two-parameter SLR case.
    
- **The machine learning types connection:** convergence algorithms apply broadly across supervised learning — whether you're training a regression model like SLR or a classification model like logistic regression, both ultimately rely on gradient descent (or a variant of it) reaching convergence to produce a usable, trained model.
    

---

## 8. Quick Recap Table

|Concept|Meaning|
|---|---|
|**Learning rate (α)**|Controls the size of each gradient descent step|
|**Convergence**|The point where further training iterations no longer meaningfully reduce cost|
|**Convergence algorithm**|The rule that decides when to stop training|
|**Epsilon (ε)**|The small threshold used to judge "no longer meaningfully improving"|
|**Divergence**|When parameters spiral away from the minimum instead of toward it, usually due to too-large α|
|**Oscillation**|Repeated overshooting back and forth across the minimum|
|**Fixed iteration count**|Stopping after a set number of steps, regardless of how close to convergence|
|**Cost-change threshold**|Stopping once the improvement in cost between steps drops below ε|
|**Gradient-norm threshold**|Stopping once the gradient itself becomes very close to zero|
|**Decaying learning rate**|A schedule that shrinks α gradually as training progresses|
|**Adaptive learning rate (Adam, RMSprop, Adagrad)**|Automatically adjusts α per-parameter based on training history|
|**Robbins–Monro algorithm (1951)**|The first rigorous mathematical proof of convergence for iterative step-based optimization|

---

## 9. Key Takeaway

> A convergence algorithm is simply the disciplined way of answering "have we learned enough yet?" — and the learning rate α is the dial that determines whether you get there efficiently, painfully slowly, or not at all. Too large an α causes overshooting and divergence; too small wastes time creeping toward the answer; a well-chosen (or automatically adapted) α, paired with a sensible stopping rule like an epsilon threshold, is what turns the raw idea of gradient descent from your Cost Function notes into a genuinely practical, reliable training process — the same core mechanism, refined and scaled up, that trains everything from Simple Linear Regression to the largest deep neural networks in use today.



## Resources

- [Machine Learning Algorithms(2) — Convergence Algorithm and Multiple Linear Regression](https://towardsdev.com/machine-learning-algorithms-2-convergence-algorithm-and-multiple-linear-regression-858f37d4e94c)
