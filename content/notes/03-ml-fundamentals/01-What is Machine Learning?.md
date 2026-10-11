---
title: "What is Machine Learning?"
description: "How learning from data differs from hand-written rules, the main types of machine learning, and the ML workflow."
date: 2026-10-11
weight: 2
status: "growing"
tags: [ml, intro]
---

## 1. Introduction: Starting With the Core Idea

Imagine you want a computer to identify whether an email is spam. In **traditional programming**, you would have to sit down and manually write rules:

> "If the email contains the word 'lottery' AND 'click here' AND comes from an unknown sender → mark as spam."

This works for a while, but spammers keep changing their tricks, and soon your rulebook becomes endless, messy, and still fails on new patterns you never thought of.

**Machine Learning (ML) flips this approach entirely.**

> **Traditional Programming:** Rules + Data → Program → Output **Machine Learning:** Data + Output (examples) → Program learns the Rules itself

Instead of _telling_ the computer exactly what to do, you **show it thousands of examples** (spam emails and non-spam emails) and let it **discover the patterns on its own**. Once it has learned these patterns, it can make predictions on emails it has never seen before.

### Formal Definition

> **Machine Learning is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from data and improve their performance on a task over time — without being explicitly programmed with fixed rules for every scenario.**

A more technical (and famous) definition by Tom Mitchell (1997):

> "A computer program is said to **learn** from experience **E** with respect to some task **T** and performance measure **P**, if its performance at task T, as measured by P, improves with experience E."

**In plain words:** the more (good) data/experience you give the model, the better it gets at its job — just like how a person improves at a skill with practice.

---

## 2. Why Machine Learning Is So Powerful

Machine Learning isn't just "another programming technique" — it changes what's even _possible_ to automate. Here's why it matters so much:

### a) It Solves Problems Too Complex for Manual Rules

Some problems (recognizing faces, understanding speech, translating languages, detecting fraud) have so many edge cases and hidden patterns that no human could ever write enough "if-else" rules to cover them all. ML models can uncover these patterns directly from data.

### b) It Improves With More Data

Unlike traditional software (which stays the same unless a developer rewrites it), ML models can **get better automatically** as more data becomes available — no manual rule-rewriting needed.

### c) It Generalizes to Unseen Situations

A well-trained ML model doesn't just memorize the training examples — it learns the **underlying pattern**, so it can make reasonably accurate predictions on **brand-new data** it has never encountered before.

### d) It Powers Modern Technology Everywhere

|Real-World Application|What ML Is Doing|
|---|---|
|Netflix/YouTube recommendations|Predicting what you're likely to watch next|
|Google Maps ETA|Predicting traffic patterns|
|Voice assistants (Siri, Alexa)|Converting speech to text and understanding intent|
|Bank fraud detection|Spotting unusual transaction patterns|
|Medical diagnosis (X-ray/scan analysis)|Detecting disease patterns in images|
|Self-driving cars|Recognizing objects, pedestrians, lanes|
|Spam filters|Classifying emails as spam/not spam|
|ChatGPT/Claude (LLMs)|Predicting the most likely next word/response|

### e) It Scales

Once trained, a single ML model can make millions of predictions per second across the globe — something no team of humans manually reviewing data could ever match in speed or consistency.

---

## 3. The Building Blocks: Key Terminology You Need First

Before diving into types of ML, these terms will come up constantly, so let's define them clearly:

|Term|Meaning|
|---|---|
|**Dataset**|The collection of data used to train and test the model|
|**Features (X)**|The input variables/attributes used to make a prediction (e.g., email text, house size)|
|**Label / Target (y)**|The correct answer the model is trying to predict (e.g., spam/not spam, house price)|
|**Model**|The mathematical system that learns patterns from data (e.g., the perceptron you studied earlier is one of the simplest models)|
|**Training**|The process of showing the model data so it can adjust itself (learn)|
|**Testing**|Evaluating the model's performance on new, unseen data|
|**Prediction/Inference**|The model's output when given new input|
|**Overfitting**|When a model memorizes training data too closely and performs poorly on new data|
|**Underfitting**|When a model is too simple to capture the pattern, performing poorly even on training data|

---

## 4. Types of Machine Learning

This is the heart of understanding ML — almost everything in the field fits into one of these categories.

### A) Supervised Learning

**Definition:** The model learns from **labeled data** — meaning every training example comes with the correct answer already attached. The model's job is to learn the mapping from input (X) to output (y).

**Analogy:** Like a student learning with a teacher who provides both the question and the correct answer, so the student can check their work and improve.

**Two main sub-types:**

|Sub-type|What it Predicts|Example|
|---|---|---|
|**Classification**|A category/class (discrete output)|Is this email spam or not? Is this tumor benign or malignant?|
|**Regression**|A continuous number|Predicting house price, predicting tomorrow's temperature|

**Common algorithms:** Perceptron, Linear Regression, Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), Neural Networks

---

### B) Unsupervised Learning

**Definition:** The model learns from **unlabeled data** — there's no "correct answer" given. Instead, the model tries to find hidden structure, patterns, or groupings within the data on its own.

**Analogy:** Like being given a huge pile of mixed photographs with no labels and being asked to sort them into meaningful groups purely based on similarities you notice.

**Two main sub-types:**

|Sub-type|What it Does|Example|
|---|---|---|
|**Clustering**|Groups similar data points together|Customer segmentation for marketing, grouping similar news articles|
|**Dimensionality Reduction**|Simplifies data by reducing the number of features while preserving important information|Compressing data for visualization, noise reduction|

**Common algorithms:** K-Means Clustering, Hierarchical Clustering, Principal Component Analysis (PCA), DBSCAN

---

### C) Semi-Supervised Learning

**Definition:** A middle ground — the model is trained on a **small amount of labeled data** combined with a **large amount of unlabeled data**. This is useful because labeling data is often expensive and time-consuming, but unlabeled data is usually abundant.

**Analogy:** Like a student who gets a few solved example problems from the teacher, and then has to figure out the rest of a giant stack of unsolved problems mostly on their own, using what little guidance they got.

**Example use case:** Labeling a few hundred medical images by an expert, then using semi-supervised techniques to help the model learn from thousands of additional unlabeled scans.

---

### D) Reinforcement Learning (RL)

**Definition:** The model (called an **agent**) learns by **interacting with an environment**, taking actions, and receiving **rewards or penalties** based on outcomes. Over time, it learns a strategy (**policy**) to maximize its total reward.

**Analogy:** Like training a dog with treats — good behavior gets rewarded, bad behavior doesn't, and over repeated trials the dog (agent) learns the best actions to take in different situations.

**Key components:**

|Term|Meaning|
|---|---|
|**Agent**|The learner/decision-maker|
|**Environment**|The world the agent interacts with|
|**Action**|A choice the agent makes|
|**Reward**|Feedback signal (positive or negative) after an action|
|**Policy**|The strategy the agent learns to maximize reward|

**Example use cases:** Game-playing AI (like AlphaGo, chess engines), robotics, self-driving car decision-making, resource management systems.

---

## 5. Comparison Table — All Types at a Glance

|Type|Data Used|Goal|Example|
|---|---|---|---|
|**Supervised**|Labeled data|Predict output from input|Spam detection, price prediction|
|**Unsupervised**|Unlabeled data|Find hidden patterns/groups|Customer segmentation|
|**Semi-Supervised**|Small labeled + large unlabeled|Learn efficiently with limited labels|Medical image analysis|
|**Reinforcement**|Rewards from environment interaction|Learn best actions/strategy|Game-playing AI, robotics|

---

## 6. The General Machine Learning Workflow

Regardless of the type of ML being used, most projects follow this general pipeline:

```
1. Collect Data        → Gather relevant raw data
2. Prepare Data        → Clean, organize, handle missing values
3. Choose a Model       → Select an algorithm suited to the problem
4. Train the Model      → Feed data in, let the model adjust itself
5. Evaluate the Model    → Test performance on unseen data
6. Tune/Improve         → Adjust settings (hyperparameters) to improve accuracy
7. Deploy               → Put the model into real-world use
8. Monitor & Retrain     → Track performance over time, retrain as needed with new data
```

This cyclical process is important — ML isn't a "train once and forget" system. Models are often retrained as new data comes in, to keep them accurate and relevant (this is why your Netflix recommendations keep improving/changing over time).

---

## 7. How Machine Learning Relates to AI and Deep Learning

It's easy to confuse these three terms, so here's the clear relationship:

![AI ML DL](/images/notes/ai_ml_dl.png)


- **AI** is the broadest field — any technique that makes machines act intelligently (this includes rule-based systems too, not just learning-based ones).
- **ML** is a _subset_ of AI — specifically, systems that learn from data rather than being explicitly programmed.
- **DL** is a _subset_ of ML — it uses layered neural networks (like the perceptron-based networks from your earlier notes) to learn especially complex patterns, usually requiring large amounts of data and computing power.

> **In short:** All Deep Learning is Machine Learning, and all Machine Learning is Artificial Intelligence — but not the other way around.

---

## 8. Key Takeaway

> Machine Learning represents a fundamental shift from **telling** computers exactly what to do, to **teaching** them by example. Its real power lies in being able to detect patterns too complex for humans to hand-code, improving automatically as more data flows in, and generalizing that learning to make accurate predictions on situations it has never explicitly seen before. Understanding its four main types — **Supervised, Unsupervised, Semi-Supervised, and Reinforcement Learning** — gives you the map to understand almost every real-world AI system you interact with today, from spam filters to self-driving cars to the very chatbot you're reading this from.

## Essential Reads:

**Videos:**
- [Machine Learning Explained: A Guide to ML, AI, & Deep Learning](https://www.youtube.com/watch?v=znF2U_3Z210)
- [Stanford CS229: Machine Learning Lecture 1 - Andrew Ng (Autumn 2018)](https://www.youtube.com/watch?v=jGwO_UgTS7I)

**Blogs:**
- https://machinelearningmastery.com/machine-learning-algorithms-mini-course/

**Books:**
- Introduction to Machine Learning, Ethem Alpaydin, Chapter 1.
- Introduction to Machine Learning, Ethem Alpaydin, Chapter 10.
- Machine Learning, Tom M Mitchell, Chapter 1
- Machine Learning, Tom M Mitchell, Chapter 4 (Sec 1-4)