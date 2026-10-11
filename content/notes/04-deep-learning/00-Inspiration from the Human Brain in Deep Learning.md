---
title: "Inspiration from the Human Brain"
description: "How biological neurons inspired artificial neural networks, and where the analogy holds."
date: 2026-10-11
weight: 1
status: "growing"
tags: [deep-learning, neural-networks]
---

## 1. Why the Brain Matters to Deep Learning

Deep Learning did not appear out of nowhere — it was inspired by how the **human brain processes information** using billions of interconnected cells called **neurons**. Researchers wanted to build machines that could "think" or "learn" the way humans do, so they modeled artificial systems loosely on biological neural structure.

> **Core idea:** If biological neurons can learn patterns, recognize faces, and understand language by passing signals between each other — maybe artificial neurons, connected in a similar way, can learn to do the same.

This gave birth to **Artificial Neural Networks (ANNs)** — the foundation of Deep Learning.

---

## 2. How a Biological Neuron Works



![Brain neuron](/images/notes/brain_neuron.png)


A neuron in the brain is a specialized cell that transmits information using electrical and chemical signals. It has four main parts:

|Part|Function|
|---|---|
|**Dendrites**|Receive signals (inputs) from other neurons|
|**Cell Body (Soma)**|Processes/combines the incoming signals|
|**Axon**|Carries the processed signal away from the cell|
|**Synapse**|The junction where signal passes to the next neuron|

### Step-by-step working:

1. **Dendrites** receive electrical impulses from neighboring neurons.
2. The **soma (cell body)** collects and sums up all incoming signals.
3. If the combined signal crosses a certain **threshold**, the neuron "fires" — it generates an electrical impulse (this is called an **action potential**).
4. This impulse travels down the **axon**.
5. At the **synapse**, the signal is passed to the next neuron (often chemically, via neurotransmitters), possibly strengthening or weakening depending on how often that connection is used.
6. Repeated use of a connection makes it **stronger** — this is the basis of **learning and memory** in the brain (known as _synaptic plasticity_ or, more famously, **"neurons that fire together, wire together"**).

---

## 3. Excitatory vs Inhibitory Signals — The Brain's "Volume Control"

Not every signal reaching a neuron pushes it toward firing. The brain has **two types of signals** arriving at the synapse:

- **Excitatory signals** → _amplify_ the incoming message, pushing the neuron **closer to its firing threshold** (more likely to fire).
- **Inhibitory signals** → _dampen_ the incoming message, pushing the neuron **away from its firing threshold** (less likely to fire, or suppresses it entirely).

A single neuron doesn't just "add up" everything blindly — it receives thousands of these excitatory and inhibitory inputs from other neurons simultaneously, and only fires if the **net balance** crosses the threshold. This balancing act is what allows the brain to:

- **Filter out noise** (irrelevant or weak signals get suppressed)
- **Sharpen focus** (important signals get amplified while competing ones are inhibited — e.g., how you focus on one voice in a noisy room)
- **Prevent runaway/overactive firing** (inhibition keeps neural circuits stable, stopping signals from spiraling out of control)

This is a key part of how the brain processes information intelligently, not just electrically.

### Correlation with Deep Learning

This excitatory/inhibitory mechanism maps directly onto **signed weights** in an artificial neuron:

|Brain Mechanism|Deep Learning Equivalent|
|---|---|
|Excitatory signal (amplifies)|**Positive weight** — increases the weighted sum, pushing the neuron toward activation|
|Inhibitory signal (dampens)|**Negative weight** — decreases the weighted sum, pushing the neuron away from activation|
|Net balance of excitation/inhibition decides firing|Net weighted sum (Σ wᵢxᵢ + bias) decides the activation function's output|
|Inhibition prevents overactive/noisy firing|Techniques like **dropout, batch normalization, and regularization** dampen certain activations to prevent overfitting or unstable (exploding) signals|
|Sharpening focus by suppressing weak signals|**Attention mechanisms** — amplify important input features while suppressing irrelevant ones|

> **Easy way to remember:** In the brain, some connections say "yes, fire more!" (excitatory) and others say "no, calm down" (inhibitory). In deep learning, this is imitated by simply allowing weights to be **positive (push up)** or **negative (pull down)** — the network is not just "adding everything," it's constantly balancing amplification and suppression, just like real neurons.

---

## 4. The Artificial Neuron (Perceptron) — The Machine's Version

Scientists translated this biological process into a mathematical model. This is called an **artificial neuron** or **perceptron**.

![Perceptron](/images/notes/perceptron.png)

|Biological Neuron|Artificial Neuron (Equivalent)|
|---|---|
|Dendrites (receive signals)|**Inputs** (x₁, x₂, x₃...)|
|Strength of synapse|**Weights** (w₁, w₂, w₃...)|
|Cell body sums signals|**Summation function** (Σ wᵢxᵢ)|
|Firing threshold|**Activation function**|
|Axon (sends output)|**Output** (y)|
|Learning by strengthening synapses|**Weight adjustment during training (backpropagation)**|

### How it works mathematically:

```
Inputs:        x1, x2, x3 ...
Weights:       w1, w2, w3 ...

Step 1: Weighted Sum
   z = (x1*w1) + (x2*w2) + (x3*w3) + bias

Step 2: Activation Function
   output = f(z)   → decides whether neuron "fires" (like brain threshold)
```

- **Bias** is like a natural tendency of a neuron to fire — even with no strong input.
- **Activation function** (like Sigmoid, ReLU, Tanh) mimics the "firing threshold" of a biological neuron — it decides whether/how strongly the signal passes forward.

---

## 5. Side-by-Side Correlation (Quick Recall Table)

|Concept|Brain|Deep Learning|
|---|---|---|
|Basic Unit|Neuron|Artificial Neuron (Node)|
|Connection|Synapse|Weight|
|Signal strength control|Strengthening/weakening synapse|Adjusting weight values|
|Decision to activate|Threshold / action potential|Activation function|
|Network of units|Billions of neurons connected in layers (brain regions)|Neurons arranged in **layers** (Input → Hidden → Output)|
|Learning|Practice strengthens synaptic connections|Training adjusts weights via **backpropagation**|
|Memory|Stored in strengthened neural pathways|Stored in learned weight values|
|Massive parallel processing|Brain processes many signals simultaneously|Neural network processes data in parallel (matrix operations)|
|Signal amplification/dampening|Excitatory (amplify) vs Inhibitory (dampen) signals at synapse|Positive vs negative weight values|

---

## 6. From One Neuron to a Network

Just like the brain doesn't rely on a single neuron but on **networks of neurons organized in layers/regions** (e.g., visual cortex for sight, auditory cortex for sound), Deep Learning uses **layers of artificial neurons**:

![Neural network](/images/notes/neural_network.png)

- **Input Layer** → receives raw data (like sensory organs receiving stimuli)
- **Hidden Layers** → process and extract patterns (like brain regions processing information step-by-step, e.g., edges → shapes → objects in vision)
- **Output Layer** → gives the final decision/result (like the brain producing a response or action)

This layered structure is why it's called **"Deep" Learning** — "deep" refers to having many hidden layers, just as the brain has many interconnected processing stages before a final perception or decision is formed.

---

## 7. Learning Process Correlation

|Brain Learning|Deep Learning Equivalent|
|---|---|
|You touch fire → pain signal → brain learns "fire = danger" → next time you avoid it|Network makes a wrong prediction → **error/loss is calculated** → weights are updated → next time prediction improves|
|Repetition strengthens the correct pathway|Multiple training **epochs** strengthen the correct weight patterns|
|Forgetting/weakening unused connections|Regularization techniques (like dropout) prevent overfitting, similar to pruning unused synapses|

---

## 8. Key Takeaway (Easy Summary)

> Think of the brain as the **original inspiration**, and deep learning as a **simplified mathematical imitation** of it.

- Brain neuron **receives, processes, and fires** signals → Artificial neuron **receives inputs, computes weighted sum, and applies activation.**
- Brain **learns by strengthening synapses** → Network **learns by adjusting weights.**
- Brain uses **layered regions** for complex processing → Deep learning uses **layered networks (deep architecture)** for the same reason: to break down complex problems into simpler stages.

**In short:** Deep Learning tries to mimic — in a simplified, mathematical way — how billions of neurons in our brain connect, communicate, and learn from experience.

---

## Fun Fact

- Human Brain contains **~86 billion neurons** (the widely cited figure, from neuroscientist Suzana Herculano-Houzel's research)
- Each neuron connects to thousands of others via synapses — total synaptic connections are estimated at **100 trillion to 1 quadrillion**