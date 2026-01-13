# SuMo
Reinforcement Learning-based models for sudoku

**KL Divergence Explosion in the Auxiliary Phase :**

During the auxiliary phase, I initially computed the kl using the probabilities coming from the post-masked logits. Since invalid actions are masked by setting their logits to -inf before the softmax, their resulting probabilities become zero.This causes a numerical issue in the KL divergence : KL contains terms like log(p) and log(q), and log(0) → -∞, which makes the KL blow up to `inf`.An easy fix was to use the raw (unmasked) logits to compute the KL divergence.
