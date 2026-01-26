* **Bias :** The environment version [sudoku-v0](https://github.com/adeotti/Gymnasium-Sudoku) on which all these policies were trained and tested does not let them corrupt the board, meaning a policy cannot modify a correct cell it has previously filled. This is one of, if not the only reason these trained policies are able to complete all Sudoku boards. A less biased and closer-to-the-truth version of the environment would let the policies corrupt the board if they try to modify a previously filled correct cell with a new bad value. This should make learning much harder, and generalization in this version of the environment should be very impressive, since error recovery can be a sign of some sort of learning.

* **KL Divergence Explosion during the Auxiliary Phase :** I initially computed KL using the probabilities drawn from the post-masked logits and since invalid actions are masked by setting their logits to `-inf` or `-1e9`(for the position mask) before Softmax, their resulting probabilities becomes zero and this cause KL to occasionally be `inf`... KL’s formula : `p * log(p / q)`, and when using the masked logits to generate a distribution:

    `q = 0 : p * log(p / 0) -→ +∞`

    `p = 0 : 0 * log(0 / q) -→ undefined (log → -∞)`

    An easy fix is to compute KL between the old and new policy using the raw pre-masked logits instead.

* **Experiments :** The model`sumo_v0_4k` was trained on a single board and on only ~20 million transitions, but is able to generalize to boards it has never seen during training while always beating the baseline on the current version of the environment, which is a random policy’s total steps for n episodes. That generalization ability is coming from the architecture of both the environment and the neural network. On the neural network side, the generalization ability is coming from the position masking method, which masks all untouchable cells (both the originals and the ones guessed right by the policy). Resetting the environment with different boards is not necessary as long as the policy is able to mask untouchable cells as the board is getting filled with correct values.

