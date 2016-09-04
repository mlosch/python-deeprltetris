# python-deeprltetris

Implements the simplest deep reinforcement learner for Tetris based on the
Google DeepMind paper [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/pdf/1312.5602v1.pdf)
including

### Requirements
 - pyglet
 - theano
 - lasagne

### How To Run
 - Training an agent:
    - Simply type `python start_rl.py`
    - Hyperparameters can be adjusted in the same file line 71. Please consult the Google DeepMind paper for further information on these parameters:
       - `learningrate`
       - `discountfactor`
       - `epsilon`
       - `minepsilong`
       - `depsilong`
    - Withing `start_rl.py`, the training method can be chosen. See `lib/Learning.py` for how to instantiate:
       - `Deep Q Learner` (default)
       - `Q-Learner`
       - `SarsaLearner`
       - `SarsaLambdaLearner`
 - Continue training:
    - The script `start_rl.py` automatically looks for policies of name `policy-*.pickle` and continues the most recent by default.
 - Watch an agent play:
    - Change the variable `DRAW` within `start_rl.py` to `True`.
 - Adjust the Tetris environment within `start_rl.py` and the type of blocks in `lib/Shape.py`
    - Note that the classical Tetris environment (board size of 14x20 and classical block types)
    are *very* difficult to get to convergence as the possibility of scoring is extremely small.


### Acknowledgements
The Tetris framework was originally created by Charles Leifer.

Visit https://github.com/coleifer/tetris


The deep reinforcement learning code by Nathan Sprague inspired the implementation for this project.

Visit https://github.com/spragunr/deep_q_rl
