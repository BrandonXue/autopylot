# AutoPylot
This is our AI Capstone project, which involves deep reinforcement learning.
More specifically, we used "Deep Q Learning with Experience Replay" to make an agent
(a box representing a car) drive around an environment avoiding hazardous walls.
The algorithm we implemented is originally from a paper written by DeepMind Technologies.

Q-values describe the return of a state-action pair.
The Q-value function comes from Markov Decision Processes, and satisfies the Bellman Equation.
This allows the value of a state to be described iteratively.

The algorithm uses a convolutional neural network to take images or sequences of images
of the environment. By the universal approximation theorem, the network acts as a
function approximator for Q*(s, a), the optimal Q-value function.

Using simple reinforcement signals, the agent improves its approximation of an optimal
Q-value function, and takes better and better actions over time.

## Bonus

As a bonus, our capstone project contains a small sub-project: Pixelworld.
Pixelworld is a small toy environment with tiny images sizes that are much quicker
for the Deep Q Network to perform its convolutions, feedforward, and backpropagation.

## Dependencies
- python 3.8
- pygame 2.0 (SDL 2.0.12)
- opencv python (cv2) 4.4.0.46
- numpy 1.18.4
- matplotlib 3.2.1
- tensorflow 2.3.1
- tensorflow.keras

## Contributors
- Brandon Xue
- Jacob Rapmund
