<style>

body { counter-reset: h1counter h2counter h3counter h4counter h5counter h6counter; }

h1 { counter-reset: h2counter; }
h2 { counter-reset: h3counter; }
h3 { counter-reset: h4counter; }
h4 { counter-reset: h5counter; }
h5 { counter-reset: h6counter; }
h6 {}

h2:before {
    counter-increment: h2counter;
    content: counter(h2counter) ".\0000a0\0000a0";
}

h3:before {
    counter-increment: h3counter;
    content: counter(h2counter) "." counter(h3counter) ".\0000a0\0000a0";
}

h4:before {
    counter-increment: h4counter;
    content: counter(h2counter) "." counter(h3counter) "." counter(h4counter) ".\0000a0\0000a0";
}

h5:before {
    counter-increment: h5counter;
    content: counter(h2counter) "." counter(h3counter) "." counter(h4counter) "." counter(h5counter) ".\0000a0\0000a0";
}

h6:before {
    counter-increment: h6counter;
    content: counter(h2counter) "." counter(h3counter) "." counter(h4counter) "." counter(h5counter) "." counter(h6counter) ".\0000a0\0000a0";
}

</style>

# This repository follows Reinforcement learning book (by Richard S. Sutto and Adrew G. Barto) and CS234 by Stanford (Emma Brunskill)

## A game Tic-Tac-Toe 

This game is played by two players by filling cells of a square matrix. It ends if one player's marks (for example X) are filled horizontally, vertically, or diagonally. This game is developed based on a temporal-difference learning method.

$$V(S_t) \gets V(S_t) + \alpha\left[{V(S_{t+1})-V(S_t)}\right]$$

where $\alpha$ is a learning rate or step size. After finishing each game (episode), if the game is not draw, then we apply 1 reward to the winner's state vales $V(S)$.

![image](./tic_tac_toe/tic-tac-toe.png)

### Running
Run the following command after installing numpy
```
python tic_tac_toe.py --eps-greedy=0.3 --num-episodes=300
```

Where ```eps-greedy```, and ```num-episodes``` are a greedy rate (default 0.3) and the number (default 300) of training steps of the two players to train them as an adversarial game. After running this code, the two players play against each other over given episodes. Then you see a option to choose the player you want to be and play with another player. Have fun!

## Multi-armed bandits
### Action value methods

One way to estimate action value is simple average the rewards over time step.

$$Q_t(a) = \frac{\sum_{i=1}^{t-1}R_i 1_{A_i=a}}{\sum_{i=1}^{t-1}1_{A_i=a}}$$

Greed selection

$$A_t \doteq argmax_a Q_t(a)$$

Run the following script to get Fig. 2.1.

```
python testbed-k-armed-fig2.1.py
```

For Fig. 2.2.

```
python testbed-k-armed-fig2.2.py
```