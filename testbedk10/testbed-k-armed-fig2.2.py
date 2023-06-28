import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_and_save(
        k: int = 10,
        time_steps: int = 1000,
        num_runs: int = 2000,
        eps_greedies: tuple = None) -> None:
    
    if eps_greedies is None:
        raise ValueError(f"eps_greedies must constain at least one element")
    
    # for 0-greedy policy
    if 0 not in eps_greedies:
        eps_greedies = (0, *eps_greedies)
    num_algs = len(eps_greedies)
    aver_rewards = np.zeros((time_steps, num_algs))
    optimal_action = np.zeros((time_steps, num_algs))
    
    for exp in range(num_runs):

        act_vals = np.zeros((num_algs, k))
        for i in range(num_algs):
            act_vals[i] = np.random.normal(0, 1)

        Q_reward = np.zeros((num_algs, k))
        Q_count = np.zeros((num_algs, k), dtype=np.int32)
        total_reward = np.zeros(num_algs)

        for time_step in range(time_steps):
            
            # Over greedies
            for i, eps in enumerate(eps_greedies):
                # Action values
                Q_t = np.zeros(k)
                cond = Q_count[i] != 0
                Q_t[cond] = Q_reward[i, cond] / Q_count[i, cond]

                if np.random.rand() < eps:
                    # Non greedy action
                    # select action randomly 
                    sel_action = np.random.randint(0, k, 1)[0]
                else:
                    # greedy selection
                    sel_action = np.argmax(Q_t)
                
                R_t = np.random.normal(Q_t[sel_action], 10)

                total_reward[i] += R_t
                Q_reward[i, sel_action] += R_t
                Q_count[i, sel_action] += 1

            aver_rewards[time_step] += total_reward / (time_step + 1)
    
    aver_rewards /= num_runs
    steps = list(range(time_steps))

    for i, eps in enumerate(eps_greedies):
        plt.plot(steps, aver_rewards[:, i], label=f'Greedy $\epsilon$={eps}')
    
    plt.legend(loc="lower right")
    plt.savefig(f'./testbedk10/Fig. 2.2 {eps_greedies}.png')

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--time-steps', type=int, default=1000)
    parser.add_argument('--num-runs', type=int, default=2000)
    parser.add_argument('--eps-greedies', type=float, nargs='+', default=(0, 0.01, 0.1))


    args = parser.parse_args()

    plot_and_save(k=args.k, 
                  time_steps=args.time_steps,
                  num_runs=args.num_runs,
                  eps_greedies=tuple(args.eps_greedies))