import numpy as np
import seaborn as sns
import pandas as pd

def plot_and_save(
        k: int = 10,
        time_steps: int = 1000,
        num_runs: int = 2000) -> None:
    
    q_actual = np.zeros((num_runs, k))

    for i in range(num_runs):
        action_values = np.random.normal(0, 1, k)
        # Generating samples from Normal(Gaussian) distributions
        # with mean zero and unit variance

        for j in range(k):
            # Implicit time steps
            R_t = np.random.normal(action_values[j], 1, time_steps)
            q_actual[i, j] = R_t.mean()

    q_actual_plot = np.zeros((k * num_runs, 2))
    q_actual_plot[:, 0] = np.reshape(q_actual, newshape=(k * num_runs, ))

    for i in range(k):
        st = i * num_runs
        en = (i + 1) * num_runs
        
        q_actual_plot[st:en, 1] = i + 1

    df = pd.DataFrame(q_actual_plot, columns=['Rewards distribution', 'Action'])
    df['Action'] = df['Action'].astype(int)
    plot = sns.violinplot(data=df, x='Action', y='Rewards distribution')
    name = f'Testbed {k}-armed with {num_runs} samples.png'
    plot.get_figure().savefig(name)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--time-steps', type=int, default=1000)
    parser.add_argument('--num-runs', type=int, default=2000)

    args = parser.parse_args()

    plot_and_save(k=args.k, time_steps=args.time_steps, num_runs=args.num_runs)