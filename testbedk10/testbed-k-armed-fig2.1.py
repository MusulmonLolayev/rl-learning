import numpy as np
import seaborn as sns
import pandas as pd

def plot_and_save(
        k: int = 10,
        num_samples: int = 2000) -> None:
    

    action_values = np.random.normal(0, 1, k)

    q_actual = np.zeros((k * num_samples, 2))

    # Generating samples from Normal(Gaussian) distributions
    # with mean zero and unit variance
    for i in range(k):

      st = i * num_samples
      en = (i + 1) * num_samples
      q_actual[st:en, 0] = np.random.normal(action_values[i], 1, num_samples)
      q_actual[st:en, 1] = i + 1

    df = pd.DataFrame(q_actual, columns=['Rewards distribution', 'Action'])
    df['Action'] = df['Action'].astype(int)
    plot = sns.violinplot(data=df, x='Action', y='Rewards distribution')
    name = f'Testbed {k}-armed with {num_samples} samples.png'
    plot.get_figure().savefig(name)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--num-samples', type=int, default=2000)

    args = parser.parse_args()

    plot_and_save(k=args.k, num_samples=args.num_samples)