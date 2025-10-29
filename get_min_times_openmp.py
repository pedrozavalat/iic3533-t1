import pandas as pd

def plot_openmp_best_times(openmp_file):
    df = pd.read_csv(openmp_file)
    idx = df.groupby('Threads')['Time'].idxmin()
    best_times = df.loc[idx, ['Threads', 'Schedule', 'Chunk', 'Time']].sort_values('Threads')
    print(best_times.to_string(index=False))

plot_openmp_best_times("results.csv")
