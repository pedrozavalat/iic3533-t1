import pandas as pd
import matplotlib.pyplot as plt

def plot_speedup_efficiency(numba_csv='numba_results.csv', openmp_file='results.txt'):
    numba_df = pd.read_csv(numba_csv)
    numba_df = numba_df.sort_values('Threads')

    T1_numba = numba_df.loc[numba_df['Threads'] == 1, 'Time'].values[0]

    # Calcular Speedup y Eficiencia
    numba_df['Speedup'] = T1_numba / numba_df['Time']
    numba_df['Eficiencia'] = numba_df['Speedup'] / numba_df['Threads']

    try:
        openmp_df = pd.read_csv(openmp_file)
        best_openmp = openmp_df.groupby('Threads')['Time'].min().reset_index()
        best_openmp = best_openmp.sort_values('Threads')

        T1_openmp = T1_numba
        best_openmp['Speedup'] = T1_openmp / best_openmp['Time']
        best_openmp['Eficiencia'] = best_openmp['Speedup'] / best_openmp['Threads']
    except FileNotFoundError:
        print(f"Archivo {openmp_file} no encontrado. Mostrando solo resultados Numba.")
        best_openmp = None

    # Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(numba_df['Threads'], numba_df['Speedup'],
             'ro-', linewidth=2, markersize=8, label='Numba (Python)')
    plt.plot(numba_df['Threads'], numba_df['Threads'],
             'k--', label='Speedup ideal (lineal)')

    if best_openmp is not None:
        plt.plot(best_openmp['Threads'], best_openmp['Speedup'],
                 'bo-', linewidth=2, markersize=8, label='OpenMP (C++) - Mejor Speedup')

    plt.xlabel('Número de Procesadores (Threads)')
    plt.ylabel('Speedup')
    plt.title('Comparación: Speedup vs Número de Procesadores')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Anotar valores
    for _, row in numba_df.iterrows():
        plt.annotate(f'{row["Speedup"]:.2f}',
                     (row['Threads'], row['Speedup']),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    if best_openmp is not None:
        for _, row in best_openmp.iterrows():
            plt.annotate(f'{row["Speedup"]:.2f}',
                         (row['Threads'], row['Speedup']),
                         textcoords="offset points", xytext=(0, -15), ha='center')

    plt.tight_layout()
    plt.savefig('speedup_vs_procesadores.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Eficiencia

    plt.figure(figsize=(10, 6))
    plt.plot(numba_df['Threads'], numba_df['Eficiencia'],
             'ro-', linewidth=2, markersize=8, label='Numba (Python)')

    if best_openmp is not None:
        plt.plot(best_openmp['Threads'], best_openmp['Eficiencia'],
                 'bo-', linewidth=2, markersize=8, label='OpenMP (C++) - Mejor Eficiencia')

    plt.xlabel('Número de Procesadores (Threads)')
    plt.ylabel('Eficiencia')
    plt.title('Comparación: Eficiencia vs Número de Procesadores')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Anotar valores
    for _, row in numba_df.iterrows():
        plt.annotate(f'{row["Eficiencia"]:.2f}',
                     (row['Threads'], row['Eficiencia']),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    if best_openmp is not None:
        for _, row in best_openmp.iterrows():
            plt.annotate(f'{row["Eficiencia"]:.2f}',
                         (row['Threads'], row['Eficiencia']),
                         textcoords="offset points", xytext=(0, -15), ha='center')

    plt.tight_layout()
    plt.savefig('eficiencia_vs_procesadores.png', dpi=300, bbox_inches='tight')
    plt.show()

    return numba_df, best_openmp

if __name__ == "__main__":
    numba_df, openmp_df = plot_speedup_efficiency(
        numba_csv='numba_results.csv',
        openmp_file='results.txt'
    )
