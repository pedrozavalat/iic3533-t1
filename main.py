import numba
import math
import time
from numba import jit, prange
import pandas as pd
import matplotlib.pyplot as plt

@jit(nopython=True)
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

@jit(nopython=True, parallel=True)
def count_primes_parallel(n):
    count = 0
    for i in prange(2, n):
        if is_prime(i):
            count += 1
    return count

def run_numba_experiments():
    limit = 400000000
    thread_counts = [1, 2, 4, 6, 8]
    
    results = []
    
    print("Iniciando experimentos con Numba")
    print(f"Límite: {limit:,}")
    
    for num_threads in thread_counts:
        numba.set_num_threads(num_threads)
        
        if num_threads == 1:
            print(f"Con {num_threads} thread:")
        else:
            print(f"Con {num_threads} threads:")

        if num_threads == thread_counts[0]:
            count_primes_parallel(1000)
        
        start_time = time.time()
        prime_count = count_primes_parallel(limit)
        end_time = time.time()
        execution_time = end_time - start_time
        
        results.append({
            'Threads': num_threads,
            'Time': execution_time,
            'Primes': prime_count
        })
        print(f"{execution_time:.2f} segundos, {prime_count:,} número primos encontrados")
    return results

def save_results(results, filename='numba_results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResultados guardados en {filename}")
    return df

def plot_times_comparison(numba_results, openmp_file='results.txt'):    
    numba_df = pd.DataFrame(numba_results)
    try:
        openmp_df = pd.read_csv(openmp_file)
        best_openmp = openmp_df.groupby('Threads')['Time'].min().reset_index()
        best_openmp = best_openmp.sort_values('Threads')
    except FileNotFoundError:
        print(f"Archivo {openmp_file} no encontrado. Mostrando solo resultados Numba.")
        best_openmp = None
    
    plt.figure(figsize=(10, 6))
    plt.plot(numba_df['Threads'], numba_df['Time'], 
             'ro-', linewidth=2, markersize=8, label='Numba (Python)')
    
    if best_openmp is not None:
        plt.plot(best_openmp['Threads'], best_openmp['Time'], 
                 'bo-', linewidth=2, markersize=8, label='OpenMP (C++) - Mejor tiempo')
    
    plt.xlabel('Número de Procesadores (Threads)')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.title('Comparación: Tiempo vs Número de Procesadores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    for i, row in numba_df.iterrows():
        plt.annotate(f'{row["Time"]:.1f}s', 
                    (row['Threads'], row['Time']), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    if best_openmp is not None:
        for i, row in best_openmp.iterrows():
            plt.annotate(f'{row["Time"]:.1f}s', 
                        (row['Threads'], row['Time']), 
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center')
    
    plt.tight_layout()
    plt.savefig('tiempo_vs_procesadores.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return numba_df, best_openmp

if __name__ == "__main__":
    numba_results = run_numba_experiments()
    numba_df = save_results(numba_results)
    numba_df, openmp_df = plot_times_comparison(numba_results)
    print(f"\nTotal de primos encontrados: {numba_results[0]['Primes']:,}")
    print(f"Mejor tiempo Numba: {numba_df['Time'].min():.2f}s")
    if openmp_df is not None:
        print(f"Mejor tiempo OpenMP: {openmp_df['Time'].min():.2f}s")
    