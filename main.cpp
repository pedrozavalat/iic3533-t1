#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <omp.h> 

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

int main() {
    int limit = 400000000;  
    std::string schedules[] = {"static", "dynamic", "guided"};
    int chunk_sizes[] = {1000, 10000, 50000, 100000, 200000, 500000, 1000000};
    int thread_counts[] = {2, 4, 6, 8}; 

    std::ofstream results_file("results_2.txt");
    results_file << "Schedule,Chunk,Threads,Time" << std::endl;
    results_file.close();

    for (const auto& sched : schedules) {
        std::cout << "Testing schedule: " << sched << std::endl;
        for (int chunk : chunk_sizes) {
            std::cout << "  Chunk size: " << chunk << std::endl;
            for (int n_threads : thread_counts) {
                std::cout << "    Threads: " << n_threads << std::endl;
                int count = 0;
                auto start = std::chrono::high_resolution_clock::now();

                if (sched == "static") {
                    #pragma omp parallel for reduction(+:count) schedule(static, chunk) num_threads(n_threads)
                    for (int n = 2; n < limit; n++) {
                        if (is_prime(n)) count++;
                    }
                } else if (sched == "dynamic") {
                    #pragma omp parallel for reduction(+:count) schedule(dynamic, chunk) num_threads(n_threads)
                    for (int n = 2; n < limit; n++) {
                        if (is_prime(n)) count++;
                    }
                } else if (sched == "guided") {
                    #pragma omp parallel for reduction(+:count) schedule(guided, chunk) num_threads(n_threads)
                    for (int n = 2; n < limit; n++) {
                        if (is_prime(n)) count++;
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;

                std::cout << sched << "," << chunk << "," << n_threads << "," << duration.count() << " s, Primes: " << count << std::endl;

                std::ofstream results_file("results_2.txt", std::ios::app);
                results_file << sched << "," << chunk << "," << n_threads << "," << duration.count() << std::endl;
            }
        }
    }

    return 0;
}
