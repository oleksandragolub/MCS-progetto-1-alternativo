#!/usr/bin/env python3
"""
Genera grafici dai risultati dei test per la relazione 
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict

def parse_results_file_real(filepath):
    """
    Parsing reale dei file di risultati nella cartella results/
    """
    results = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cerca i blocchi per ogni metodo
        methods = {
            'jacobi': 'Jacobi',
            'gauss_seidel': 'Gauss-Seidel', 
            'gradient_method': 'Gradient',
            'conjugate_gradient': 'Conjugate Gradient'
        }
        
        for method_key, method_name in methods.items():
            # Pattern per trovare le informazioni di ogni metodo
            pattern = rf"Metodo: {method_name}\n.*?Convergenza: (.*?)\n.*?Iterazioni: (\d+)\n.*?Tempo: ([\d.]+) s\n.*?Errore residuo: ([\d.e+-]+)\n.*?Errore relativo: ([\d.e+-]+)"
            
            match = re.search(pattern, content, re.DOTALL)
            if match:
                converged = match.group(1).strip() == "Sì"
                iterations = int(match.group(2))
                time = float(match.group(3))
                relative_error = float(match.group(5))
                
                results[method_key] = {
                    'iterations': iterations,
                    'time': time,
                    'relative_error': relative_error,
                    'converged': converged
                }
    
    except Exception as e:
        print(f"Errore nel parsing di {filepath}: {e}")
        return {}
    
    return results

def load_all_results_auto(results_dir="results"):
    """
    Carica automaticamente tutti i risultati dalla cartella results/
    """
    results = {}
    
    matrices = ["spa1", "spa2", "vem1", "vem2"]
    tolerances = ["1e-04", "1e-06", "1e-08", "1e-10"]
    
    print("Caricamento automatico dei risultati...")
    
    for matrix in matrices:
        results[matrix] = {}
        for tol in tolerances:
            filename = f"{matrix}_tol_{tol}.txt"
            filepath = os.path.join(results_dir, filename)
            
            if os.path.exists(filepath):
                print(f"  Caricando {filename}...")
                matrix_results = parse_results_file_real(filepath)
                if matrix_results:
                    results[matrix][tol] = matrix_results
                else:
                    print(f"    ⚠️ Nessun dato trovato in {filename}")
            else:
                print(f"    ❌ File {filename} non trovato")
    
    # Verifica completezza dati
    total_files = 0
    loaded_files = 0
    for matrix in matrices:
        for tol in tolerances:
            total_files += 1
            if matrix in results and tol in results[matrix]:
                loaded_files += 1
    
    print(f"\nRiepilogo caricamento: {loaded_files}/{total_files} file caricati")
    return results

def plot_performance_all_tolerances(results, output_dir="plots"):
    """
    Crea grafici a barre per tutte le tolleranze
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tolerances = ['1e-04', '1e-06', '1e-08', '1e-10']
    tol_values = [1e-4, 1e-6, 1e-8, 1e-10]
    
    matrices = ['spa1', 'spa2', 'vem1', 'vem2']
    matrix_labels = ['spa1\n(1000×1000)', 'spa2\n(3000×3000)', 'vem1\n(1681×1681)', 'vem2\n(2601×2601)']
    methods = ['jacobi', 'gauss_seidel', 'gradient_method', 'conjugate_gradient']
    method_names = ['Jacobi', 'Gauss-Seidel', 'Gradiente', 'Gradiente Coniugato']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for tol_idx, (tol_str, tol_val) in enumerate(zip(tolerances, tol_values)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raccolta dati per questa tolleranza
        iterations_data = []
        times_data = []
        
        for matrix in matrices:
            matrix_iters = []
            matrix_times = []
            
            for method in methods:
                if (matrix in results and tol_str in results[matrix] and 
                    method in results[matrix][tol_str]):
                    data = results[matrix][tol_str][method]
                    matrix_iters.append(data['iterations'])
                    matrix_times.append(data['time'])
                else:
                    matrix_iters.append(None)
                    matrix_times.append(None)
            
            iterations_data.append(matrix_iters)
            times_data.append(matrix_times)
        
        # Grafico iterazioni
        x = np.arange(len(matrices))
        width = 0.2
        
        for i, (method_name, color) in enumerate(zip(method_names, colors)):
            iters = [data[i] if data[i] is not None else 0 for data in iterations_data]
            iters = [max(x, 1) for x in iters]
            ax1.bar(x + i*width - 1.5*width, iters, width, label=method_name, color=color, alpha=0.8)
        
        ax1.set_xlabel('Matrice')
        ax1.set_ylabel('Iterazioni (scala log)')
        ax1.set_title(f'Iterazioni per Convergenza (tol={tol_val:.0e})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(matrix_labels)
        ax1.legend()
        ax1.set_yscale('log')
        
        # Grafico tempi
        for i, (method_name, color) in enumerate(zip(method_names, colors)):
            times = [data[i] if data[i] is not None else 0 for data in times_data]
            times = [max(x, 0.001) for x in times]
            ax2.bar(x + i*width - 1.5*width, times, width, label=method_name, color=color, alpha=0.8)
        
        ax2.set_xlabel('Matrice')
        ax2.set_ylabel('Tempo (s, scala log)')
        ax2.set_title(f'Tempo di Calcolo (tol={tol_val:.0e})')
        ax2.set_xticks(x)
        ax2.set_xticklabels(matrix_labels)
        ax2.legend()
        ax2.set_yscale('log')
        
        plt.tight_layout()
        filename = f'performance_tol_{tol_str}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grafico salvato: {output_dir}/{filename}")

def plot_all_matrices_comparison(results, output_dir="plots"):
    """
    Crea un grafico 2x2 con iterazioni vs tolleranza per tutte le matrici
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    matrices = ['spa1', 'spa2', 'vem1', 'vem2']
    matrix_info = {
        'spa1': 'spa1: 1000×1000, 81.77% sparse',
        'spa2': 'spa2: 3000×3000, 81.87% sparse', 
        'vem1': 'vem1: 1681×1681, 99.53% sparse',
        'vem2': 'vem2: 2601×2601, 99.69% sparse'
    }
    
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    methods = ['jacobi', 'gauss_seidel', 'gradient_method', 'conjugate_gradient']
    method_names = ['Jacobi', 'Gauss-Seidel', 'Gradiente', 'Gradiente Coniugato']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, matrix in enumerate(matrices):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        matrix_data = results.get(matrix, {})
        
        for method, name, color in zip(methods, method_names, colors):
            valid_tols = []
            valid_iters = []
            
            for tol_val, tol_str in zip(tolerances, ['1e-04', '1e-06', '1e-08', '1e-10']):
                if tol_str in matrix_data and method in matrix_data[tol_str]:
                    valid_tols.append(tol_val)
                    valid_iters.append(matrix_data[tol_str][method]['iterations'])
            
            if valid_iters:
                ax.loglog(valid_tols, valid_iters, 'o-', label=name, 
                         linewidth=2, markersize=6, color=color)
        
        ax.set_xlabel('Tolleranza')
        ax.set_ylabel('Numero di Iterazioni')
        ax.set_title(matrix_info[matrix])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_matrices_iterations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico salvato: {output_dir}/all_matrices_iterations.png")

def plot_performance_summary(results, output_dir="plots"):
    """
    Grafico di riepilogo: tempi per tolleranza 1e-06
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    matrices = ['spa1', 'spa2', 'vem1', 'vem2']
    matrix_labels = ['spa1\n(1000×1000)', 'spa2\n(3000×3000)', 'vem1\n(1681×1681)', 'vem2\n(2601×2601)']
    methods = ['jacobi', 'gauss_seidel', 'gradient_method', 'conjugate_gradient']
    method_names = ['Jacobi', 'Gauss-Seidel', 'Gradiente', 'Gradiente Coniugato']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Raccolta dati per tolleranza 1e-06
    iterations_data = []
    times_data = []
    
    for matrix in matrices:
        matrix_iters = []
        matrix_times = []
        
        for method in methods:
            if (matrix in results and '1e-06' in results[matrix] and 
                method in results[matrix]['1e-06']):
                data = results[matrix]['1e-06'][method]
                matrix_iters.append(data['iterations'])
                matrix_times.append(data['time'])
            else:
                matrix_iters.append(None)
                matrix_times.append(None)
        
        iterations_data.append(matrix_iters)
        times_data.append(matrix_times)
    
    # Grafico iterazioni
    x = np.arange(len(matrices))
    width = 0.2
    
    for i, (method_name, color) in enumerate(zip(method_names, colors)):
        iters = [data[i] if data[i] is not None else 0 for data in iterations_data]
        # Sostituisci 0 con un valore piccolo per la scala log
        iters = [max(x, 1) for x in iters]
        ax1.bar(x + i*width - 1.5*width, iters, width, label=method_name, color=color, alpha=0.8)
    
    ax1.set_xlabel('Matrice')
    ax1.set_ylabel('Iterazioni (scala log)')
    ax1.set_title('Iterazioni per Convergenza (tol=1e-06)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(matrix_labels)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Grafico tempi
    for i, (method_name, color) in enumerate(zip(method_names, colors)):
        times = [data[i] if data[i] is not None else 0 for data in times_data]
        # Sostituisci 0 con un valore piccolo per la scala log
        times = [max(x, 0.001) for x in times]
        ax2.bar(x + i*width - 1.5*width, times, width, label=method_name, color=color, alpha=0.8)
    
    ax2.set_xlabel('Matrice')
    ax2.set_ylabel('Tempo (s, scala log)')
    ax2.set_title('Tempo di Calcolo (tol=1e-06)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(matrix_labels)
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico salvato: {output_dir}/performance_summary.png")

def main():
    """
    Genera tutti i grafici per la relazione usando i dati reali
    """
    print("=" * 60)
    print("GENERAZIONE GRAFICI PER LA RELAZIONE")
    print("=" * 60)
    
    # Carica automaticamente i risultati
    results = load_all_results_auto("results")
    
    if not results:
        print(" Nessun risultato trovato! Verifica che la cartella 'results/' contenga i file.")
        return
    
    print("\n" + "=" * 60)
    print("GENERAZIONE GRAFICI")
    print("=" * 60)
    
    # Genera i grafici
    plot_all_matrices_comparison(results)
    plot_performance_all_tolerances(results)  # Per tutte le tolleranze
    
    print("\n" + "=" * 60)
    print("GRAFICI COMPLETATI")
    print("=" * 60)
    print("Grafici generati nella cartella 'plots/':")
    print("- all_matrices_iterations.png: Confronto iterazioni per tutte le matrici")
    print("- performance_tol_1e-04.png: Prestazioni per tolleranza 1e-04")
    print("- performance_tol_1e-06.png: Prestazioni per tolleranza 1e-06")
    print("- performance_tol_1e-08.png: Prestazioni per tolleranza 1e-08")
    print("- performance_tol_1e-10.png: Prestazioni per tolleranza 1e-10")
    print("\nI grafici sono pronti!")

if __name__ == "__main__":
    main()