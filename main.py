#!/usr/bin/env python3
"""
Programma principale per il testing dei metodi iterativi

MODALITÀ VALIDAZIONE:
    python main.py                              # Test su tutte le matrici con tutte tolerranze
    python main.py --matrix spa1.mtx            # Test su matrice specifica con tutte tolerranze 
    python main.py --test                       # Test con matrice generata default 100×100 con tutte tolerranze 
    python main.py --test -n 500 -t 1e-6        # Test con matrice generata 500×500 e testa solo alla tolleranza indicata
    python main.py -t 1e-4 1e-6                 # Test su tutte le matrici con tolleranze specifiche


MODALITÀ GENERICA:   # Esegue i 4 metodi su A=matrice.mtx, usando i miei vettori b_spa1.txt e x_spa1.txt (invece di costruire b=Ax*) con tolleranza 1e-6
    python main.py -m spa1.mtx --b test_data/b_spa1.txt --x test_data/x_spa1.txt --single-tol 1e-6
    python main.py -m spa2.mtx --b test_data/b_spa2.txt --x test_data/x_spa2.txt --single-tol 1e-6
    python main.py -m vem1.mtx --b test_data/b_vem1.txt --x test_data/x_vem1.txt --single-tol 1e-6
    python main.py -m vem2.mtx --b test_data/b_vem2.txt --x test_data/x_vem2.txt --single-tol 1e-6


AIUTO:
    python main.py --help                       # Mostra aiuto completo
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Importa la nostra libreria
from src import (
    LinearIterativeSolver,
    load_mtx_matrix,
    generate_standard_test_system,
    print_matrix_info,
    save_results_to_file,
    create_comparison_table
)


def test_single_matrix(matrix_path: str, tolerances: list, output_dir: str = "results"):
    """
    Testa tutti i metodi su una singola matrice per diverse tolleranze.
    
    Args:
        matrix_path: Percorso alla matrice .mtx
        tolerances: Lista delle tolleranze da testare
        output_dir: Directory per salvare i risultati
    """
    print(f"\n Testing matrice: {os.path.basename(matrix_path)}")
    print("="*60)
    
    try:
        # Carica la matrice
        A, matrix_info = load_mtx_matrix(matrix_path)
        print_matrix_info(A, f"Matrice {matrix_info['filename']}")
        
        # Genera il sistema test standard: x = [1,1,...,1], b = A*x
        b, x_exact = generate_standard_test_system(A)
        
        print(f"\n Sistema generato:")
        print(f"   Soluzione esatta: x = [1, 1, ..., 1] (dimensione {len(x_exact)})")
        print(f"   Termine noto: b = A * x")
        
        # Inizializza il solver
        solver = LinearIterativeSolver(max_iter=20000, verbose=False)
        
        # Risultati per tutte le tolleranze
        all_results = {}
        
        for tol in tolerances:
            print(f"\n Testing con tolleranza: {tol:.0e}")
            print("-" * 40)
            
            # Esegui tutti i metodi
            results = solver.solve_all_methods(A, b, x_exact, tol)
            all_results[f"{tol:.0e}"] = results
            
            # Stampa risultati per questa tolleranza
            print(f"{'Metodo':<20} {'Conv.':<6} {'Iter.':<8} {'Tempo (s)':<12} {'Err. Rel.':<12}")
            print("-" * 65)
            
            for method_name, result in results.items():
                conv_str = "Sì" if result['converged'] else "No"
                err_str = f"{result['relative_error']:.4e}" if result['relative_error'] != float('inf') else "∞"
                
                print(f"{result['method']:<20} {conv_str:<6} {result['iterations']:<8} "
                      f"{result['time']:<12.6f} {err_str:<12}")
        
        # Salva risultati dettagliati
        matrix_name = os.path.splitext(matrix_info['filename'])[0]
        
        # Crea tabella di confronto
        comparison_table = create_comparison_table(all_results)
        print(f"\n{comparison_table}")
        
        # Salva tutto in file
        os.makedirs(output_dir, exist_ok=True)
        
        # File con tabella di confronto
        summary_file = os.path.join(output_dir, f"{matrix_name}_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RISULTATI COMPLETI - Matrice: {matrix_info['filename']}\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Informazioni matrice
            f.write("INFORMAZIONI MATRICE:\n")
            f.write(f"Dimensione: {matrix_info['shape'][0]}x{matrix_info['shape'][1]}\n")
            f.write(f"Elementi non zero: {matrix_info['nnz']}\n")
            f.write(f"Sparsità: {matrix_info['sparsity']:.2%}\n")
            f.write(f"Simmetrica: {matrix_info['is_symmetric']}\n\n")
            
            # Tabella risultati
            f.write(comparison_table)
        
        # File dettagliati per ogni tolleranza
        for tol_str, results in all_results.items():
            detail_file = os.path.join(output_dir, f"{matrix_name}_tol_{tol_str}.txt")
            save_results_to_file(results, detail_file, matrix_info['filename'], float(tol_str))
        
        print(f"\n Risultati salvati in {output_dir}/")
        return all_results
        
    except Exception as e:
        print(f" Errore nel testing della matrice {matrix_path}: {str(e)}")
        return None


def test_generated_matrix(n: int = 100, matrix_type: str = 'tridiagonal', 
                         tolerances: list = None, output_dir: str = "results"):
    """
    Testa i metodi su una matrice generata automaticamente.
    
    Args:
        n: Dimensione della matrice
        matrix_type: Tipo di matrice da generare
        tolerances: Lista delle tolleranze
        output_dir: Directory per i risultati
    """
    if tolerances is None:
        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    
    print(f"\n Generando matrice test {matrix_type} {n}x{n}")
    print("="*60)
    
    from src.matrix_utils import create_test_system
    
    try:
        # Genera sistema test
        A, b, x_exact = create_test_system(n, matrix_type)
        print_matrix_info(A, f"Matrice {matrix_type}")
        
        # Inizializza solver
        solver = LinearIterativeSolver(max_iter=20000, verbose=False)
        
        all_results = {}
        
        for tol in tolerances:
            print(f"\n Testing con tolleranza: {tol:.0e}")
            print("-" * 40)
            
            results = solver.solve_all_methods(A, b, x_exact, tol)
            all_results[f"{tol:.0e}"] = results
            
            # Stampa risultati
            print(f"{'Metodo':<20} {'Conv.':<6} {'Iter.':<8} {'Tempo (s)':<12} {'Err. Rel.':<12}")
            print("-" * 65)
            
            for method_name, result in results.items():
                conv_str = "Sì" if result['converged'] else "No"
                err_str = f"{result['relative_error']:.4e}" if result['relative_error'] != float('inf') else "∞"
                
                print(f"{result['method']:<20} {conv_str:<6} {result['iterations']:<8} "
                      f"{result['time']:<12.6f} {err_str:<12}")
        
        # Salva risultati
        comparison_table = create_comparison_table(all_results)
        print(f"\n{comparison_table}")
        
        return all_results
        
    except Exception as e:
        print(f" Errore nel test con matrice generata: {str(e)}")
        return None


def find_mtx_files(data_dir: str = "data") -> list:
    """
    Trova tutti i file .mtx nella directory data.
    
    Args:
        data_dir: Directory dove cercare i file .mtx
        
    Returns:
        Lista dei percorsi ai file .mtx trovati
    """
    mtx_files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.mtx'):
                mtx_files.append(os.path.join(data_dir, file))
    return sorted(mtx_files)

def load_vector_from_file(filepath: str) -> np.ndarray:
    """
    Carica un vettore da file .txt o .npy
    
    Args:
        filepath: Percorso al file
    
    Returns:
        Vettore come array NumPy
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} non trovato")
    
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.txt'):
        return np.loadtxt(filepath)
    else:
        # Prova a caricare come testo
        try:
            return np.loadtxt(filepath)
        except:
            raise ValueError(f"Formato file {filepath} non supportato. Usa .txt o .npy")


def test_generic_system(matrix_path: str, b_path: str, x_path: str, 
                       tolerance: float, output_dir: str = "results"):
    """
    Testa tutti i metodi su una terna (A, b, x) generica fornita dall'utente.
    
    Args:
        matrix_path: Percorso alla matrice A (.mtx)
        b_path: Percorso al vettore b
        x_path: Percorso alla soluzione esatta x
        tolerance: Tolleranza per i metodi iterativi
        output_dir: Directory per salvare i risultati
    """
    print(f"\n MODALITÀ GENERICA - Terna (A, b, x) fornita dall'utente")
    print("="*60)
    
    try:
        # Carica la matrice A
        print(f"Caricando matrice: {os.path.basename(matrix_path)}")
        A, matrix_info = load_mtx_matrix(matrix_path)
        print_matrix_info(A, f"Matrice {matrix_info['filename']}")
        
        # Carica i vettori b e x
        print(f"Caricando termine noto: {os.path.basename(b_path)}")
        b = load_vector_from_file(b_path)
        
        print(f"Caricando soluzione esatta: {os.path.basename(x_path)}")
        x_exact = load_vector_from_file(x_path)
        
        # Verifiche di compatibilità
        if A.shape[0] != len(b):
            raise ValueError(f"Dimensioni incompatibili: A è {A.shape[0]}x{A.shape[1]}, b è {len(b)}")
        
        if A.shape[0] != len(x_exact):
            raise ValueError(f"Dimensioni incompatibili: A è {A.shape[0]}x{A.shape[1]}, x è {len(x_exact)}")
        
        print(f"\n Sistema configurato:")
        print(f" Matrice A: {A.shape[0]}x{A.shape[1]}")
        print(f" Vettore b: dimensione {len(b)}")
        print(f" Soluzione x: dimensione {len(x_exact)}")
        print(f" Tolleranza: {tolerance:.0e}")
        
        # Verifica che b = A*x (opzionale, per debug)
        residual_check = np.linalg.norm(A @ x_exact - b) / np.linalg.norm(b)
        print(f" Verifica b = A*x: residuo relativo = {residual_check:.2e}")
        if residual_check > 1e-10:
            print(f" ⚠️  Attenzione: il residuo è elevato, potrebbe esserci un errore nei dati")
        
        # Inizializza il solver
        solver = LinearIterativeSolver(max_iter=20000, verbose=False)
        
        # Esegui tutti i metodi
        print(f"\n Esecuzione metodi iterativi con tolleranza {tolerance:.0e}")
        print("-" * 60)
        
        results = solver.solve_all_methods(A, b, x_exact, tolerance)
        
        # Stampa risultati
        print(f"{'Metodo':<20} {'Conv.':<6} {'Iter.':<8} {'Tempo (s)':<12} {'Err. Rel.':<12}")
        print("-" * 65)
        
        for method_name, result in results.items():
            conv_str = "Sì" if result['converged'] else "No"
            err_str = f"{result['relative_error']:.4e}" if result['relative_error'] != float('inf') else "∞"
            print(f"{result['method']:<20} {conv_str:<6} {result['iterations']:<8} "
                  f"{result['time']:<12.6f} {err_str:<12}")
        
        # Salva risultati
        os.makedirs(output_dir, exist_ok=True)
        matrix_name = os.path.splitext(matrix_info['filename'])[0]
        
        result_file = os.path.join(output_dir, f"{matrix_name}_generic_tol_{tolerance:.0e}.txt")
        save_results_to_file(results, result_file, matrix_info['filename'], tolerance)
        
        print(f"\n Risultati salvati in {result_file}")
        
        return results
        
    except Exception as e:
        print(f" Errore nell'elaborazione del sistema generico: {str(e)}")
        return None

def main():
    """
    Funzione principale del programma.
    """
    parser = argparse.ArgumentParser(
        description='Test dei metodi iterativi per sistemi lineari - Progetto MCS 2024-2025',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:

MODALITÀ VALIDAZIONE:
    python main.py                              # Test su tutte le matrici con tutte tolerranze
    python main.py --matrix spa1.mtx            # Test su matrice specifica con tutte tolerranze 
    python main.py --test                       # Test con matrice generata default 100×100 con tutte tolerranze 
    python main.py --test -n 500 -t 1e-6        # Test con matrice generata 500×500 e testa solo alla tolleranza indicata
    python main.py -t 1e-4 1e-6                 # Test su tutte le matrici con tolleranze specifiche


MODALITÀ GENERICA:  # Esegue i 4 metodi su A=matrice.mtx, usando i miei vettori b_spa1.txt e x_spa1.txt (invece di costruire b=Ax*) con tolleranza 1e-6
    python main.py -m spa1.mtx --b test_data/b_spa1.txt --x test_data/x_spa1.txt --single-tol 1e-6
    python main.py -m spa2.mtx --b test_data/b_spa2.txt --x test_data/x_spa2.txt --single-tol 1e-6
    python main.py -m vem1.mtx --b test_data/b_vem1.txt --x test_data/x_vem1.txt --single-tol 1e-6
    python main.py -m vem2.mtx --b test_data/b_vem2.txt --x test_data/x_vem2.txt --single-tol 1e-6


AIUTO:
    python main.py --help                       # Mostra aiuto completo
        """
    )
    
    parser.add_argument('-m', '--matrix', type=str,
                       help='File .mtx specifico da testare')
    parser.add_argument('--test', action='store_true',
                       help='Esegue test su matrice generata automaticamente')
    parser.add_argument('-n', '--size', type=int, default=100,
                       help='Dimensione della matrice generata (default: 100)')
    parser.add_argument('-t', '--tolerances', nargs='+', type=float,
                       default=[1e-4, 1e-6, 1e-8, 1e-10],
                       help='Tolleranze da testare (default: 1e-4 1e-6 1e-8 1e-10)')
    parser.add_argument('-o', '--output', type=str, default='results',
                       help='Directory per i risultati (default: results)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory con i file .mtx (default: data)')
    parser.add_argument('--b', type=str, help='File con vettore termine noto b (.txt o .npy)')
    parser.add_argument('--x', type=str, help='File con soluzione esatta x (.txt o .npy)')
    parser.add_argument('--single-tol', type=float, help='Singola tolleranza per modalità generica')
    
    args = parser.parse_args()

    print(" PROGETTO MCS 2024-2025 - METODI ITERATIVI")
    print("=" * 60)
    print("Metodi implementati:")
    print("  1. Jacobi")
    print("  2. Gauss-Seidel")  
    print("  3. Gradiente")
    print("  4. Gradiente Coniugato")

    # Verifica modalità generica
    generic_mode = bool(args.b and args.x and args.single_tol and args.matrix)

    if generic_mode:
        print(f"\n MODALITÀ: Generica (A, b, x forniti dall'utente)")
        print(f"Tolleranza: {args.single_tol:.0e}")
        print(f"Limite iterazioni: 20000")
        
        # Costruisci percorsi
        matrix_path = args.matrix
        if not os.path.isabs(matrix_path):
            matrix_path = os.path.join(args.data_dir, matrix_path)
        
        # Esegui test generico
        test_generic_system(matrix_path, args.b, args.x, args.single_tol, args.output)
        return

    # MODALITÀ VALIDAZIONE 
    print(f"\n MODALITÀ: Validazione (x = [1,1,...,1], b = A*x)")
    print(f"Tolleranze: {[f'{t:.0e}' for t in args.tolerances]}")
    print(f"Limite iterazioni: 20000")
    
    # Test su matrice generata
    if args.test:
        print(f"\n MODALITÀ TEST - Matrice generata {args.size}x{args.size}")
        test_generated_matrix(args.size, 'tridiagonal', args.tolerances, args.output)
        return
    
    # Test su matrice specifica
    if args.matrix:
        matrix_path = args.matrix
        if not os.path.isabs(matrix_path):
            matrix_path = os.path.join(args.data_dir, matrix_path)
        
        if not os.path.exists(matrix_path):
            print(f" Errore: File {matrix_path} non trovato")
            sys.exit(1)
        
        test_single_matrix(matrix_path, args.tolerances, args.output)
        return
    
    # Test su tutte le matrici .mtx trovate
    mtx_files = find_mtx_files(args.data_dir)
    
    if not mtx_files:
        print(f" Nessun file .mtx trovato in {args.data_dir}/")
        print(" Suggerimenti:")
        print("   - Scarica i file spa1.mtx, spa2.mtx, vem1.mtx, vem2.mtx nella cartella data/")
        print("   - Oppure usa --test per testare con matrici generate")
        print("   - Oppure specifica un file con -m nome_file.mtx")
        sys.exit(1)
    
    print(f"\n  Trovati {len(mtx_files)} file .mtx:")
    for f in mtx_files:
        print(f"   - {os.path.basename(f)}")
    
    # Testa tutte le matrici
    all_matrices_results = {}
    
    for matrix_path in mtx_files:
        results = test_single_matrix(matrix_path, args.tolerances, args.output)
        if results:
            matrix_name = os.path.splitext(os.path.basename(matrix_path))[0]
            all_matrices_results[matrix_name] = results
    
    # Riepilogo finale
    if all_matrices_results:
        print(f"\n RIEPILOGO FINALE")
        print("=" * 60)
        print(f"Matrici testate: {len(all_matrices_results)}")
        print(f"Metodi per matrice: 4")
        print(f"Tolleranze per metodo: {len(args.tolerances)}")
        print(f"Total test eseguiti: {len(all_matrices_results) * 4 * len(args.tolerances)}")
        print(f"\n Tutti i risultati salvati in {args.output}/")


if __name__ == "__main__":
    main()