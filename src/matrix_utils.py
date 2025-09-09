"""
Utility per la gestione di matrici e file Matrix Market (.mtx)
"""

import numpy as np
import scipy.io
from scipy.sparse import csr_matrix, coo_matrix
from typing import Tuple, Union
import os


def load_mtx_matrix(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Carica una matrice dal formato Matrix Market (.mtx).
    
    Args:
        filepath: Percorso al file .mtx
        
    Returns:
        Tuple contenente:
        - Matrice come array NumPy denso
        - Dizionario con informazioni sulla matrice
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} non trovato")
    
    try:
        # Carica la matrice usando scipy
        matrix_sparse = scipy.io.mmread(filepath)
        
        # Converti in formato denso
        if hasattr(matrix_sparse, 'toarray'):
            matrix_dense = matrix_sparse.toarray()
        else:
            matrix_dense = np.array(matrix_sparse)
        
        # Informazioni sulla matrice
        info = {
            'filename': os.path.basename(filepath),
            'shape': matrix_dense.shape,
            'nnz': np.count_nonzero(matrix_dense),  # Numero elementi non zero
            'sparsity': 1 - (np.count_nonzero(matrix_dense) / (matrix_dense.shape[0] * matrix_dense.shape[1])),
            'is_symmetric': np.allclose(matrix_dense, matrix_dense.T, rtol=1e-12),
            'min_eigenvalue': None,  # Calcolato dopo se necessario
            'max_eigenvalue': None,
            'condition_number': None
        }
        
        print(f" Caricata matrice {info['filename']}")
        print(f"   Dimensione: {info['shape'][0]}x{info['shape'][1]}")
        print(f"   Elementi non zero: {info['nnz']}")
        print(f"   Sparsità: {info['sparsity']:.2%}")
        print(f"   Simmetrica: {info['is_symmetric']}")
        
        return matrix_dense, info
        
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento del file {filepath}: {str(e)}")


def check_matrix_properties(A: np.ndarray, compute_eigenvalues: bool = False) -> dict:
    """
    Verifica le proprietà di una matrice (simmetria, definitezza positiva).
    
    Args:
        A: Matrice da verificare
        compute_eigenvalues: Se calcolare gli autovalori (può essere costoso)
        
    Returns:
        Dizionario con le proprietà della matrice
    """
    n, m = A.shape
    
    properties = {
        'is_square': n == m,
        'is_symmetric': False,
        'is_positive_definite': False,
        'min_eigenvalue': None,
        'max_eigenvalue': None,
        'condition_number': None,
        'errors': []
    }
    
    if not properties['is_square']:
        properties['errors'].append("Matrice non quadrata")
        return properties
    
    # Verifica simmetria
    if np.allclose(A, A.T, rtol=1e-12):
        properties['is_symmetric'] = True
    else:
        properties['errors'].append("Matrice non simmetrica")
    
    # Verifica definitezza positiva
    if compute_eigenvalues and properties['is_symmetric']:
        try:
            eigenvals = np.linalg.eigvals(A)
            properties['min_eigenvalue'] = np.min(eigenvals)
            properties['max_eigenvalue'] = np.max(eigenvals)
            
            if properties['min_eigenvalue'] > 0:
                properties['is_positive_definite'] = True
                properties['condition_number'] = properties['max_eigenvalue'] / properties['min_eigenvalue']
            else:
                properties['errors'].append("Matrice non definita positiva")
        except Exception as e:
            properties['errors'].append(f"Errore nel calcolo autovalori: {str(e)}")
    
    return properties


def create_test_system(n: int, matrix_type: str = 'tridiagonal') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea un sistema test Ax = b con soluzione nota x = [1,1,...,1].
    
    Args:
        n: Dimensione del sistema
        matrix_type: Tipo di matrice ('tridiagonal', 'diagonal_dominant', 'hilbert')
        
    Returns:
        Tuple (A, b, x_exact) dove x_exact = [1,1,...,1]
    """
    x_exact = np.ones(n)
    
    if matrix_type == 'tridiagonal':
        # Matrice tridiagonale simmetrica e definita positiva
        # Diagonale principale = 3, diagonali superiore/inferiore = -1
        A = np.diag(3 * np.ones(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
        
    elif matrix_type == 'diagonal_dominant':
        # Matrice diagonalmente dominante
        A = np.random.rand(n, n)
        A = (A + A.T) / 2  # Rende simmetrica
        A += (n + 1) * np.eye(n)  # Rende diagonalmente dominante e definita positiva
        
    elif matrix_type == 'hilbert':
        # Matrice di Hilbert (mal condizionata)
        A = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])
        
    else:
        raise ValueError(f"Tipo di matrice '{matrix_type}' non supportato")
    
    # Calcola b = A * x_exact
    b = A @ x_exact
    
    return A, b, x_exact


def generate_standard_test_system(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera il sistema test standard come richiesto dal progetto:
    x_exact = [1,1,...,1], b = A * x_exact
    
    Args:
        A: Matrice del sistema
        
    Returns:
        Tuple (b, x_exact)
    """
    n = A.shape[0]
    x_exact = np.ones(n)  # Soluzione esatta = [1,1,...,1]
    b = A @ x_exact       # b = A * x_exact
    
    return b, x_exact


def print_matrix_info(A: np.ndarray, name: str = "Matrix"):
    """
    Stampa informazioni dettagliate su una matrice.
    
    Args:
        A: Matrice da analizzare
        name: Nome della matrice per l'output
    """
    print(f"\n Informazioni {name}:")
    print(f"   Dimensione: {A.shape[0]}x{A.shape[1]}")
    print(f"   Elementi non zero: {np.count_nonzero(A)}")
    print(f"   Norma di Frobenius: {np.linalg.norm(A, 'fro'):.4e}")
    
    if A.shape[0] == A.shape[1]:  # Matrice quadrata
        properties = check_matrix_properties(A, compute_eigenvalues=True)
        
        print(f"   Simmetrica: {properties['is_symmetric']}")
        print(f"   Definita positiva: {properties['is_positive_definite']}")
        
        if properties['min_eigenvalue'] is not None:
            print(f"   Autovalore minimo: {properties['min_eigenvalue']:.4e}")
            print(f"   Autovalore massimo: {properties['max_eigenvalue']:.4e}")
            print(f"   Numero di condizione: {properties['condition_number']:.4e}")
        
        if properties['errors']:
            print(f"     Problemi rilevati: {', '.join(properties['errors'])}")


def save_results_to_file(results: dict, filepath: str, matrix_name: str, tolerance: float):
    """
    Salva i risultati in un file di testo.
    
    Args:
        results: Risultati dei metodi iterativi
        filepath: Percorso del file di output
        matrix_name: Nome della matrice testata
        tolerance: Tolleranza utilizzata
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Risultati per matrice: {matrix_name}\n")
        f.write(f"Tolleranza: {tolerance:.0e}\n")
        f.write("="*60 + "\n\n")
        
        for method_name, result in results.items():
            f.write(f"Metodo: {result['method']}\n")
            f.write(f"Convergenza: {'Sì' if result['converged'] else 'No'}\n")
            f.write(f"Iterazioni: {result['iterations']}\n")
            f.write(f"Tempo: {result['time']:.6f} s\n")
            f.write(f"Errore residuo: {result['residual_norm']:.6e}\n")
            
            if 'relative_error' in result:
                f.write(f"Errore relativo: {result['relative_error']:.6e}\n")
            
            f.write("-"*40 + "\n\n")
    
    print(f" Risultati salvati in {filepath}")


def create_comparison_table(all_results: dict) -> str:
    """
    Crea una tabella di confronto dei risultati per diversi valori di tolleranza.
    
    Args:
        all_results: Dizionario con risultati per diverse tolleranze
        
    Returns:
        Stringa con la tabella formattata
    """
    table_lines = []
    table_lines.append("TABELLA DI CONFRONTO METODI ITERATIVI")
    table_lines.append("="*80)
    
    # Header
    header = f"{'Tolleranza':<12} {'Metodo':<18} {'Conv.':<6} {'Iter.':<8} {'Tempo (s)':<12} {'Err. Rel.':<12}"
    table_lines.append(header)
    table_lines.append("-"*80)
    
    # Dati
    for tol_str, results in all_results.items():
        first_method = True
        for method_name, result in results.items():
            tol_display = tol_str if first_method else ""
            conv_display = "Sì" if result['converged'] else "No"
            
            line = f"{tol_display:<12} {result['method']:<18} {conv_display:<6} "
            line += f"{result['iterations']:<8} {result['time']:<12.6f} "
            
            if 'relative_error' in result and result['relative_error'] != float('inf'):
                line += f"{result['relative_error']:<12.6e}"
            else:
                line += f"{'∞':<12}"
            
            table_lines.append(line)
            first_method = False
        
        table_lines.append("-"*80)
    
    return "\n".join(table_lines)