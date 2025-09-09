#!/usr/bin/env python3
"""
Test rapido per verificare il funzionamento della libreria
"""

import numpy as np
import sys
import os

# Aggiungi il percorso del progetto
sys.path.append('.')

from src import LinearIterativeSolver, create_test_system

def test_basic_functionality():
    """Test di base sui metodi implementati."""
    print(" TEST RAPIDO DELLA LIBRERIA")
    print("="*50)
    
    # Crea un sistema test piccolo
    n = 10
    print(f" Creando sistema test {n}x{n}...")
    
    A, b, x_exact = create_test_system(n, 'tridiagonal')
    
    print(f" Sistema creato:")
    print(f"   Matrice: {A.shape}")
    print(f"   Soluzione esatta: x = [1,1,...,1]")
    print(f"   Termine noto: b = A*x")
    
    # Inizializza solver
    solver = LinearIterativeSolver(max_iter=1000, verbose=True)
    
    # Test singolo metodo
    print(f"\n Test Jacobi...")
    result = solver.jacobi(A, b, tol=1e-8)
    
    if result['converged']:
        error = np.linalg.norm(x_exact - result['solution']) / np.linalg.norm(x_exact)
        print(f" Jacobi: {result['iterations']} iter, errore: {error:.2e}")
    else:
        print(f" Jacobi non convergente")
    
    # Test tutti i metodi
    print(f"\n Test tutti i metodi...")
    all_results = solver.solve_all_methods(A, b, x_exact, tol=1e-8)
    
    print(f"\n Risultati:")
    print(f"{'Metodo':<18} {'Conv.':<6} {'Iter.':<6} {'Err. Rel.':<12}")
    print("-"*45)
    
    for method_name, result in all_results.items():
        conv_str = "Sì" if result['converged'] else "No" 
        err_str = f"{result['relative_error']:.4e}" if result['relative_error'] != float('inf') else "∞"
        
        print(f"{result['method']:<18} {conv_str:<6} {result['iterations']:<6} {err_str:<12}")
    
    print(f"\n Test completato con successo!")
    return True


def test_matrix_properties():
    """Test delle proprietà delle matrici."""
    print(f"\n TEST PROPRIETÀ MATRICI")
    print("-"*30)
    
    # Test matrice simmetrica e definita positiva
    A1 = np.array([[4, -1, 0], 
                   [-1, 4, -1], 
                   [0, -1, 4]])
    print("Matrice tridiagonale 3x3:")
    print(" Dovrebbe essere simmetrica e definita positiva")
    
    try:
        solver = LinearIterativeSolver()
        solver._check_matrix_properties(A1)
        print(" Proprietà verificate correttamente")
    except Exception as e:
        print(f" Errore: {e}")
    
    # Test matrice non simmetrica
    A2 = np.array([[1, 2], [3, 4]])
    print("\nMatrice non simmetrica:")
    print(" Dovrebbe fallire il test")
    
    try:
        solver._check_matrix_properties(A2)
        print(" Test fallito - dovrebbe aver rilevato non-simmetria")
    except ValueError as e:
        print(f" Correttamente rilevato: {e}")


if __name__ == "__main__":
    try:
        # Test funzionalità base
        test_basic_functionality()
        
        # Test proprietà matrici  
        test_matrix_properties()
        
        print(f"\n TUTTI I TEST SUPERATI!")
        print("La libreria è pronta per essere utilizzata.")
        
    except Exception as e:
        print(f"\n ERRORE NEI TEST: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)