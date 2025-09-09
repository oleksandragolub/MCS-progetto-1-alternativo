#!/usr/bin/env python3
"""
Implementazione della decomposizione di Cholesky da zero
per verificare matrici simmetriche definite positive (SPD)
"""

import numpy as np
import sys
import os

# Aggiungi il percorso del progetto per importare le tue matrici
sys.path.append('.')

def cholesky_decomposition(A):
    """
    Implementa la decomposizione di Cholesky A = L*L^T da zero.
    
    Args:
        A: Matrice simmetrica definita positiva
        
    Returns:
        L: Matrice triangolare inferiore tale che A = L*L^T
        
    Raises:
        ValueError: Se la matrice non è SPD
    """
    n = A.shape[0]
    
    # Verifica che sia quadrata
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matrice deve essere quadrata")
    
    # Verifica simmetria
    if not np.allclose(A, A.T, rtol=1e-12):
        raise ValueError("La matrice deve essere simmetrica")
    
    # Inizializza L come matrice di zeri
    L = np.zeros((n, n))
    
    # Algoritmo di Cholesky
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Elementi diagonali
                # L[i,i] = sqrt(A[i,i] - sum(L[i,k]^2 for k=0..i-1))
                sum_sq = sum(L[i, k]**2 for k in range(j))
                diagonal_element = A[i, i] - sum_sq
                
                if diagonal_element <= 0:
                    raise ValueError(f"Matrice non definita positiva: elemento diagonale {diagonal_element} <= 0 alla posizione ({i},{i})")
                
                L[i, j] = np.sqrt(diagonal_element)
                
            else:  # Elementi sotto la diagonale
                # L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k=0..j-1)) / L[j,j]
                sum_prod = sum(L[i, k] * L[j, k] for k in range(j))
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]
    
    return L


def verify_cholesky(A, L):
    """
    Verifica che L*L^T = A entro tolleranza numerica.
    
    Args:
        A: Matrice originale
        L: Fattore di Cholesky
        
    Returns:
        bool: True se la decomposizione è corretta
    """
    reconstructed = L @ L.T
    return np.allclose(A, reconstructed, rtol=1e-12, atol=1e-14)


def is_spd_cholesky(A):
    """
    Verifica se una matrice è SPD tentando la decomposizione di Cholesky.
    
    Args:
        A: Matrice da testare
        
    Returns:
        tuple: (is_spd, L_factor, error_message)
    """
    try:
        L = cholesky_decomposition(A)
        
        # Verifica che la decomposizione sia corretta
        if verify_cholesky(A, L):
            return True, L, None
        else:
            return False, None, "Errore nella ricostruzione A = L*L^T"
            
    except ValueError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Errore inaspettato: {str(e)}"


def test_cholesky_examples():
    """Test su esempi noti."""
    print("TESTING DECOMPOSIZIONE DI CHOLESKY")
    print("="*50)
    
    # Test 1: Matrice 2x2 semplice
    print("\n1. Test matrice 2x2 semplice:")
    A1 = np.array([[4.0, 2.0],
                   [2.0, 3.0]])
    
    print("A1 =")
    print(A1)
    
    is_spd, L, error = is_spd_cholesky(A1)
    
    if is_spd:
        print("✓ Matrice SPD")
        print("L =")
        print(L)
        print("Verifica L*L^T =")
        print(L @ L.T)
        print("Differenza ||A - L*L^T|| =", np.linalg.norm(A1 - L @ L.T))
    else:
        print(f"✗ Non SPD: {error}")
    
    # Test 2: Matrice tridiagonale classica
    print("\n2. Test matrice tridiagonale 3x3:")
    A2 = np.array([[2.0, -1.0, 0.0],
                   [-1.0, 2.0, -1.0],
                   [0.0, -1.0, 2.0]])
    
    print("A2 =")
    print(A2)
    
    is_spd, L, error = is_spd_cholesky(A2)
    
    if is_spd:
        print("✓ Matrice SPD")
        print("L =")
        print(L)
        # Calcola numero di condizione
        eigenvals = np.linalg.eigvals(A2)
        cond_num = np.max(eigenvals) / np.min(eigenvals)
        print(f"Numero di condizione: {cond_num:.4f}")
    else:
        print(f"✗ Non SPD: {error}")
    
    # Test 3: Matrice NON definita positiva
    print("\n3. Test matrice NON definita positiva:")
    A3 = np.array([[1.0, 2.0],
                   [2.0, 1.0]])  # Autovalori: 3, -1
    
    print("A3 =")
    print(A3)
    
    is_spd, L, error = is_spd_cholesky(A3)
    
    if is_spd:
        print("✗ ERRORE: rilevata come SPD ma non dovrebbe esserlo!")
    else:
        print(f"✓ Correttamente rilevata come non SPD: {error}")
    
    # Test 4: Matrice NON simmetrica
    print("\n4. Test matrice NON simmetrica:")
    A4 = np.array([[1.0, 2.0],
                   [3.0, 4.0]])
    
    print("A4 =")
    print(A4)
    
    is_spd, L, error = is_spd_cholesky(A4)
    
    if is_spd:
        print("✗ ERRORE: rilevata come SPD ma non è simmetrica!")
    else:
        print(f"✓ Correttamente rilevata come non simmetrica: {error}")


def test_project_matrices():
    """Test sulle matrici del progetto."""
    print("\n" + "="*50)
    print("TEST SULLE MATRICI DEL PROGETTO")
    print("="*50)
    
    # Importa le funzioni del progetto
    try:
        from src import load_mtx_matrix
    except ImportError:
        print("Impossibile importare le funzioni del progetto")
        return
    
    # Lista delle matrici da testare
    matrices = ['spa1.mtx', 'spa2.mtx', 'vem1.mtx', 'vem2.mtx']
    
    for matrix_name in matrices:
        matrix_path = f"data/{matrix_name}"
        
        if not os.path.exists(matrix_path):
            print(f"\n{matrix_name}: File non trovato in {matrix_path}")
            continue
        
        print(f"\n{matrix_name}:")
        print("-" * 30)
        
        try:
            # Carica la matrice
            A, info = load_mtx_matrix(matrix_path)
            
            print(f"Dimensione: {A.shape[0]}x{A.shape[1]}")
            print(f"Sparsità: {info['sparsity']:.2%}")
            
            # Test Cholesky (solo su sottomatrice per matrici grandi)
            if A.shape[0] > 100:
                print("Matrice troppo grande per Cholesky completa")
                print("Test su sottomatrice 10x10 nell'angolo superiore sinistro:")
                A_sub = A[:10, :10]
                is_spd, L, error = is_spd_cholesky(A_sub)
            else:
                print("Test Cholesky sulla matrice completa:")
                is_spd, L, error = is_spd_cholesky(A)
                A_sub = A
            
            if is_spd:
                print("✓ Sottomatrice SPD")
                eigenvals = np.linalg.eigvals(A_sub)
                print(f"  Autovalori: min={np.min(eigenvals):.4e}, max={np.max(eigenvals):.4e}")
                print(f"  Numero di condizione: {np.max(eigenvals)/np.min(eigenvals):.4e}")
            else:
                print(f"✗ Non SPD: {error}")
                
        except Exception as e:
            print(f"Errore nel caricamento: {str(e)}")


def compare_with_numpy():
    """Confronta la nostra implementazione con numpy."""
    print("\n" + "="*50)
    print("CONFRONTO CON NUMPY")
    print("="*50)
    
    # Matrice test
    A = np.array([[4.0, 2.0, 1.0],
                  [2.0, 5.0, 3.0],
                  [1.0, 3.0, 6.0]])
    
    print("Matrice test:")
    print(A)
    
    # Nostra implementazione
    print("\nNostra implementazione:")
    is_spd, L_ours, error = is_spd_cholesky(A)
    
    if is_spd:
        print("L (nostra) =")
        print(L_ours)
    
    # NumPy/SciPy
    print("\nNumPy (scipy.linalg.cholesky):")
    try:
        from scipy.linalg import cholesky
        L_numpy = cholesky(A, lower=True)
        print("L (numpy) =")
        print(L_numpy)
        
        # Confronto
        if is_spd:
            diff = np.linalg.norm(L_ours - L_numpy)
            print(f"\nDifferenza ||L_ours - L_numpy|| = {diff:.2e}")
            
            if diff < 1e-12:
                print("✓ Implementazioni identiche entro precisione macchina")
            else:
                print("✗ Differenze significative")
        
    except ImportError:
        print("SciPy non disponibile per il confronto")
    except Exception as e:
        print(f"Errore nel confronto: {e}")


if __name__ == "__main__":
    # Esegui tutti i test
    test_cholesky_examples()
    test_project_matrices()
    compare_with_numpy()
    
    print("\n" + "="*50)
    print("RIEPILOGO SUL METODO DI CHOLESKY")
    print("="*50)
    print("""
Il metodo di Cholesky per verificare le matrici SPD:

1. PRINCIPIO: Una matrice A è SPD se e solo se esiste una matrice 
   triangolare inferiore L tale che A = L*L^T

2. ALGORITMO: 
   - Calcola L elemento per elemento
   - Se durante il calcolo si ottiene sqrt(numero negativo) 
     → la matrice NON è definita positiva
   - Se tutti gli elementi diagonali sono positivi → matrice SPD

3. VANTAGGI:
   - Test diretto: fallisce immediatamente se A non è SPD
   - Computazionalmente efficiente: O(n³/3)
   - Fornisce anche la fattorizzazione utile per risolvere sistemi

4. SVANTAGGI:
   - Meno stabile numericamente degli autovalori
   - Se la matrice è "quasi" singolare può dare falsi negativi

Per il tuo progetto, il metodo degli autovalori che hai usato è più robusto
e fornisce informazioni aggiuntive (numero di condizione).
""")