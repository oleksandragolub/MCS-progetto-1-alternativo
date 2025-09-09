#!/usr/bin/env python3
"""
Test per verificare che scipy.io.mmread gestisca correttamente 
la conversione da indicizzazione 1-based a 0-based per file .mtx
"""

import numpy as np
import scipy.io
import tempfile
import os

def test_mtx_indexing():
    """
    Test per verificare la gestione dell'indicizzazione nei file .mtx
    """
    
    print("Test della gestione indicizzazione .mtx")
    print("=" * 50)
    
    # Crea un file .mtx di test con indicizzazione 1-based
    mtx_content_1based = """%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 2.0
2 1 -1.0
2 2 2.0
3 3 2.0
"""
    
    # Crea un file temporaneo
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mtx', delete=False) as f:
        f.write(mtx_content_1based)
        temp_filename = f.name
    
    try:
        print("Contenuto file .mtx (indicizzazione 1-based):")
        print(mtx_content_1based)
        
        # Carica la matrice usando scipy
        print("Caricamento con scipy.io.mmread()...")
        matrix_sparse = scipy.io.mmread(temp_filename)
        matrix_dense = matrix_sparse.toarray()
        
        print(f"Matrice caricata (shape: {matrix_dense.shape}):")
        print(matrix_dense)
        print()
        
        # Verifica che la matrice sia quella attesa (con indicizzazione 0-based)
        expected_matrix = np.array([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        print("Matrice attesa (dopo conversione a 0-based):")
        print(expected_matrix)
        print()
        
        # Confronta
        if np.allclose(matrix_dense, expected_matrix):
            print(" TEST SUPERATO: scipy.io.mmread gestisce correttamente la conversione!")
            print("   Gli indici 1-based del file .mtx sono stati convertiti correttamente a 0-based")
        else:
            print(" TEST FALLITO: Possibile problema con la conversione degli indici")
            print("   Differenza trovata tra matrice caricata e quella attesa")
            
        # Test aggiuntivo: verifica simmetria
        if np.allclose(matrix_dense, matrix_dense.T):
            print(" La matrice è correttamente simmetrica")
        else:
            print(" Problema: la matrice non risulta simmetrica")
            
    finally:
        # Pulisci il file temporaneo
        os.unlink(temp_filename)
    
    print("\n" + "=" * 50)

def test_with_actual_matrix(matrix_path: str):
    """
    Test aggiuntivo con una delle tue matrici reali
    """
    if not os.path.exists(matrix_path):
        print(f"File {matrix_path} non trovato, salto il test")
        return
        
    print(f"\nTest con matrice reale: {os.path.basename(matrix_path)}")
    print("-" * 40)
    
    try:
        # Carica la matrice
        matrix_sparse = scipy.io.mmread(matrix_path)
        
        # Verifica che sia in formato sparse
        print(f"Tipo matrice caricata: {type(matrix_sparse)}")
        print(f"Shape: {matrix_sparse.shape}")
        print(f"Elementi non zero: {matrix_sparse.nnz}")
        
        # Converti a denso (solo per matrici piccole!)
        if matrix_sparse.shape[0] <= 100:  # Solo per matrici piccole
            matrix_dense = matrix_sparse.toarray()
            print(f"Primi elementi (0,0): {matrix_dense[0,0]}")
            print(f"Simmetrica: {np.allclose(matrix_dense, matrix_dense.T, rtol=1e-12)}")
        else:
            print("Matrice troppo grande per conversione completa a denso")
            # Verifica solo alcuni elementi
            print(f"Elemento (0,0): {matrix_sparse[0,0]}")
            print(f"Elemento (1,1): {matrix_sparse[1,1]}")
            
        print(" Caricamento della matrice reale completato con successo")
        
    except Exception as e:
        print(f" Errore nel caricamento: {str(e)}")

def check_mtx_file_format(filepath: str):
    """
    Verifica il formato del file .mtx ispezionando le prime righe
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} non trovato")
        return
        
    print(f"\nIspezione formato file: {os.path.basename(filepath)}")
    print("-" * 40)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Mostra header
    print("Prime righe del file:")
    for i, line in enumerate(lines[:10]):  # Prime 10 righe
        print(f"{i+1:2d}: {line.rstrip()}")
        
    # Cerca la riga con le dimensioni
    for line in lines:
        if not line.startswith('%') and line.strip():
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    rows, cols, nnz = map(int, parts[:3])
                    print(f"\nDimensioni trovate nel file:")
                    print(f"  Righe: {rows}")
                    print(f"  Colonne: {cols}") 
                    print(f"  Elementi non zero: {nnz}")
                    break
                except ValueError:
                    continue

if __name__ == "__main__":
    # Test principale
    test_mtx_indexing()
    
    # Test con matrici reali (sostituisci con i tuoi percorsi)
    test_matrices = [
        "data/spa1.mtx",
        "data/vem1.mtx"
    ]
    
    for matrix_path in test_matrices:
        test_with_actual_matrix(matrix_path)
        check_mtx_file_format(matrix_path)