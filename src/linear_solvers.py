"""
Mini libreria per la risoluzione di sistemi lineari con metodi iterativi

Metodi implementati:
1. Jacobi
2. Gauss-Seidel  
3. Gradiente
4. Gradiente Coniugato

Limitato a matrici simmetriche e definite positive
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, Any


class LinearIterativeSolver:
    """
    Classe principale per i metodi iterativi di risoluzione di sistemi lineari.
    """
    
    def __init__(self, max_iter: int = 20000, verbose: bool = True):
        """
        Inizializza il solver.
        
        Args:
            max_iter: Numero massimo di iterazioni (>=20000 per il progetto)
            verbose: Se stampare informazioni durante l'esecuzione
        """
        self.max_iter = max_iter
        self.verbose = verbose
        self.results_history = []
    
    def _check_matrix_properties(self, A: np.ndarray) -> bool:
        """
        Verifica che la matrice sia simmetrica e definita positiva.
        
        Args:
            A: Matrice del sistema
            
        Returns:
            True se la matrice soddisfa le proprietà richieste
        """
        n, m = A.shape
        if n != m:
            raise ValueError("La matrice deve essere quadrata")
        
        # Verifica simmetria
        if not np.allclose(A, A.T, rtol=1e-12):
            raise ValueError("La matrice deve essere simmetrica")
        
        # Verifica definitezza positiva tramite autovalori
        eigenvals = np.linalg.eigvals(A)
        if np.any(eigenvals <= 0):
            raise ValueError("La matrice deve essere definita positiva")
        
        return True
    
    def _compute_residual_norm(self, A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
        """
        Calcola la norma del residuo ||Ax - b|| / ||b||.
        
        Args:
            A: Matrice del sistema
            x: Vettore soluzione corrente
            b: Termine noto
            
        Returns:
            Norma relativa del residuo
        """
        residual = A @ x - b
        return np.linalg.norm(residual) / np.linalg.norm(b)
    
    def jacobi(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-6, 
               x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Metodo di Jacobi per la risoluzione di sistemi lineari.
        
        Args:
            A: Matrice del sistema (simmetrica e definita positiva)
            b: Termine noto
            tol: Tolleranza per il criterio di arresto
            x0: Guess iniziale (se None, usa il vettore nullo)
            
        Returns:
            Dizionario con risultati: soluzione, iterazioni, tempo, errore
        """
        self._check_matrix_properties(A)
        n = len(b)
        
        # Vettore iniziale nullo (come richiesto dal progetto)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        # Estrazione delle matrici per Jacobi: x^(k+1) = D^(-1) * (b - (L+U) * x^(k))
        D = np.diag(np.diag(A))  # Matrice diagonale
        D_inv = np.diag(1.0 / np.diag(A))  # Inversa della diagonale
        LU = A - D  # L + U (matrici triangolari inferiore e superiore)
        
        # Verifica che non ci siano zeri sulla diagonale
        if np.any(np.diag(A) == 0):
            raise ValueError("La matrice ha elementi nulli sulla diagonale")
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            x_new = D_inv @ (b - LU @ x)
            
            # Criterio di arresto: ||Ax^(k) - b|| / ||b|| < tol
            error = self._compute_residual_norm(A, x_new, b)
            
            if error < tol:
                elapsed_time = time.time() - start_time
                
                result = {
                    'solution': x_new,
                    'iterations': k + 1,
                    'time': elapsed_time,
                    'residual_norm': error,
                    'converged': True,
                    'method': 'Jacobi'
                }
                
                if self.verbose:
                    print(f"Jacobi converged in {k+1} iterations, "
                          f"residual: {error:.2e}, time: {elapsed_time:.4f}s")
                
                return result
            
            x = x_new
        
        # Se non converge entro max_iter
        elapsed_time = time.time() - start_time
        final_error = self._compute_residual_norm(A, x, b)
        
        result = {
            'solution': x,
            'iterations': self.max_iter,
            'time': elapsed_time,
            'residual_norm': final_error,
            'converged': False,
            'method': 'Jacobi'
        }
        
        if self.verbose:
            print(f"Jacobi did not converge in {self.max_iter} iterations, "
                  f"final residual: {final_error:.2e}")
        
        return result
    
    def gauss_seidel(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-6,
                     x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Metodo di Gauss-Seidel per la risoluzione di sistemi lineari.
        
        Args:
            A: Matrice del sistema (simmetrica e definita positiva)
            b: Termine noto
            tol: Tolleranza per il criterio di arresto
            x0: Guess iniziale (se None, usa il vettore nullo)
            
        Returns:
            Dizionario con risultati: soluzione, iterazioni, tempo, errore
        """
        self._check_matrix_properties(A)
        n = len(b)
        
        # Vettore iniziale nullo
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        # Estrazione matrici per Gauss-Seidel: (D+L)x^(k+1) = b - U*x^(k)
        L = np.tril(A, k=-1)  # Triangolare inferiore stretta
        D = np.diag(np.diag(A))  # Diagonale
        U = np.triu(A, k=1)   # Triangolare superiore stretta
        DL = D + L  # D + L
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            x_new = x.copy()
            
            # Risoluzione per forward substitution: (D+L)x = b - U*x_old
            rhs = b - U @ x
            
            # Forward substitution manuale
            for i in range(n):
                sum_ax = sum(DL[i, j] * x_new[j] for j in range(i))
                x_new[i] = (rhs[i] - sum_ax) / DL[i, i]
            
            # Criterio di arresto
            error = self._compute_residual_norm(A, x_new, b)
            
            if error < tol:
                elapsed_time = time.time() - start_time
                
                result = {
                    'solution': x_new,
                    'iterations': k + 1,
                    'time': elapsed_time,
                    'residual_norm': error,
                    'converged': True,
                    'method': 'Gauss-Seidel'
                }
                
                if self.verbose:
                    print(f"Gauss-Seidel converged in {k+1} iterations, "
                          f"residual: {error:.2e}, time: {elapsed_time:.4f}s")
                
                return result
            
            x = x_new
        
        # Se non converge
        elapsed_time = time.time() - start_time
        final_error = self._compute_residual_norm(A, x, b)
        
        result = {
            'solution': x,
            'iterations': self.max_iter,
            'time': elapsed_time,
            'residual_norm': final_error,
            'converged': False,
            'method': 'Gauss-Seidel'
        }
        
        if self.verbose:
            print(f"Gauss-Seidel did not converge in {self.max_iter} iterations, "
                  f"final residual: {final_error:.2e}")
        
        return result
    
    def gradient_method(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-6,
                       x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Metodo del Gradiente per la risoluzione di sistemi lineari.
        
        Args:
            A: Matrice del sistema (simmetrica e definita positiva)
            b: Termine noto
            tol: Tolleranza per il criterio di arresto
            x0: Guess iniziale (se None, usa il vettore nullo)
            
        Returns:
            Dizionario con risultati: soluzione, iterazioni, tempo, errore
        """
        self._check_matrix_properties(A)
        n = len(b)
        
        # Vettore iniziale nullo
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            # Calcolo del residuo r = b - Ax
            residual = b - A @ x
            
            # Calcolo step size ottimale: α = (r^T * r) / (r^T * A * r)
            r_dot_r = residual.T @ residual
            r_A_r = residual.T @ (A @ residual)
            
            if r_A_r == 0:  # Evita divisione per zero
                break
                
            alpha = r_dot_r / r_A_r
            
            # Aggiornamento della soluzione
            x_new = x + alpha * residual
            
            # Criterio di arresto
            error = self._compute_residual_norm(A, x_new, b)
            
            if error < tol:
                elapsed_time = time.time() - start_time
                
                result = {
                    'solution': x_new,
                    'iterations': k + 1,
                    'time': elapsed_time,
                    'residual_norm': error,
                    'converged': True,
                    'method': 'Gradient'
                }
                
                if self.verbose:
                    print(f"Gradient Method converged in {k+1} iterations, "
                          f"residual: {error:.2e}, time: {elapsed_time:.4f}s")
                
                return result
            
            x = x_new
        
        # Se non converge
        elapsed_time = time.time() - start_time
        final_error = self._compute_residual_norm(A, x, b)
        
        result = {
            'solution': x,
            'iterations': self.max_iter,
            'time': elapsed_time,
            'residual_norm': final_error,
            'converged': False,
            'method': 'Gradient'
        }
        
        if self.verbose:
            print(f"Gradient Method did not converge in {self.max_iter} iterations, "
                  f"final residual: {final_error:.2e}")
        
        return result
    
    def conjugate_gradient(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-6,
                          x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Metodo del Gradiente Coniugato per la risoluzione di sistemi lineari.
        
        Args:
            A: Matrice del sistema (simmetrica e definita positiva)
            b: Termine noto
            tol: Tolleranza per il criterio di arresto
            x0: Guess iniziale (se None, usa il vettore nullo)
            
        Returns:
            Dizionario con risultati: soluzione, iterazioni, tempo, errore
        """
        self._check_matrix_properties(A)
        n = len(b)
        
        # Vettore iniziale nullo
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        # Inizializzazione
        r = b - A @ x  # Residuo iniziale
        p = r.copy()   # Direzione iniziale
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            # Calcolo step size: α = (r^T * r) / (p^T * A * p)
            Ap = A @ p
            r_dot_r = r.T @ r
            p_Ap = p.T @ Ap
            
            if p_Ap == 0:  # Evita divisione per zero
                break
                
            alpha = r_dot_r / p_Ap
            
            # Aggiornamento soluzione
            x_new = x + alpha * p
            
            # Aggiornamento residuo
            r_new = r - alpha * Ap
            
            # Criterio di arresto
            error = self._compute_residual_norm(A, x_new, b)
            
            if error < tol:
                elapsed_time = time.time() - start_time
                
                result = {
                    'solution': x_new,
                    'iterations': k + 1,
                    'time': elapsed_time,
                    'residual_norm': error,
                    'converged': True,
                    'method': 'Conjugate Gradient'
                }
                
                if self.verbose:
                    print(f"Conjugate Gradient converged in {k+1} iterations, "
                          f"residual: {error:.2e}, time: {elapsed_time:.4f}s")
                
                return result
            
            # Calcolo parametro β per la nuova direzione
            r_new_dot = r_new.T @ r_new
            beta = r_new_dot / r_dot_r
            
            # Aggiornamento direzione
            p_new = r_new + beta * p
            
            # Aggiornamento per prossima iterazione
            x, r, p = x_new, r_new, p_new
        
        # Se non converge
        elapsed_time = time.time() - start_time
        final_error = self._compute_residual_norm(A, x, b)
        
        result = {
            'solution': x,
            'iterations': self.max_iter,
            'time': elapsed_time,
            'residual_norm': final_error,
            'converged': False,
            'method': 'Conjugate Gradient'
        }
        
        if self.verbose:
            print(f"Conjugate Gradient did not converge in {self.max_iter} iterations, "
                  f"final residual: {final_error:.2e}")
        
        return result
    
    def solve_all_methods(self, A: np.ndarray, b: np.ndarray, x_exact: np.ndarray,
                         tol: float = 1e-6) -> Dict[str, Dict[str, Any]]:
        """
        Risolve il sistema con tutti i 4 metodi e calcola gli errori relativi.
        
        Args:
            A: Matrice del sistema
            b: Termine noto
            x_exact: Soluzione esatta per il calcolo dell'errore
            tol: Tolleranza
            
        Returns:
            Dizionario con i risultati di tutti i metodi
        """
        methods = ['jacobi', 'gauss_seidel', 'gradient_method', 'conjugate_gradient']
        results = {}
        
        for method_name in methods:
            method = getattr(self, method_name)
            result = method(A, b, tol)
            
            # Calcolo errore relativo rispetto alla soluzione esatta
            if result['converged']:
                relative_error = np.linalg.norm(x_exact - result['solution']) / np.linalg.norm(x_exact)
                result['relative_error'] = relative_error
            else:
                result['relative_error'] = float('inf')
            
            results[method_name] = result
        
        return results