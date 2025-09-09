"""
Mini libreria per metodi iterativi
"""

from .linear_solvers import LinearIterativeSolver
from .matrix_utils import (
    load_mtx_matrix,
    check_matrix_properties, 
    create_test_system,
    generate_standard_test_system,
    print_matrix_info,
    save_results_to_file,
    create_comparison_table
)

__version__ = "1.0.0"
__author__ = "Oleksandra Golub"

__all__ = [
    'LinearIterativeSolver',
    'load_mtx_matrix',
    'check_matrix_properties',
    'create_test_system', 
    'generate_standard_test_system',
    'print_matrix_info',
    'save_results_to_file',
    'create_comparison_table'
]