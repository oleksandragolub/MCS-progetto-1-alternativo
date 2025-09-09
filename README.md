# MCS-progetto-1-alternativo

## Struttura del progetto:
- src/linear_solvers.py - implementazioni Jacobi, Gauss-Seidel, Gradiente, Gradiente Coniugato

- src/matrix_utils.py - caricamento .mtx, generazione sistemi di test, salvataggio risultati

- src/__init__.py - inizializzatore del pacchetto

- main.py -  entry point per lanciare esperimenti

- plots/generate_plots.py - generazione automatica dei grafici dai risultati

- tests/test_quick.py - test rapido di funzionalità end-to-end

- tests/mtx_indexing_test.py - test corretta indicizzazione dei file .mtx

- results/ - output dei test (summary + file per tolleranza)

- test_data/ - dati di input per la modalità generica (b, x*)
  
## Istruzioni per l’esecuzione:

### Esecuzione standard (validazione): 
- **python main.py** 

### Esecuzione su matrice specifica: 

- **python main.py -m spa1.mtx**

### Modalità generica (input forniti dall’utente): 

- **python main.py -m spa1.mtx --b test_data/b_spa1.txt --x test_data/x_spa1.txt --single-tol 1e-6**

### Generazione matrice di test: 

- **python main.py --test -n 50 --size 50**

### Test rapidi: 

- **python tests/test_quick.py** - esegue un test veloce con matrice piccola per verificare convergenza e proprietà.

- **python tests/mtx_indexing_test.py** - verifica che scipy.io.mmread converta correttamente indici da 1-based a 0-based.

### Generazione grafici: 

- **python plots/generate_plots.py** 
