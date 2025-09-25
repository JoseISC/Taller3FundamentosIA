import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import os   
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logging.basicConfig(level=logging.INFO)
logger = create_logger(__name__)

# Crear el directorio ./plots si no existe
if not os.path.exists("plots"):
    os.makedirs("plots")

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

REPETICIONES = 4
N = range(8,21,2)
RESULTADOS_SCHEMA = {
    "algoritmo": [],
    "n": [],
    "tiempo (ms)": [],
    "llamadas": [],
    "backtracks": []
}

def mostrar_tablero(tablero):
    for fila in range(len(tablero)):
        linea = ""
        for col in range(len(tablero)):
            if tablero[fila][col] == 1:
                linea += " Q "
            else:
                linea += " . "
        print(linea)
    print("\n")

def es_valido(tablero, fila, col):
    n = len(tablero)
    for i in range(fila):
        for j in range(n):
            if tablero[i][j] == 1:
                if j == col or abs(j - col) == abs(i - fila):
                    return False
    return True

def backtracking(tablero, fila, recursive_call, backtrack):
    if fila == len(tablero):
        return tablero, recursive_call, backtrack 
    
    for i in range(len(tablero)):
        if es_valido(tablero, fila, i):
            tablero[fila][i] = 1
            resultado, recursive_call, backtrack = backtracking(tablero, fila + 1, recursive_call + 1, backtrack)
            if resultado:
                return resultado, recursive_call, backtrack
            tablero[fila][i] = 0
            backtrack += 1  # Incrementar backtrack cuando se deshace una asignación
    return None, recursive_call, backtrack

def forward_checking(tablero, fila, dominios, recursive_call, backtrack):
    n = len(tablero)

    if fila == n:
        return tablero, recursive_call, backtrack

    for col in dominios[fila][:]:
        tablero[fila][col] = 1
        nuevos = {f: list(dominios[f]) for f in dominios}
        for f in range(fila + 1, n):
            if col in nuevos[f]:
                nuevos[f].remove(col)
            diag1 = col + (f -fila)
            diag2 = col - (f -fila)
            if diag1 in nuevos[f]:
                nuevos[f].remove(diag1)
            if diag2 in nuevos[f]:
                nuevos[f].remove(diag2) 

        if any(len(nuevos[f]) == 0 for f in range(fila + 1, n)):
            tablero[fila][col] = 0
            backtrack += 1  # Incrementar backtrack cuando se detecta un dominio vacío
            continue

        resultado, recursive_call, backtrack = forward_checking(tablero, fila+1, nuevos, recursive_call + 1, backtrack)
        if resultado: 
            return resultado, recursive_call, backtrack

        tablero[fila][col] = 0
        backtrack += 1  # Incrementar backtrack cuando se deshace una asignación

    return None, recursive_call, backtrack

def forward_checking_MRV(tablero, dominios, restantes, recursive_call, backtrack):
    if not restantes:
        return [row[:] for row in tablero], recursive_call, backtrack

    fila = min(restantes, key=lambda x: len(dominios[x]))
        
    for col in dominios[fila][:]:
        tablero[fila][col] = 1
        nuevos = {f: list(dominios[f]) for f in dominios}
        for f in restantes - {fila}:
            if col in nuevos[f]:
                nuevos[f].remove(col)
            diag1 = col + (f -fila)
            diag2 = col - (f -fila)
            if diag1 in nuevos[f]:
                nuevos[f].remove(diag1)
            if diag2 in nuevos[f]:
                nuevos[f].remove(diag2) 

        if any(len(nuevos[f]) == 0 for f in restantes - {fila}):
            tablero[fila][col] = 0
            backtrack += 1  # Incrementar backtrack cuando se detecta un dominio vacío
            continue

        resultado, recursive_call, backtrack = forward_checking_MRV(tablero, nuevos, restantes - {fila}, recursive_call + 1, backtrack)
        if resultado: 
            return resultado, recursive_call, backtrack

        tablero[fila][col] = 0
        backtrack += 1  # Incrementar backtrack cuando se deshace una asignación

    return None, recursive_call, backtrack

def experimento_backtracking(REPETICIONES, N, thread_id=""):
    thread_logger = create_logger(f"backtracking{thread_id}")
    thread_logger.info("Experimento backtracking")
    resultados = RESULTADOS_SCHEMA.copy()
    for n in N:
        thread_logger.info(f"Experimento backtracking para n = {n}")
        tiempos = []
        llamadas = []
        backtracks_lista = []
        for _ in range(REPETICIONES):
            thread_logger.info(f"Repetición {_} para n = {n}")
            tablero = [[0]*n for _ in range(n)]
            t0 = time.time()*1000000
            # Guardamos la referencia original de la función backtracking
            resultado_bt, recursive_call, backtrack = backtracking(tablero, 0, 1, 0)
            t1 = time.time()*1000000
            ejecucion = (t1 - t0) / 1000
            tiempos.append(ejecucion)
            llamadas.append(recursive_call)
            backtracks_lista.append(backtrack)
        resultados["algoritmo"].append("backtracking")
        resultados["n"].append(n)
        resultados["tiempo (ms)"].append(sum(tiempos)/len(tiempos))
        resultados["llamadas"].append(sum(llamadas)/len(llamadas))
        resultados["backtracks"].append(sum(backtracks_lista)/len(backtracks_lista))
    return resultados

def experimento_forward_checking(REPETICIONES, N, thread_id=""):
    thread_logger = create_logger(f"forward_checking{thread_id}")
    thread_logger.info("Experimento forward checking")
    resultados = RESULTADOS_SCHEMA.copy()
    for n in N:
        thread_logger.info(f"Experimento forward checking para n = {n}")
        tiempos = []
        llamadas = []
        backtracks_lista = []
        for _ in range(REPETICIONES):
            thread_logger.info(f"Repetición {_} para n = {n}")
            tablero = [[0]*n for _ in range(n)]
            t0 = time.time()*1000000
            resultado_fch, recursive_call, backtrack = forward_checking(tablero, 0, {i: list(range(n)) for i in range(n)}, 1, 0)
            t1 = time.time()*1000000
            ejecucion = (t1 - t0) / 1000
            tiempos.append(ejecucion)
            llamadas.append(recursive_call)
            backtracks_lista.append(backtrack)
        resultados["algoritmo"].append("forward_checking")
        resultados["n"].append(n)
        resultados["tiempo (ms)"].append(sum(tiempos)/len(tiempos))
        resultados["llamadas"].append(sum(llamadas)/len(llamadas))
        resultados["backtracks"].append(sum(backtracks_lista)/len(backtracks_lista))
    return resultados

def experimento_forward_checking_MRV(REPETICIONES, N, thread_id=""):
    thread_logger = create_logger(f"forward_checking_MRV{thread_id}")
    thread_logger.info("Experimento forward checking MRV")
    resultados = RESULTADOS_SCHEMA.copy()
    for n in N:
        thread_logger.info(f"Experimento forward checking MRV para n = {n}")
        tiempos = []
        llamadas = []
        backtracks_lista = []
        for _ in range(REPETICIONES):
            thread_logger.info(f"Repetición {_} para n = {n}")
            tablero = [[0]*n for _ in range(n)]
            t0 = time.time()*1000000
            resultado_fch_MRV, recursive_call, backtrack = forward_checking_MRV(tablero, {i: list(range(n)) for i in range(n)}, set(range(n)), 1, 0)
            t1 = time.time()*1000000
            ejecucion = (t1 - t0) / 1000
            tiempos.append(ejecucion)
            llamadas.append(recursive_call)
            backtracks_lista.append(backtrack)
        resultados["algoritmo"].append("forward_checking_MRV")
        resultados["n"].append(n)
        resultados["tiempo (ms)"].append(sum(tiempos)/len(tiempos))
        resultados["llamadas"].append(sum(llamadas)/len(llamadas))
        resultados["backtracks"].append(sum(backtracks_lista)/len(backtracks_lista))
    return resultados

def ejecutar_experimentos_paralelos(REPETICIONES, N):
    """
    Ejecuta los tres experimentos en paralelo usando ThreadPoolExecutor
    """
    logger.info("Iniciando ejecución paralela de experimentos")
    inicio_total = time.time()
    
    # Crear un ThreadPoolExecutor con 3 hilos (uno para cada experimento)
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Enviar las tareas a los hilos
        future_backtracking = executor.submit(experimento_backtracking, REPETICIONES, N, "_thread1")
        future_forward_checking = executor.submit(experimento_forward_checking, REPETICIONES, N, "_thread2")
        future_forward_checking_MRV = executor.submit(experimento_forward_checking_MRV, REPETICIONES, N, "_thread3")
        
        # Esperar a que terminen todos los experimentos y obtener los resultados
        logger.info("Esperando resultados de todos los experimentos...")
        resultados_backtracking = future_backtracking.result()
        resultados_forward_checking = future_forward_checking.result()
        resultados_forward_checking_MRV = future_forward_checking_MRV.result()
    
    fin_total = time.time()
    tiempo_total = fin_total - inicio_total
    logger.info(f"Todos los experimentos completados en {tiempo_total:.2f} segundos")
    
    return resultados_backtracking, resultados_forward_checking, resultados_forward_checking_MRV

if __name__ == "__main__":
    
    # Ejecutar experimentos en paralelo
    resultados_backtracking, resultados_forward_checking, resultados_forward_checking_MRV = ejecutar_experimentos_paralelos(REPETICIONES, N)

    temp_df0 = pd.DataFrame(resultados_forward_checking_MRV)
    temp_df1 = pd.DataFrame(resultados_forward_checking)
    temp_df2 = pd.DataFrame(resultados_backtracking)
    dfResultados = pd.concat([temp_df0, temp_df1, temp_df2])

    marcadores = {'forward_checking_MRV': 'o', 'forward_checking': 's', 'backtracking': 'D'}
    colores = {'forward_checking_MRV': '#1f77b4', 'forward_checking': '#ff7f0e', 'backtracking': '#2ca02c'}

    # Gráfico 1: Tiempo vs n
    plt.figure(figsize=(10,6))
    for algoritmo, df_alg in dfResultados.groupby("algoritmo"):
        sns.lineplot(
            data=df_alg,
            x="n",
            y="tiempo (ms)",
            label=algoritmo.replace("_", " ").capitalize(),
            marker=marcadores.get(algoritmo, 'o'),
            color=colores.get(algoritmo, None),
            linewidth=2.5,
            markersize=9
        )
    plt.xlabel("n", fontsize=13, fontweight='bold')
    plt.ylabel("Tiempo (ms)", fontsize=13, fontweight='bold')
    plt.title("Evolución del tiempo según algoritmo", fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Algoritmo", fontsize=11, title_fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig("plots/tiempo_vs_n.png")
    plt.close()

    # Gráfico 2: Llamadas vs n (estandarizado)
    plt.figure(figsize=(10,6))
    for algoritmo, df_alg in dfResultados.groupby("algoritmo"):
        sns.lineplot(
            data=df_alg,
            x="n",
            y="llamadas",
            label=algoritmo.replace("_", " ").capitalize(),
            marker=marcadores.get(algoritmo, 'o'),
            color=colores.get(algoritmo, None),
            linewidth=2.5,
            markersize=9
        )
    plt.xlabel("n", fontsize=13, fontweight='bold')
    plt.ylabel("Llamadas recursivas", fontsize=13, fontweight='bold')
    plt.title("Evolución de llamadas recursivas según algoritmo", fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Algoritmo", fontsize=11, title_fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig("plots/llamadas_vs_n.png")
    plt.close()

    # Gráfico 3: Backtracks vs n (estandarizado)
    plt.figure(figsize=(10,6))
    for algoritmo, df_alg in dfResultados.groupby("algoritmo"):
        sns.lineplot(
            data=df_alg,
            x="n",
            y="backtracks",
            label=algoritmo.replace("_", " ").capitalize(),
            marker=marcadores.get(algoritmo, 'o'),
            color=colores.get(algoritmo, None),
            linewidth=2.5,
            markersize=9
        )
    plt.xlabel("n", fontsize=13, fontweight='bold')
    plt.ylabel("Backtracks", fontsize=13, fontweight='bold')
    plt.title("Evolución de backtracks según algoritmo", fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Algoritmo", fontsize=11, title_fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig("plots/backtracks_vs_n.png")
    plt.close()

    dfResultados.to_csv("resultados.csv", index=False)
