# Requisitos: matplotlib únicamente (ya incluido en este entorno).
import random, time, heapq, math
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Crear la carpeta si no existe
carpeta = "graficos_informe"
if not os.path.exists(carpeta):
    os.makedirs(carpeta)
    
# Utilidad para visualizar (una figura por gráfico)
def plot_grid(grid, path=None, title="Mapa", archivo=None):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(grid, cmap="gray_r")
    if path:
        x, y = zip(*path)
        ax.plot(y, x, linewidth=2, marker="o", markersize=3, label="Ruta")
        ax.scatter([y[0]],[x[0]], s=60, marker="s", label="Inicio")
        ax.scatter([y[-1]],[x[-1]], s=60, marker="s", label="Meta")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    ax.legend(loc="upper right")
    if archivo is None:
        nombre_archivo = title.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace("-", "_") + ".png"
    else:
        nombre_archivo = archivo + ".png" if not archivo.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')) else archivo

    ruta_guardado = os.path.join(carpeta, nombre_archivo)
    plt.savefig(ruta_guardado, bbox_inches='tight')
    plt.close(fig)

def generate_maze(width=41, height=41, seed=7, openings = None):
    if openings is None:
        openings = (width * height) // 200
    random.seed(seed)
    # Asegurar dimensiones impares (corredores de 1 celda, paredes de 1 celda)
    width = width if width % 2 == 1 else width + 1
    height = height if height % 2 == 1 else height + 1
    maze = [[1 for _ in range(width)] for _ in range(height)]  # 1 = muro, 0 = libre

    def neighbors(cx, cy):
        for dx, dy in [(2,0),(-2,0),(0,2),(0,-2)]:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < height-1 and 1 <= ny < width-1:
                yield nx, ny, dx, dy

    stack = [(1,1)]
    maze[1][1] = 0

    while stack:
        x, y = stack[-1]
        unvisited = [(nx,ny,dx,dy) for nx,ny,dx,dy in neighbors(x,y) if maze[nx][ny] == 1]
        if unvisited:
            nx, ny, dx, dy = random.choice(unvisited)
            maze[x + dx//2][y + dy//2] = 0  # derribar pared intermedia
            maze[nx][ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Agregar algunas aperturas aleatorias para permitir múltiples rutas
    for _ in range(int(openings)):
        x = random.randrange(1, height-1)
        y = random.randrange(1, width-1)
        maze[x][y] = 0

    return maze, openings

class Node:
    """
    state: estado actual del nodo en el espacio de busqueda
    parent: Puntero al nodo padre del arbol de busqueda, nos permite reconstruir la ruta.
    g: costo del camino desde el inicio hasta el estado actual
    h: heurística del estado actual, representa el costo estimado de la ruta mas corta desde el estado actual hasta el objetivo.
    f: costo total del camino desde el inicio hasta el estado actual, f = g + w*h
    """

    __slots__ = ("state","parent","g","h","f")
    def __init__(self, state, parent=None, g=0, h=0, w=1.0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + w*h
    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b, manhattan : bool = True):
    if manhattan:
        """ 
        La heuristica no sobreestima el costo real de la ruta, puesto que se basa en 
        que dentro de esta cuadricula el desplazamiento responde a un movimiento que es
        representado en la heuristica manhattan (NSEW).
        """
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    else:
        """
        La heuristica euclidiana subestima el costo real de la ruta, puesto que se basa en 
        que dentro de esta cuadricula el desplazamiento responde a un movimiento que es
        representado en la heuristica euclidiana (diagonal).
        """
        return (math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))

def reconstruct_path(n):
    path = []
    while n:
        path.append(n.state)
        n = n.parent
    return path[::-1]

def awastar(start, goal, grid, max_time=2.0, w_start=2.5, w_end=1.0, w_step=0.5, manhattan : bool = True):
    H = len(grid); W = len(grid[0])
    t0 = time.time()
    best_path = None
    best_len = math.inf
    atLeastOnePath = False
    """
    w es el factor de inflacion de la heuristica, mientras mas grande es el factor, mas
    importa es la velocidad de la solucion, por lo que mientras mas grande es el factor,
    menos tiempo se le dará a la busqueda para encontrar la solucion.
    """
    w = w_start
    """
    max_time es el tiempo maximo que se le dará a la busqueda para encontrar la solucion.
    En caso de que el tiempo se agote, se devolverá la mejor solucion encontrada hasta el momento.
    """
    solutions_found = []
    while w >= w_end - 1e-9 and (time.time() - t0) < max_time:
        """
        al decrementar w, se le da menos importancia a la velocidad de la solucion, por lo que
        se le dará mas tiempo a la busqueda para encontrar la solucion. Esto tiende a la solucion 
        de A estrella clásica, la cual por definición es la solución optima.
        """
        open_list = []
        g_cost = {start: 0}
        h0 = heuristic(start, goal, manhattan)
        heapq.heappush(open_list, Node(start, None, 0, h0, w))
        closed = set()

        while open_list and (time.time() - t0) < max_time:
            current = heapq.heappop(open_list)
            if current.state in closed:
                continue
            closed.add(current.state)

            # Poda basada en la mejor cota superior disponible
            if current.g + current.h >= best_len:
                continue

            if current.state == goal:
                atLeastOnePath = True
                cand = reconstruct_path(current)
                current_time = time.time() - t0
                
                if len(cand) <= best_len:
                    best_len = len(cand)
                    best_path = cand
                    solutions_found.append((current_time, len(cand)))
                break  # Reducimos w y repetimos para refinar
            x, y = current.state
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == 0:
                    ng = current.g + 1
                    ns = (nx, ny)
                    if ng < g_cost.get(ns, math.inf):
                        g_cost[ns] = ng
                        h = heuristic(ns, goal)
                        if ng + h >= best_len:
                            continue
                        heapq.heappush(open_list, Node(ns, current, ng, h, w))
        w -= w_step

    return best_path, atLeastOnePath, solutions_found

def greedy_best_first(start, goal, grid, max_time=2.0, manhattan=True):
    H, W = len(grid), len(grid[0])
    t0 = time.time()
    best_path = None
    best_len = math.inf
    atLeastOnePath = False
    solution = []

    while (time.time() - t0) < max_time:
        open_list = []
        h0 = heuristic(start, goal, manhattan)
        heapq.heappush(open_list, (h0, Node(start, None, 0, h0, 0)))
        closed = set()

        while open_list and (time.time() - t0) < max_time:
            _, current = heapq.heappop(open_list)
            if current.state in closed:
                continue
            closed.add(current.state)

            if current.state == goal:
                atLeastOnePath = True
                cand = reconstruct_path(current)
                current_time = time.time() - t0
                solution.append((current_time, len(cand)))
                break

            x, y = current.state
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == 0:
                    h = heuristic((nx, ny), goal, manhattan)
                    node = Node((nx, ny), current, current.g+1, h, 0)
                    heapq.heappush(open_list, (h, node))

    return best_path, atLeastOnePath, solution

def plot_solutions(solutions_found, name : str):
    times = [t for t, _ in solutions_found]
    lengths = [l for _, l in solutions_found]
    plt.figure(figsize=(10, 5))
    plt.plot(times, lengths, marker='o', linestyle='-', color='b')
    plt.xlabel('Tiempo')
    plt.title(f"Tiempo vs Longitud de la ruta {name}")
    plt.savefig(os.path.join(carpeta, f"Tiempo_vs_Longitud_de_la_ruta_{name}.png"), bbox_inches='tight')
    plt.close()

def run_experimentos_base():
    print("Ejecutando experimentos base...")
    data = pd.DataFrame(columns=["size", "seed", "max_time", "w_start", "w_end", "w_step", "manhattan", "length", "openings"])
    maze, openings = generate_maze(41, 41, seed=7)
    start = (1,1)
    goal = (len(maze)-2, len(maze[0])-2)
    maze[start[0]][start[1]] = 0
    maze[goal[0]][goal[1]] = 0

    path_quick, find_quick, solutions_quick = awastar(start, goal, maze, max_time=0.2, w_start=4.5, w_end=1.0, w_step=0.5)
    plot_grid(maze, path_quick, "AW-A*: 0.2s (rapidamente)", "Base_grid_0.2s")

    path_mid, find_mid, solutions_mid = awastar(start, goal, maze, max_time=1.0, w_start=2.5, w_end=1.0, w_step=0.5)
    plot_grid(maze, path_mid, "AW-A*: 1.0s (normal)", "Base_grid_1.0s")

    path_longer, find_longer, solutions_longer = awastar(start, goal, maze, max_time=3.0, w_start=2.5, w_end=1.0, w_step=0.5)
    plot_grid(maze, path_longer, "AW-A*: 3.0s (lentamente)", "Base_grid_3.0s")

    print("Longitud ruta 0.2s:", len(path_quick) if path_quick else None)
    print("Longitud ruta 1.0s:", len(path_mid) if path_mid else None)
    print("Longitud ruta 3.0s:", len(path_longer) if path_longer else None)

    plot_solutions(solutions_quick, "Base_solutions_0.2s")
    plot_solutions(solutions_mid, "Base_solutions_1.0s")
    plot_solutions(solutions_longer, "Base_solutions_3.0s")

    row_quick = {"size": (41, 41), 
                    "seed": 7, 
                    "max_time": 0.2, 
                    "w_start": 4.5, 
                    "w_end": 1.0, 
                    "w_step": 0.5, 
                    "manhattan": True, 
                    "length": len(path_quick) if path_quick else math.inf, 
                    "found": find_quick, 
                    "solutions": solutions_quick,
                    "openings": openings
                }
    row_mid = {"size": (41, 41), 
                    "seed": 7, 
                    "max_time": 1.0, 
                    "w_start": 2.5, 
                    "w_end": 1.0, 
                    "w_step": 0.5, 
                    "manhattan": False, 
                    "length": len(path_mid) if path_mid else math.inf, 
                    "found": find_mid, 
                    "solutions": solutions_mid,
                    "openings": openings
                }
    row_longer = {"size": (41, 41), 
                    "seed": 7, 
                    "max_time": 3.0, 
                    "w_start": 2.5, 
                    "w_end": 1.0, 
                    "w_step": 0.5, 
                    "manhattan": False, 
                    "length": len(path_longer) if path_longer else math.inf, 
                    "found": find_longer, 
                    "solutions": solutions_longer,
                    "openings": openings
                }

    data = pd.concat([data, pd.DataFrame([row_quick, row_mid, row_longer])], ignore_index=True)
    return data

def experimento_iterativo(
    sizes: list[tuple[int, int]],
    seeds: list[int],
    max_time: list[float],
    w_start: list[float],
    w_step: list[float],
    w_end: float = 1.0,
    manhattan: list[bool] = [True, False]
):
    print("Ejecutando experimentos iterativos...")
    data = pd.DataFrame(columns=["size", "seed", "max_time", "w_start", "w_end", "w_step", "manhattan", "length", "openings"])
    for size in sizes:
        for seed in seeds:
            print(f"Procesando tamaño {size} con semilla {seed}")
            maze, openings = generate_maze(size[0], size[1], seed=seed)
            start = (1,1)
            goal = (len(maze)-2, len(maze[0])-2)
            maze[start[0]][start[1]] = 0
            maze[goal[0]][goal[1]] = 0
            # Bucles anidados para todas las combinaciones
            for startW in w_start:
                for time in max_time:
                    for step in w_step:
                        for manhattan_val in manhattan:
                            print(f"  Ejecutando: w={startW}, t={time}, step={step}, manhattan={manhattan_val}")
                            p, find, solutions = awastar(start, goal, maze, max_time=time, w_start=startW, w_end=w_end, w_step=step, manhattan=manhattan_val)
                            if len(solutions) > 0:
                                plot_solutions(solutions, f"Iterativo_solutions_{time}-{seed}-{startW}-{w_end}-{step}-{manhattan_val}")
                            if p is not None:
                                plot_grid(maze, p, f"AW-A*: {time}s ", archivo = f"Iterativo_grid_{time}-{seed}-{startW}-{w_end}-{step}-{manhattan_val}.png")
                            row = { "size": size, 
                                    "seed": seed, 
                                    "max_time": time, 
                                    "w_start": startW, 
                                    "w_end": w_end, 
                                    "w_step": step, 
                                    "manhattan": manhattan_val, 
                                    "length": len(p) if p else math.inf, 
                                    "found": find, 
                                    "solutions": solutions,
                                    "openings": openings
                                }
                            data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
    return data

def run_experimentos_iterativos():
    SIZES = [(1500,1300),(2000,2000)]
    WHEIGHTS = [2.0,2.5,3.0]
    STEPS = [0.25,0.5]
    SEEDS = [7]
    TIMES = [0.2,1.0,3.0,30.0,60,120]
    data = experimento_iterativo(sizes=SIZES, seeds=SEEDS, max_time=TIMES, w_start=WHEIGHTS, w_step=STEPS)
    return data

def run_experimentos_aperturas():
    print("Ejecutando experimentos aperturas...")
    data = pd.DataFrame(columns=["size", "seed", "max_time", "w_start", "w_end", "w_step", "manhattan", "length", "openings"])
    OPENINGS = [0, 50, 150]
    SIZE = [(1000,1000)]
    SEEDS = [7]
    TIMES = [0.2,1.0,3.0,30.0,60,120]
    STEPS = [0.25,0.5]
    MANHATTAN = [True, False]
    for size in SIZE:
        for seed in SEEDS:
            for opening in OPENINGS:
                maze, _ = generate_maze(size[0], size[1], seed=SEEDS[0], openings=opening)
                start = (1,1)
                goal = (len(maze)-2, len(maze[0])-2)
                maze[start[0]][start[1]] = 0
                maze[goal[0]][goal[1]] = 0
                for time in TIMES:
                    for w_step in STEPS:
                        for manhattan in MANHATTAN:
                            p, find, solutions = greedy_best_first(start = start, goal = goal, grid = maze, max_time = time, manhattan = manhattan)
                            plot_solutions(solutions, f"Openings_Greedy Best First_{time}-{seed}-{w_step}-{manhattan}")
                            plot_grid(maze, p, f"Greedy Best First: {time}s (comparativa)", archivo = f"Openings_Greedy Best First_{time}-{seed}-{w_step}-{manhattan}.png")
                            row = { "size": size, 
                                    "seed": SEEDS[0], 
                                    "max_time": time, 
                                    "w_start": None, 
                                    "w_end": None, 
                                    "w_step": w_step, 
                                    "manhattan": manhattan, 
                                    "length": len(p) if p else math.inf, 
                                    "found": find, 
                                    "solutions": solutions,
                                    "openings": opening, 
                                    "name": "Greedy Best First"
                                }
                            data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
                            # awastar(start, goal, grid, max_time=2.0, w_start=2.5, w_end=1.0, w_step=0.5, manhattan : bool = True)
                            p, find, solutions = awastar(start = start, goal = goal, grid = maze, max_time = time, w_start = 2.5, w_end = 1.0, w_step = w_step, manhattan = manhattan)
                            plot_solutions(solutions, f"Openings_AW-A*_{time}-{seed}-{w_step}-{manhattan}")
                            plot_grid(maze, p, f"AW-A*: {time}s (comparativa)", archivo = f"Openings_AW-A*_{time}-{seed}-{w_step}-{manhattan}.png")
                            row = { "size": size, 
                                    "seed": SEEDS[0], 
                                    "max_time": time, 
                                    "w_start": 2.5, 
                                    "w_end": 1.0, 
                                    "w_step": w_step, 
                                    "manhattan": manhattan, 
                                    "length": len(p) if p else math.inf, 
                                    "found": find, 
                                    "solutions": solutions,
                                    "openings": opening, 
                                    "name": "AW-A*"
                                }
                            data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
                            p, find, solutions = awastar(start = start, goal = goal, grid = maze, max_time = time, w_start = 1.0, w_end = 1.0, w_step = w_step, manhattan = manhattan)
                            plot_solutions(solutions, f"Openings_A*_{time}-{seed}-{w_step}-{manhattan}")
                            plot_grid(maze, p, f"A* - {time}s (comparativa)", archivo = f"Openings_A*_{time}-{seed}-{w_step}-{manhattan}.png")
                            row = { "size": size, 
                                    "seed": SEEDS[0], 
                                    "max_time": time, 
                                    "w_start": 1.0, 
                                    "w_end": 1.0, 
                                    "w_step": w_step, 
                                    "manhattan": manhattan, 
                                    "length": len(p) if p else math.inf, 
                                    "found": find, 
                                    "solutions": solutions,
                                    "openings": opening, 
                                    "name": "A*"
                                }
                            data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
    return data

def run_taller():
    print("Ejecutando experimentos base...")
    data = run_experimentos_base()
    data.to_csv("resultados_base.csv", index=False)
    print("Ejecutando experimentos iterativos...")
    data = run_experimentos_iterativos()
    data.to_csv("resultados_iterativo.csv", index=False)
    print("Ejecutando experimentos aperturas...")
    data = run_experimentos_aperturas()
    data.to_csv("resultados_aperturas.csv", index=False)
    print("Experimentos completados")

if __name__ == "__main__":
    run_taller()