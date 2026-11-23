
"""
Proyecto Final: Hiper-heuristica de Seleccion para TSP
Generacion de etiquetas usando los algoritmos de Tarea 2.

Ejecutar desde la raiz:
python src/generate_labels.py
"""

import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INSTANCES_DIR = ROOT / "data" / "instances"
OUTPUT_DIR = ROOT / "results" / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# UTILIDADES TSP
# --------------------------------------------------

def dist_matrix(points):
    n = len(points)
    D = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            D[i][j] = math.hypot(
                points[i][0] - points[j][0],
                points[i][1] - points[j][1]
            )

    return D


def route_cost(route, D):
    n = len(route)

    return sum(
        D[route[i]][route[(i + 1) % n]]
        for i in range(n)
    )


def nearest_neighbour(D, start):
    n = len(D)

    route = [start]
    remaining = set(range(n))
    remaining.remove(start)

    while remaining:
        nxt = min(
            remaining,
            key=lambda j: D[route[-1]][j]
        )

        route.append(nxt)
        remaining.remove(nxt)

    return route


def random_route(n, rng):
    route = list(range(n))
    rng.shuffle(route)
    return route


# --------------------------------------------------
# ALGORITMO 1: 2-OPT HILL CLIMBING
# --------------------------------------------------

def two_opt(route, D, first_improvement=True):

    route = list(route)

    while True:

        best_route = route
        best_cost = route_cost(route, D)
        found = False

        n = len(route)

        for i in range(1, n - 1):

            for k in range(i + 1, n):

                if i == 1 and k == n - 1:
                    continue

                candidate = (
                    route[:i]
                    + route[i:k+1][::-1]
                    + route[k+1:]
                )

                cost = route_cost(candidate, D)

                if cost < best_cost - 1e-10:

                    best_route = candidate
                    best_cost = cost
                    found = True

                    if first_improvement:
                        break

            if found and first_improvement:
                break

        if not found:
            break

        route = best_route

    return route, route_cost(route, D)


def two_opt_hill_climbing(D, seed):

    rng = random.Random(seed)

    start = rng.randrange(len(D))

    initial = nearest_neighbour(D, start)

    route, cost = two_opt(initial, D)

    return route, cost


# --------------------------------------------------
# ALGORITMO 2: RECOCIDO SIMULADO + 2-OPT
# --------------------------------------------------

def simulated_annealing(
    D,
    iterations=2500,
    seed=1,
    t0=1.0,
    alpha=0.995
):

    rng = random.Random(seed)

    n = len(D)

    current = random_route(n, rng)
    current_cost = route_cost(current, D)

    best = current[:]
    best_cost = current_cost

    for step in range(iterations):

        T = t0 * (alpha ** step)

        i = rng.randrange(1, n - 1)
        k = rng.randrange(i + 1, n)

        candidate = (
            current[:i]
            + current[i:k+1][::-1]
            + current[k+1:]
        )

        cand_cost = route_cost(candidate, D)

        delta = cand_cost - current_cost

        if delta < 0 or rng.random() < math.exp(
            -delta / max(T, 1e-12)
        ):

            current = candidate
            current_cost = cand_cost

        if current_cost < best_cost:

            best = current[:]
            best_cost = current_cost

    # Refinamiento final mediante 2-opt
    polished, polished_cost = two_opt(best, D)

    return polished, polished_cost


# --------------------------------------------------
# ALGORITMO 3: MST + 2-OPT
# --------------------------------------------------

def mst_prim(D):

    n = len(D)

    in_tree = {0}
    parent = {}

    while len(in_tree) < n:

        best = (float("inf"), None, None)

        for u in in_tree:

            for v in range(n):

                if v not in in_tree and D[u][v] < best[0]:

                    best = (D[u][v], u, v)

        _, u, v = best

        parent.setdefault(u, []).append(v)

        in_tree.add(v)

    return parent


def preorder_mst(tree, root=0):

    result = []

    def visit(u):

        result.append(u)

        for v in tree.get(u, []):
            visit(v)

    visit(root)

    return result


def mst_two_opt(D):

    tree = mst_prim(D)

    initial = preorder_mst(tree)

    route, cost = two_opt(initial, D)

    return route, cost


# --------------------------------------------------
# CARGAR INSTANCIAS
# --------------------------------------------------

def load_instance(instance_id):
    # Obtener la carpeta raiz del proyecto
    project_dir = Path(__file__).resolve().parent

    # Construir la ruta de la instancia
    path = project_dir / "data" / "instances" / f"{instance_id}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro la instancia: {path}"
        )

    points = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            points.append((x, y))

    return points


# --------------------------------------------------
# EJECUTAR LOS TRES ALGORITMOS
# --------------------------------------------------

def evaluate_instance(row):

    instance_id = row["instance_id"]

    points = load_instance(instance_id)

    D = dist_matrix(points)

    # Semilla reproducible derivada del ID
    numeric_id = int(instance_id.split("_")[1])

    _, cost_2opt = two_opt_hill_climbing(
        D,
        seed=20000 + numeric_id
    )

    _, cost_sa = simulated_annealing(
        D,
        iterations=2500,
        seed=10000 + numeric_id,
        t0=1.0,
        alpha=0.995
    )

    _, cost_mst = mst_two_opt(D)

    costs = {
        "2opt_hill_climbing": cost_2opt,
        "sa_2opt": cost_sa,
        "mst_2opt": cost_mst
    }

    best_algorithm = min(costs, key=costs.get)

    result = dict(row)

    result["cost_2opt_hill_climbing"] = cost_2opt
    result["cost_sa_2opt"] = cost_sa
    result["cost_mst_2opt"] = cost_mst
    result["best_algorithm"] = best_algorithm

    return result


# --------------------------------------------------
# PROCESAR TRAIN Y TEST
# --------------------------------------------------


def process_csv(filename):

    # Obtener la carpeta raiz del proyecto
    project_dir = Path(__file__).resolve().parent

    # Construir la ruta del CSV de entrada
    input_path = project_dir / "data" / filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {input_path}"
        )

    with open(input_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"El archivo {filename} esta vacio.")

    results = []

    total = len(rows)

    print(f"\nProcesando {filename}: {total} instancias")

    for index, row in enumerate(rows, start=1):

        result = evaluate_instance(row)

        results.append(result)

        print(
            f"[{index}/{total}] {row['instance_id']} "
            f"-> {result['best_algorithm']}"
        )

    fieldnames = list(results[0].keys())

    # Guardar en results/data
    output_dir = project_dir / "results" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = filename.replace(".csv", "_labeled.csv")
    output_path = output_dir / output_name

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nArchivo generado: {output_path}")

    return results



# --------------------------------------------------
# RESUMEN DE RESULTADOS
# --------------------------------------------------

def main():

    print("=" * 55)
    print("HIPER-HEURISTICA DE SELECCION PARA TSP")
    print("GENERACION DE ETIQUETAS")
    print("=" * 55)

    train = process_csv("train.csv")
    test = process_csv("test.csv")

    all_results = train + test

    algorithms = [
        "2opt_hill_climbing",
        "sa_2opt",
        "mst_2opt"
    ]

    print("\n========== BALANCE DE CLASES ==========")

    for algorithm in algorithms:

        count = sum(
            r["best_algorithm"] == algorithm
            for r in all_results
        )

        percentage = count / len(all_results) * 100

        print(
            f"{algorithm}: {count} "
            f"({percentage:.2f}%)"
        )

    print("\n========== PARAMETROS SA ==========")
    print("Esquema: exponencial")
    print("Temperatura inicial: 1.0")
    print("Alpha: 0.995")
    print("Iteraciones: 2500")
    print("Refinamiento final: 2-opt")

    print("\nProceso terminado.")


if __name__ == "__main__":
    main()
