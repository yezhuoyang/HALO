#Optimize the data qubit mapping of a given hardware
#First, calculate the cost of a mapping
from collections import deque
from typing import Dict, List
import numpy as np
from qiskit.transpiler import CouplingMap
import matplotlib.pyplot as plt


import random
import math



class circuit_topology:


    def __init__(self, data_qubit_number: int, helper_qubit_number: int ,data_interaction: list[tuple[int, int]], data_helper_interaction: list[tuple[int, int]]):
        self._data_qubit_number = data_qubit_number
        self._helper_qubit_number = helper_qubit_number
        self._data_interaction = data_interaction
        self._data_helper_interaction = data_helper_interaction
        self._data_data_weight={x:{y:0 for y in range(data_qubit_number)} for x in range(data_qubit_number)}
        self._data_helper_weight={x:0 for x in range(data_qubit_number)}
        for a,b in data_interaction:
            self._data_data_weight[a][b]+=1
            self._data_data_weight[b][a]+=1
        for a,b in data_helper_interaction:
            self._data_helper_weight[a]+=1


    def get_data_data_weight(self, data_qubit_a: int, data_qubit_b: int) -> int:
        return self._data_data_weight[data_qubit_a][data_qubit_b]
    
    def get_data_helper_weight(self, data_qubit: int) -> int:
        return self._data_helper_weight[data_qubit]




class process:


    def __init__(self, process_id: int, num_data_qubits: int, num_helper_qubits: int, topology: circuit_topology):
        self._process_id = process_id
        self._num_data_qubits = num_data_qubits
        self._num_helper_qubits = num_helper_qubits
        self._topology = topology

    def get_process_id(self) -> int:
        return self._process_id
    
    def get_num_data_qubits(self) -> int:
        return self._num_data_qubits

    def get_num_helper_qubits(self) -> int:
        return self._num_helper_qubits

    def get_topology(self) -> circuit_topology:
        return self._topology
    

    def intro_costs(self,mapping:Dict[int, int],distance:list[list[int]])->float:
        """
        Calculate the intro cost of this process based on the given mapping and distance matrix.
        """
        cost = 0.0
        for i in range(self._num_data_qubits):
            for j in range(i+1,self._num_data_qubits):
                    cost += distance[mapping[i]][mapping[j]] * self._topology.get_data_data_weight(i, j)
        return cost


def torino_coupling_map():
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], # The first long row
        [0,15], [15,19], [4,16], [16,23], [8,17], [17,27], [12,18], [18,31], # Short row 1
        [19,20], [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], # The second long row
        [21,34], [34,40], [25,35], [35,44], [29,36], [36,48], [33,37], [37,52], # Short row 2
        [38,39], [39,40], [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], # The third long row
        [38,53], [53,57], [42,54], [54,61], [46,55], [55,65], [50,56], [56,69], # Short row 3
        [57,58], [58,59], [59,60], [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], # The forth long row
        [59,72], [72,78], [63,73], [73,82], [67,74], [74,86], [71,75], [75,90], # Short row 4
        [76,77], [77,78], [78,79], [79,80], [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], # The fifth long row
        [76,91], [91,95], [80,92], [92,99], [84,93], [93,103], [88,94], [94,107], # Short row 5
        [95,96], [96,97], [97,98], [98,99], [99,100], [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], # The sixth long row
        [97,110], [110,116], [101,111], [111,120], [105,112], [112,124],[109,113], [113,128], # Short row 6
        [114,115], [115,116], [116,117], [117,118], [118,119], [119,120], [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], # The seventh long row
        [114,129], [118, 130], [122,131], [126,132]  # Short row 7
    ]
    return COUPLING


def simple_10_qubit_coupling_map():
    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain
    return COUPLING

def get_10_qubit_hardware_coords() -> list[tuple[float, float]]:
    edge_length = 1
    coords = [ ]
    for i in range(10):
        if i<5:
            coords.append( (float(i*edge_length), 0.0) )
        else:
            coords.append( (float((i-5)*edge_length), -edge_length))
    return coords


def plot_process_schedule_on_10_qubit_hardware(coupling_edges: list[list[int]],
                               process_list: list[process],
                               mapping: Dict[int, tuple[int, int]],
                               out_png: str = "hardware_mapping_torino.png",
                               figsize=(12, 4.5)):
    coords = get_10_qubit_hardware_coords()
    cm = CouplingMap(coupling_edges)

    # undirected edges for a clean look
    pairs = cm.get_edges()
    undirected = sorted(set(tuple(sorted((a, b))) for a, b in pairs))

    fig, ax = plt.subplots(figsize=figsize)

    # edges
    for a, b in undirected:
        xa, ya = coords[a]; xb, yb = coords[b]
        ax.plot([xa, xb], [ya, yb], linewidth=1.5, alpha=0.7, color="#20324d")

    # nodes
    xs = [xy[0] for xy in coords]; ys = [xy[1] for xy in coords]
    ax.scatter(xs, ys, s=620, color="#0b1e3f", zorder=3)

    # indices
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white",
                zorder=4, clip_on=False)  # avoid text clipping


    # --- give each process a unique color ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(process_list)))  # up to 10 distinct
    for phys, (pid, data_qubit) in mapping.items():
        # find the process
        color = colors[pid]
        x, y = coords[phys]
        ax.scatter([x], [y], s=780, facecolors="none", edgecolors=color,
                   linewidths=2.6, zorder=5)
        ax.text(x, y + 0.15, f"P{pid}-D{data_qubit}" , ha="center", va="bottom",
                fontsize=6, color=color, weight="bold", zorder=6, clip_on=False)


    ax.set_aspect("equal", adjustable="datalim")

    # ---- Key fix: add padding around data limits ----
    pad = 0.75                     # increase if labels still feel tight
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.axis("off")

    # Avoid overly tight cropping; keep a little page margin
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def torino_qubit_coords() -> list[tuple[float, float]]:
    coords = [(0.0, 0.0)] * 133

    # Long rows: each has 16 nodes, at x=0..15
    long_starts = [0, 19, 38, 57, 76, 95, 114]
    for r, start in enumerate(long_starts):
        y = -2.0 * r
        for k in range(15):
            coords[start + k] = (float(k), y)

    # Short rows: each has 4 nodes, alternating column anchors
    short_starts = [15, 34, 53, 72, 91, 110, 129]

    anchors_odd  = [0, 4, 8, 12]  # short rows 1,3,5,7
    anchors_even = [2, 6, 10, 14]   # short rows 2,4,6
    for s, start in enumerate(short_starts):
        y = -(2.0 * s + 1.0)
        xs = anchors_odd if (s % 2 == 0) else anchors_even
        for j, x in enumerate(xs):
            coords[start + j] = (float(x), y)

    return coords



def plot_process_schedule_on_torino(coupling_edges: list[list[int]],
                               process_list: list[process],
                               mapping: Dict[int, tuple[int, int]],
                               out_png: str = "hardware_mapping_torino.png",
                               figsize=(11, 9)):
    """
    Plot the data qubit layout and mapping of multiple processes on the torino hardware.
    """
    coords = torino_qubit_coords()
    cm = CouplingMap(coupling_edges)

    # undirected edges for a clean look
    pairs = cm.get_edges()
    undirected = sorted(set(tuple(sorted((a, b))) for a, b in pairs))

    fig, ax = plt.subplots(figsize=figsize)

    # edges
    for a, b in undirected:
        xa, ya = coords[a]; xb, yb = coords[b]
        ax.plot([xa, xb], [ya, yb], linewidth=1.5, alpha=0.7, color="#20324d")

    # nodes
    xs = [xy[0] for xy in coords]; ys = [xy[1] for xy in coords]
    ax.scatter(xs, ys, s=620, color="#0b1e3f", zorder=3)

    # small index inside each node (physical index)
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white", zorder=4)


    # --- give each process a unique color ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(process_list)))  # up to 10 distinct
    for phys, (pid, data_qubit) in mapping.items():
        # find the process
        color = colors[pid]
        x, y = coords[phys]
        ax.scatter([x], [y], s=780, facecolors="none", edgecolors=color,
                   linewidths=2.6, zorder=5)
        ax.text(x, y + 0.38, f"P{pid}-D{data_qubit}" , ha="center", va="bottom",
                fontsize=5, color=color, weight="bold", zorder=6)


    ax.set_aspect("equal"); ax.axis("off"); plt.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def all_pairs_distances(n, edges) -> list[list[int]]:
    """
    n: number of vertices labeled 0..n-1
    edges: list of [u, v] pairs (undirected)
    returns: n x n list of ints, distances; -1 means unreachable
    """
    # build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge ({u},{v}) out of range for n={n}")
        adj[u].append(v)
        adj[v].append(u)

    # BFS from every source
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dq = deque([s])
        dist[s][s] = 0
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if dist[s][w] == -1:
                    dist[s][w] = dist[s][u] + 1
                    dq.append(w)
    return dist


alpha=0.3
beta=100
gamma=300
delta=200
N_qubits=133
hardware_distance_pair=all_pairs_distances(N_qubits, torino_coupling_map())
# N_qubits=10
# hardware_distance_pair=all_pairs_distances(N_qubits, simple_10_qubit_coupling_map())

def calculate_mapping_cost(process_list: List[process],mapping: Dict[int, tuple[int, int]]) -> float:
    """
    Input:
        mapping: Dict[int, tuple[int, int]]
            A dictionary where the key is physical, and the value is a tuple (process_id, data_qubit_index)
        
        Output:
            A float representing the total mapping cost, calculated as:
            cost = alpha * intro_cost + beta * inter_cost + gamma * helper_cost
    """

    # Initialize the helper qubit Zone
    is_helper_qubit={phys:True for phys in range(N_qubits)}
    for phys in mapping.keys():
        is_helper_qubit[phys]=False
    helper_qubit_list=[phys for phys in range(N_qubits) if is_helper_qubit[phys]]
    #print("Helper qubit list:", helper_qubit_list)

    #Convert the mapping L to the mapping per process
    proc_mapping={pid:{} for pid in [proc.get_process_id() for proc in process_list]}
    for phys,(pid,data_qubit) in mapping.items():
        proc_mapping[pid][data_qubit]=phys


    intro_cost=0.0
    #First step, calculate the intra cost of all process
    for proc in process_list:
        intro_cost+=proc.intro_costs(proc_mapping[proc.get_process_id()],hardware_distance_pair)


    #print("Intro cost:", intro_cost)


    compact_cost=0.0
    #Second step, calculate the compact cost of all process
    #This is defined as the distance between the furthest data qubits of a process
    for proc in process_list:
        max_distance=0.0
        for data_qubit_i in range(proc.get_num_data_qubits()):
            phys_i=proc_mapping[proc.get_process_id()][data_qubit_i]
            for data_qubit_j in range(data_qubit_i+1,proc.get_num_data_qubits()):
                phys_j=proc_mapping[proc.get_process_id()][data_qubit_j]
                max_distance=max(max_distance,hardware_distance_pair[phys_i][phys_j])
        compact_cost+=max_distance



    inter_cost=0.0
    #First step, calculate the inter cost across all processes
    #This is done by calculatin the average distance between all mapped data qubits of two processes 
    for i in range(len(process_list)):
        for j in range(i+1,len(process_list)):
            proc_i=process_list[i]
            proc_j=process_list[j]
            pid_i=proc_i.get_process_id()
            pid_j=proc_j.get_process_id()
            total_distance=0.0
            count=0
            for data_qubit_i in range(proc_i.get_num_data_qubits()):
                phys_i=proc_mapping[pid_i][data_qubit_i]
                for data_qubit_j in range(proc_j.get_num_data_qubits()):
                    phys_j=proc_mapping[pid_j][data_qubit_j]
                    total_distance+=hardware_distance_pair[phys_i][phys_j]
                    count+=1
            if count>0:
                inter_cost+=total_distance/count


    #print("Inter cost:", inter_cost)


    helper_cost=0.0
    #Last step, calculate the helper cost of all process
    #This is done by calculating the weighted distance from data qubits to unmapped helper qubit Zone
    for proc in process_list:
        for data_qubit in range(proc.get_num_data_qubits()):
            phys=proc_mapping[proc.get_process_id()][data_qubit]
            helper_weight=proc.get_topology().get_data_helper_weight(data_qubit)
            if helper_weight==0:
                continue
            min_helper_distance=10000
            for helper_qubit in helper_qubit_list:
                min_helper_distance=min(min_helper_distance,hardware_distance_pair[phys][helper_qubit])
            helper_cost+=min_helper_distance*helper_weight

    #print("Helper cost:", helper_cost)

    return alpha * intro_cost - beta * inter_cost + gamma * helper_cost + delta * compact_cost







def random_initial_mapping(process_list: List[process],
                           n_qubits: int) -> Dict[int, tuple[int, int]]:
    """
    Randomly assign all data qubits of all processes to distinct physical qubits.
    Remaining physical qubits are helpers (implicitly).
    """
    total_data_qubits = sum(p.get_num_data_qubits() for p in process_list)
    if total_data_qubits > n_qubits:
        raise ValueError("Not enough physical qubits for all data qubits")

    phys_indices = list(range(n_qubits))
    random.shuffle(phys_indices)
    used_phys = phys_indices[:total_data_qubits]

    mapping = {}
    k = 0
    for proc in process_list:
        pid = proc.get_process_id()
        for dq in range(proc.get_num_data_qubits()):
            mapping[used_phys[k]] = (pid, dq)
            k += 1
    return mapping



def greedy_initial_mapping(process_list: List[process],
                           n_qubits: int,
                           distance: List[List[int]]) -> Dict[int, tuple[int, int]]:
    """
    Greedy placement:
    - Place the very first data qubit of the first process on phys 0.
    - For every next data qubit across all processes:
         choose the unused physical qubit
         that is closest to ANY already-used physical qubit.
    Produces a compact cluster-like initial layout.
    """
    total_data_qubits = sum(p.get_num_data_qubits() for p in process_list)
    if total_data_qubits > n_qubits:
        raise ValueError("Not enough physical qubits for all data qubits")

    mapping: Dict[int, tuple[int, int]] = {}

    # --- Step 1: place the very first data qubit onto physical qubit 0 ---
    mapping[0] = (process_list[0].get_process_id(), 0)

    used_phys = {0}
    remaining_phys = set(range(n_qubits)) - used_phys

    # Iterator for (pid, data_qubit)
    placement_list = []
    for proc in process_list:
        pid = proc.get_process_id()
        for dq in range(proc.get_num_data_qubits()):
            placement_list.append((pid, dq))

    # We already placed first one, so skip it
    placement_list = placement_list[1:]

    # --- Step 2: greedy expansion ---
    for pid, dq in placement_list:
        best_phys = None
        best_score = float("inf")

        for phys in remaining_phys:
            # distance to the closest used phys (cluster expansion)
            dist_to_cluster = min(distance[phys][u] for u in used_phys)
            if dist_to_cluster < best_score:
                best_score = dist_to_cluster
                best_phys = phys

        # Assign the chosen physical location
        mapping[best_phys] = (pid, dq)

        # Update sets
        used_phys.add(best_phys)
        remaining_phys.remove(best_phys)

    return mapping

def propose_neighbor(mapping: Dict[int, tuple[int, int]],
                     n_qubits: int,
                     move_prob: float = 0.3) -> Dict[int, tuple[int, int]]:
    """
    Given a mapping, return a new mapping by either:
    - Swapping two mapped physical qubits, or
    - Moving a data qubit to a helper location.
    """
    new_mapping = dict(mapping)  # shallow copy is enough

    used_phys = list(new_mapping.keys())
    all_phys = list(range(n_qubits))
    helper_phys = [p for p in all_phys if p not in used_phys]

    # If we have no helper qubits, we can only swap.
    if not helper_phys or random.random() > move_prob:
        # swap two mapped physical locations
        if len(used_phys) < 2:
            return new_mapping
        a, b = random.sample(used_phys, 2)
        new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
    else:
        # move one data qubit to a helper physical qubit
        a = random.choice(used_phys)
        h = random.choice(helper_phys)
        new_mapping[h] = new_mapping[a]
        del new_mapping[a]

    return new_mapping



def iteratively_find_the_best_mapping(process_list: List[process],
                                      n_qubits: int,
                                      n_restarts: int = 300,
                                      steps_per_restart: int = 2000
                                      ) -> Dict[int, tuple[int, int]]:
    """
    Heuristic search for a good mapping using simulated annealing
    with multiple random restarts.

    Returns the best mapping found.
    """
    global_best_mapping = None
    global_best_cost = float("inf")

    for r in range(n_restarts):
        # 1) random initial mapping
        # current_mapping = random_initial_mapping(process_list, n_qubits)
        current_mapping = greedy_initial_mapping(process_list, n_qubits,hardware_distance_pair)
        current_cost = calculate_mapping_cost(process_list, current_mapping)

        # temperature schedule (very simple linear cooling)
        # scale the initial T with magnitude of the cost to get something reasonable
        T0 = max(1.0, abs(current_cost) * 0.1)

        for step in range(steps_per_restart):
            # temperature decreases over time
            t = step / max(1, steps_per_restart - 1)
            T = T0 * (1.0 - t) + 1e-3  # from T0 -> ~0

            # 2) propose a neighbor and compute its cost
            candidate_mapping = propose_neighbor(current_mapping, n_qubits)
            candidate_cost = calculate_mapping_cost(process_list, candidate_mapping)

            delta = candidate_cost - current_cost

            # 3) acceptance rule (simulated annealing)
            if delta < 0 or math.exp(-delta / T) > random.random():
                current_mapping = candidate_mapping
                current_cost = candidate_cost

                # track global best
                if current_cost < global_best_cost:
                    global_best_cost = current_cost
                    global_best_mapping = current_mapping

        print(f"[Restart {r}] best so far: {global_best_cost}")

    print("Final best cost:", global_best_cost)
    return global_best_mapping



def make_process_topology(num_data: int = 20,
                          num_helper: int = 10,
                          extra_edges_per_qubit: int = 2,
                          seed: int | None = None) -> circuit_topology:
    """
    Create a toy topology with:
      - ring + a few random long-range data-data interactions
      - each data qubit connected to 2 helper qubits (for weight)
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    data_interaction: list[tuple[int, int]] = []

    # Ring: i -- i+1 (mod num_data)
    for i in range(num_data):
        j = (i + 1) % num_data
        data_interaction.append((i, j))

    # Add some random long-range edges
    for i in range(num_data):
        for _ in range(extra_edges_per_qubit):
            j = rng.randrange(num_data)
            if j != i:
                a, b = sorted((i, j))
                data_interaction.append((a, b))

    # Data-helper interactions: each data qubit talks to 2 helpers
    data_helper_interaction: list[tuple[int, int]] = []
    for i in range(num_data):
        if rng.random() < 0.2:
            h1 = i % num_helper
            data_helper_interaction.append((i, h1))


    return circuit_topology(
        data_qubit_number=num_data,
        helper_qubit_number=num_helper,
        data_interaction=data_interaction,
        data_helper_interaction=data_helper_interaction,
    )




if __name__ == "__main__":
    random.seed(42)  # for reproducibility

    # ----- build 5 processes, each with 20 data and 10 helper qubits -----
    process_list: list[process] = []
    NUM_PROCS = 4
    NUM_DATA = 15
    NUM_HELPERS = 1

    for pid in range(NUM_PROCS):
        topo = make_process_topology(
            num_data=NUM_DATA,
            num_helper=NUM_HELPERS,
            extra_edges_per_qubit=30,
            seed=100 + pid,   # different but reproducible per process
        )
        proc = process(
            process_id=pid,
            num_data_qubits=NUM_DATA,
            num_helper_qubits=NUM_HELPERS,
            topology=topo,
        )
        process_list.append(proc)

    # Sanity check: total data qubits must fit into hardware
    total_data_qubits = sum(p.get_num_data_qubits() for p in process_list)
    print("Total data qubits:", total_data_qubits, "Hardware qubits:", N_qubits)

    # ----- run the heuristic search on Torino -----
    best_mapping = iteratively_find_the_best_mapping(
        process_list,
        n_qubits=N_qubits,
    )

    print("Best mapping found:", best_mapping)
    best_cost = calculate_mapping_cost(process_list, best_mapping)
    print("Best cost:", best_cost)

    # ----- visualize on Torino -----
    plot_process_schedule_on_torino(
        torino_coupling_map(),
        process_list,
        best_mapping,
        out_png="best_torino_mapping_5proc_20data.png",
    )

    # plot_process_schedule_on_10_qubit_hardware(
    #     simple_10_qubit_coupling_map(),
    #     [process1, process2],
    #     best_mapping,
    #     out_png="best_10_qubit_mapping.png",
    # )

# if __name__ == "__main__":


#     data_interaction_1=[[0, 1], [0, 3], [1, 3],[0,2]]
#     data_helper_interaction_1=[[3,0],[3,1],[2,0],[2,1]]
#     circuit_topology1 = circuit_topology(data_qubit_number=4,
#                                          helper_qubit_number=2,
#                                          data_interaction=data_interaction_1,
#                                          data_helper_interaction=data_helper_interaction_1)
    
#     process1 = process(process_id=0,
#                        num_data_qubits=4,
#                        num_helper_qubits=2,
#                        topology=circuit_topology1)



#     data_interaction_2=[[0, 1], [0, 3], [1, 3],[0,2]]
#     data_helper_interaction_2=[[3,0],[3,1],[2,0],[2,1]]
#     circuit_topology2 = circuit_topology(data_qubit_number=4,
#                                          helper_qubit_number=2,
#                                          data_interaction=data_interaction_2,
#                                          data_helper_interaction=data_helper_interaction_2)

#     process2 = process(process_id=1,
#                        num_data_qubits=4,
#                        num_helper_qubits=2,
#                        topology=circuit_topology2)
    

#     mapping_example={0:(0,0),1:(0,2),5:(0,1),6:(0,3),
#                      2:(1,3),7:(1,2),3:(1,1),8:(1,0)}


#     # plot_process_schedule_on_torino(torino_coupling_map(),
#     #                                [process1, process2],
#     #                                mapping_example)
    

#     plot_process_schedule_on_10_qubit_hardware(simple_10_qubit_coupling_map(),
#                                    [process1, process2],
#                                    mapping_example,
#                                    out_png="example_10_qubit_mapping.png")


#     cost=calculate_mapping_cost([process1,process2],mapping_example)    

#     print(f"Example mapping cost: {cost}")