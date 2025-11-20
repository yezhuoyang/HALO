#Optimize the data qubit mapping of a given hardware
#First, calculate the cost of a mapping
from collections import deque
import copy
from typing import Dict, List
import numpy as np
from qiskit.transpiler import CouplingMap
import matplotlib.pyplot as plt
import random
import math

from enum import Enum
from instruction import instruction, parse_program_from_file
from hardwares import torino_coupling_map, simple_10_qubit_coupling_map, get_10_qubit_hardware_coords, torino_qubit_coords


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



    def best_helper_qubit_location(self, hardware_distance: List[List[int]], data_qubit_mapping: Dict[int, int], available_helper_qubits: List[int], helper_qubit_index: int) -> int:
        """
        Given the data qubit mapping, get the best physical location for the helper qubit with index helper_qubit_index.
        Inputs:
        - hardware_distance: distance matrix of the hardware
        - data_qubit_mapping: mapping from data qubit index to physical qubit index
        - available_helper_qubits: list of available physical qubits for helper qubits
        - helper_qubit_index: index of the helper qubit in the process
        Returns:
        - best_location: physical qubit index for the helper qubit
        """
        best_location = -1
        best_cost = float('inf')
        all_connected_data_qubits = [dq for dq, hq in self._data_helper_interaction if hq == helper_qubit_index]
        for hq in available_helper_qubits:
            cost = 0.0
            for dq in all_connected_data_qubits:
                phys_dq = data_qubit_mapping[dq]
                dist = hardware_distance[phys_dq][hq]
                cost += dist
            if cost < best_cost:
                best_cost = cost
                best_location = hq
        return best_location



    def __repr__(self):
        return f"circuit_topology(data_interaction={self._data_interaction}, data_helper_interaction={self._data_helper_interaction})"




def analyze_topo_from_instructions(inst_list: List[instruction]) -> circuit_topology:
    """
    We analyze the instructions to extract the circuit topology.
    We only consider CNOT and CZ gates for interaction analysis.
    1) data-data interactions: both qubits are data qubits (q0, q1, ...)
    2) data-helper interactions: one qubit is data (q0, q1, ...), the other is helper (s0, s1, ...)
    """
    data_qubit_set=set()
    helper_qubit_set=set()
    data_interaction=[]
    data_helper_interaction=[]

    for inst in inst_list:
        if inst.is_two_qubit_gate():
            q1=inst.get_qubitaddress()[0]
            q2=inst.get_qubitaddress()[1]
            if q1.startswith('q') and q2.startswith('q'):
                dq1=int(q1[1:])
                dq2=int(q2[1:])
                data_qubit_set.add(dq1)
                data_qubit_set.add(dq2)
                a,b=sorted((dq1,dq2))
                data_interaction.append( (a,b) )
            elif q1.startswith('q') and q2.startswith('s'):
                dq=int(q1[1:])
                sq=int(q2[1:])
                data_qubit_set.add(dq)
                helper_qubit_set.add(sq)
                data_helper_interaction.append( (dq,sq) )
            elif q1.startswith('s') and q2.startswith('q'):
                dq=int(q2[1:])
                sq=int(q1[1:])
                data_qubit_set.add(dq)
                helper_qubit_set.add(sq)
                data_helper_interaction.append( (dq,sq) )
            else:
                #both are helper qubits, ignore
                pass

    return circuit_topology(
        data_qubit_number=len(data_qubit_set),
        helper_qubit_number=len(helper_qubit_set),
        data_interaction=data_interaction,
        data_helper_interaction=data_helper_interaction,
    )




class ProcessStatus(Enum):
    WAIT_TO_START = 0
    RUNNING = 1
    WAIT_FOR_HELPER = 2
    FINISHED = 3




class process:


    def __init__(self, process_id: int, num_data_qubits: int, num_helper_qubits: int, shots: int,inst_list=List[instruction]):
        self._process_id = process_id
        self._num_data_qubits = num_data_qubits
        self._num_helper_qubits = num_helper_qubits
        self._topology = analyze_topo_from_instructions(inst_list)
        self._inst_list = inst_list
        self._data_qubit_mapping={}
        self._shots = shots
        self._remaining_shots = shots
        self._status = ProcessStatus.WAIT_TO_START
        self._result_counts = {}
        self._pointer = 0  # points to the next instruction to be scheduled


    def set_status(self, status: ProcessStatus):
        self._status = status


    def is_running(self) -> bool:
        return self._status == ProcessStatus.RUNNING
    

    def is_finished(self) -> bool:
        return self._status == ProcessStatus.FINISHED
    

    def get_remaining_shots(self) -> int:
        return self._remaining_shots


    def get_result_counts(self) -> Dict[str, int]:
        return self._result_counts
    

    def is_waiting_for_helper(self) -> bool:
        return self._status == ProcessStatus.WAIT_FOR_HELPER



    def get_next_instruction(self) -> instruction | None:
        """
        Get the next instruction to be scheduled, or None if all instructions are scheduled.
        """
        if self._pointer < len(self._inst_list):
            inst = self._inst_list[self._pointer]
            return inst
        else:
            self._status = ProcessStatus.FINISHED
            return None



    def execute_next_instruction(self):
        """
        Execute the next instruction by advance the pointer.
        """
        if self._pointer < len(self._inst_list):
            self._pointer += 1
        if self._pointer >= len(self._inst_list):
            self._status = ProcessStatus.FINISHED



    def update_data_qubit_mapping(self, L: Dict[int, tuple[int, int]]):
        """
        Update the data qubit mapping for all instruction, given the total data qubit layout mapping L.
        """
        for phys_qubit, (pid, data_qubit) in L.items():
            if pid == self._process_id:
                self._data_qubit_mapping[data_qubit] = phys_qubit
        for inst in self._inst_list:
            qubit_addresses = inst.get_qubitaddress()
            for qa in qubit_addresses:
                if qa.startswith('q'):
                    inst.set_scheduled_mapped_address(qa)


    def update_result(self, shots, counts: Dict[str, int]):
        self._remaining_shots -= shots
        for key, value in counts.items():
            if key in self._result_counts:
                self._result_counts[key] += value
            else:
                self._result_counts[key] = value


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

        Thus is defined as the sum of distance between all data qubit pairs weighted by their interaction weight.
        """
        cost = 0.0
        for i in range(self._num_data_qubits):
            for j in range(i+1,self._num_data_qubits):
                    cost += distance[mapping[i]][mapping[j]] * self._topology.get_data_data_weight(i, j)
        return cost


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
    color_map = {p.get_process_id(): colors[i] for i, p in enumerate(process_list)}

    for phys, (pid, data_qubit) in mapping.items():
        # find the process
        color = color_map[pid]
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
    # --- give each process a unique color ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(process_list)))  # up to 10 distinct
    color_map = {p.get_process_id(): colors[i] for i, p in enumerate(process_list)}
    for phys, (pid, data_qubit) in mapping.items():
        # find the process
        color = color_map[pid]
        x, y = coords[phys]
        ax.scatter([x], [y], s=780, facecolors="none", edgecolors=color,
                   linewidths=2.6, zorder=5)
        ax.text(x, y + 0.38, f"P{pid}-D{data_qubit}" , ha="center", va="bottom",
                fontsize=5, color=color, weight="bold", zorder=6)


    ax.set_aspect("equal"); ax.axis("off"); plt.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    file_path = "C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\cat_state_prep_n4"
    inst_list, data_n, syn_n = parse_program_from_file(file_path)


    circ_topo = analyze_topo_from_instructions(inst_list)

    print("Extracted circuit topology:", circ_topo)





# if __name__ == "__main__":
#     random.seed(42)  # for reproducibility

#     # ----- build 5 processes, each with 20 data and 10 helper qubits -----
#     process_list: list[process] = []
#     NUM_PROCS = 4
#     NUM_DATA = 15
#     NUM_HELPERS = 1

#     for pid in range(NUM_PROCS):
#         topo = make_process_topology(
#             num_data=NUM_DATA,
#             num_helper=NUM_HELPERS,
#             extra_edges_per_qubit=30,
#             seed=100 + pid,   # different but reproducible per process
#         )
#         proc = process(
#             process_id=pid,
#             num_data_qubits=NUM_DATA,
#             num_helper_qubits=NUM_HELPERS,
#             topology=topo,
#         )
#         process_list.append(proc)

#     # Sanity check: total data qubits must fit into hardware
#     total_data_qubits = sum(p.get_num_data_qubits() for p in process_list)
#     print("Total data qubits:", total_data_qubits, "Hardware qubits:", N_qubits)

#     # ----- run the heuristic search on Torino -----
#     best_mapping = iteratively_find_the_best_mapping(
#         process_list,
#         n_qubits=N_qubits,
#     )

#     print("Best mapping found:", best_mapping)
#     best_cost = calculate_mapping_cost(process_list, best_mapping)
#     print("Best cost:", best_cost)

#     # ----- visualize on Torino -----
#     plot_process_schedule_on_torino(
#         torino_coupling_map(),
#         process_list,
#         best_mapping,
#         out_png="best_torino_mapping_5proc_20data.png",
#     )

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