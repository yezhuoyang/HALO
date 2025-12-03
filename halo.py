import pickle
import json
from hardwares import construct_30_qubit_hardware, simple_10_qubit_coupling_map, simple_20_qubit_coupling_map, torino_coupling_map, construct_fake_ibm_torino, construct_10_qubit_hardware, simple_30_qubit_coupling_map
from instruction import instruction, Instype, parse_program_from_file, construct_qiskit_circuit
from process import all_pairs_distances, plot_process_schedule_on_torino, process, ProcessStatus
from typing import Dict, List, Optional, Set, Tuple
import random
import math
import queue
import threading
import time
from enum import Enum
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator  # <- use AerSimulator (Qiskit 2.x)
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService


import matplotlib
matplotlib.use('Agg') 
#Mute the qiskit_ibm_runtime logging info
import logging
logging.getLogger("qiskit_ibm_runtime").setLevel(logging.ERROR)



class benchmarktype(Enum):
    RANDOM_SMALL = 0
    RANDOM_MEDIUM = 1
    MULTI_CONTROLLED_X_SMALL = 2
    MULTI_CONTROLLED_X_MEDIUM = 3
    STABILIZER_MEASUREMENT_SMALL =4
    STABILIZER_MEASUREMENT_MEDIUM =5
    CLASSICAL_LOGIC_SMALL =6
    CLASSICAL_LOGIC_MEDIUM = 7
    MIX = 8



benchmark_file_path={
    benchmarktype.RANDOM_SMALL:"benchmark//randomsmall//",
    benchmarktype.RANDOM_MEDIUM:"benchmark//randommedium//",
    benchmarktype.MULTI_CONTROLLED_X_SMALL:"benchmark//multiXsmall//",
    benchmarktype.MULTI_CONTROLLED_X_MEDIUM:"benchmark//multiXmedium//",
    benchmarktype.STABILIZER_MEASUREMENT_SMALL:"benchmark//qecsmall//",
    benchmarktype.STABILIZER_MEASUREMENT_MEDIUM:"benchmark//qecmedium//",
    benchmarktype.CLASSICAL_LOGIC_SMALL:"benchmark//arithsmall//",
    benchmarktype.CLASSICAL_LOGIC_MEDIUM:"benchmark//arithmedium//",
}


benchmark_result_path={
    benchmarktype.RANDOM_SMALL:"benchmark//result2000shots//randomsmall//",
    benchmarktype.RANDOM_MEDIUM:"benchmark//result2000shots//randommedium//",
    benchmarktype.MULTI_CONTROLLED_X_SMALL:"benchmark//result2000shots//multiXsmall//",
    benchmarktype.MULTI_CONTROLLED_X_MEDIUM:"benchmark//result2000shots//multiXmedium//",
    benchmarktype.STABILIZER_MEASUREMENT_SMALL:"benchmark//result2000shots//qecsmall//",
    benchmarktype.STABILIZER_MEASUREMENT_MEDIUM:"benchmark//result2000shots//qecmedium//",
    benchmarktype.CLASSICAL_LOGIC_SMALL:"benchmark//result2000shots//arithsmall//",
    benchmarktype.CLASSICAL_LOGIC_MEDIUM:"benchmark//result2000shots//arithmedium//",
}




random_small_benchmark = {
    0: "data_4_syn_4_gc_10_0",
    1: "data_4_syn_4_gc_15_0",
    2: "data_5_syn_4_gc_15_0",
    3: "data_5_syn_5_gc_15_0",
    4: "data_5_syn_5_gc_20_0",
    5: "data_6_syn_6_gc_15_0",
    6: "data_6_syn_6_gc_18_0",
    7: "data_7_syn_7_gc_10_0",
    8: "data_7_syn_7_gc_25_0",
    9: "data_8_syn_8_gc_22_0",
    10: "data_9_syn_8_gc_20_0",
}


random_medium_benchmark = {
    0: "data_10_syn_20_gc_45_0",
    1: "data_10_syn_20_gc_55_0",
    2: "data_11_syn_19_gc_48_0",
    3: "data_12_syn_18_gc_58_0",
    4: "data_13_syn_17_gc_52_0",
    5: "data_14_syn_15_gc_45_0",
    6: "data_15_syn_12_gc_40_0",
    7: "data_15_syn_15_gc_30_0",
    8: "data_15_syn_15_gc_40_0",
    9: "data_17_syn_13_gc_50_0",
    10: "data_20_syn_10_gc_50_0",
}




qec_small_benchmark = {
    0: "cat_state_prep_n4",
    1: "cat_state_verification_n4",
    2: "repetition_code_distance3_n3",

    # --- Repetition code with specific error patterns ---
    3: "repetition_d3_round_2_error_IIY",
    4: "repetition_d3_round_2_error_XII",
    5: "repetition_d3_round_2_error_ZII",
    6: "repetition_d3_round_3_error_IYI",
    7: "repetition_d3_round_3_error_IZI",

    # --- Shor code circuits ---
    8: "shor_parity_measurement_n4",
    9: "shor_stabilizer_XZZX_n3",
    10: "shor_stabilizer_ZZZZ_n4",

    # --- Surface-code syndrome extraction ---
    11: "syndrome_extraction_surface_n4",
}

qec_medium_benchmark = {
    # --- Clean circuits (your original entries) ---

    # --- Five-qubit code (distance-3), round 2 ---
    0:  "fivequbit_round_2_error_IIZII",
    1:  "fivequbit_round_2_error_IXIII",
    2:  "fivequbit_round_2_error_YIIII",
    3:  "fivequbit_round_3_error_IIIXI",
    4:  "fivequbit_round_3_error_IXIII",


    # --- Repetition code distance-5, round 2 ---
    5: "repetition_d5_round_2_error_IIIYI",
    6: "repetition_d5_round_2_error_IIIZI",
    7: "repetition_d5_round_2_error_XIIII",
    8: "repetition_d5_round_3_error_IIIIZ",
    9: "repetition_d5_round_3_error_IYIII",
    10: "repetition_d5_round_3_error_ZIIII",


    # --- Shor code, round 2 ---
    11: "shor_round_2_error_IIIIIIIIZ",
    12: "shor_round_2_error_IIIIIIZII",
    13: "shor_round_2_error_XIIIIIIII",

    14: "shor_round_3_error_IIYIIIIII",
    15: "shor_round_3_error_IYIIIIIII",
    16: "shor_round_3_error_XIIIIIIII",

    # --- Steane code, round 2 ---
    17: "steane_round_2_error_IIIIIXI",
    18: "steane_round_2_error_IIIYIII",
    19: "steane_round_2_error_ZIIIIII",

    # --- Steane code, round 3 ---
    20: "steane_round_3_error_IXIIIII",
    21: "steane_round_3_error_XIIIIII",
    22: "steane_round_3_error_ZIIIIII",

    # --- Surface code distance-3, round 2 ---
    23: "surface_d3_round_2_error_IIIIIIIIZ",
    24: "surface_d3_round_2_error_IIIIIIIXI",
    25: "surface_d3_round_2_error_IIIIIXIII",

    # --- Surface code distance-3, round 3 ---
    26: "surface_d3_round_3_error_IIIIIXIII",
    27: "surface_d3_round_3_error_IIIXIIIII",
    28: "surface_d3_round_3_error_XIIIIIIII",
}


multix_small_benchmark={
    0:"mcx_2_0",
    1:"mcx_2_1",
    2:"mcx_3_0",
    3:"mcx_3_1",
    4:"mcx_4_0",
    5:"mcx_4_1",
    6:"mcx_5_0",
    7:"mcx_5_1",
    8:"mcx_6_0",
    9:"mcx_6_1",
    10:"mcx_7_0",
    11:"mcx_7_1"
}


multix_medium_benchmark={
    0: "mcx_9_0",
    1: "mcx_9_1",
    2: "mcx_10_0",
    3: "mcx_10_1",
    4: "mcx_11_0",
    5: "mcx_11_1",
    6: "mcx_12_0",
    7: "mcx_12_1",
    8: "mcx_13_0",
    9: "mcx_13_1",
    10: "mcx_14_0",
    11: "mcx_14_1"
}


classical_logic_small_benchmark={
    0: "varnum_3_d_3_0",
    1: "varnum_3_d_3_1",
    2: "varnum_3_d_4_0",
    3: "varnum_3_d_4_1",
    4: "varnum_4_d_2_0",
    5: "varnum_4_d_2_1",
    6: "varnum_4_d_3_0",
    7: "varnum_4_d_3_1",
    8: "varnum_4_d_4_0",
    9: "varnum_4_d_4_1"
}




classical_logic_medium_benchmark = {
    0:  "varnum_7_d_5_0",
    1:  "varnum_7_d_5_1",
    2:  "varnum_8_d_4_0",
    3:  "varnum_8_d_5_0",
    4:  "varnum_9_d_4_0",
    5:  "varnum_10_d_3_0",
    6:  "varnum_10_d_3_1",
    7:  "varnum_10_d_4_0",
    8:  "varnum_11_d_4_0",
    9:  "varnum_11_d_4_1",
    10: "varnum_12_d_4_0",
    11: "varnum_12_d_4_1",
}


benchmark_type_to_benchmark={
    benchmarktype.RANDOM_SMALL: random_small_benchmark,
    benchmarktype.RANDOM_MEDIUM: random_medium_benchmark,
    benchmarktype.MULTI_CONTROLLED_X_SMALL: multix_small_benchmark,
    benchmarktype.MULTI_CONTROLLED_X_MEDIUM: multix_medium_benchmark,
    benchmarktype.STABILIZER_MEASUREMENT_SMALL: qec_small_benchmark,
    benchmarktype.STABILIZER_MEASUREMENT_MEDIUM: qec_medium_benchmark,
    benchmarktype.CLASSICAL_LOGIC_SMALL: classical_logic_small_benchmark,
    benchmarktype.CLASSICAL_LOGIC_MEDIUM: classical_logic_medium_benchmark
}




class SchedulingOptions(Enum):
    BASELINE_SEQUENTIAL = 0
    NO_SHARING = 1
    HALO =2 
    SHOT_UNAWARE=3


def distribution_fidelity(dist1: dict, dist2: dict) -> float:
    """
    Compute fidelity between two distributions based on L1 distance.
    1. Normalize both distributions to get probability distributions.
    2. Compute the L1 distance between the two probability distributions.
    3. Fidelity = 1 - L1/2, which lies in [0,1].
    
    Args:
        dist1 (dict): key=str (event), value=int (count)
        dist2 (dict): key=str (event), value=int (count)
    
    Returns:
        float: fidelity in [0, 1]
    """
    # Step 1: normalize distributions
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())
    prob1 = {k: v / total1 for k, v in dist1.items()}
    prob2 = {k: v / total2 for k, v in dist2.items()}
    
    # Step 2: union of keys
    all_keys = set(prob1.keys()).union(set(prob2.keys()))
    
    # Step 3: compute L1 distance
    l1_distance = sum(abs(prob1.get(k, 0) - prob2.get(k, 0)) for k in all_keys)
    
    # Step 4: fidelity = 1 - L1/2
    fidelity = 1 - l1_distance / 2
    return fidelity



def load_ideal_count_output(benchmark_type:benchmarktype, benchmark_id: int) -> Dict[str, int]:
    """
    Load the ideal count output from the benchmark suit file
    """
    benchmark_dict= benchmark_type_to_benchmark[benchmark_type]
    filename=benchmark_dict[benchmark_id]
    filename_result = benchmark_result_path[benchmark_type] + filename + "_counts.pkl"
    with open(filename_result, 'rb') as f:
        ideal_counts = pickle.load(f)
    return ideal_counts




"""
Hyper parameters for mapping cost calculation:
alpha: weight for intra-process cost
beta: weight for inter-process cost
gamma: weight for helper qubit cost
delta: weight for compact cost
"""
alpha=0.5
beta=0.3
gamma=1
delta=0.3
eps = 0.3

#Set the scheduling option here
Scheduling_Option=SchedulingOptions.NO_SHARING
benchmark_Option=benchmarktype.MULTI_CONTROLLED_X_MEDIUM


# N_qubits=133
# hardware_distance_pair=all_pairs_distances(N_qubits, torino_coupling_map())
fake_torino_backend=construct_fake_ibm_torino()


N_qubits=133
hardware_distance_pair=all_pairs_distances(N_qubits, torino_coupling_map())

# N_qubits=133
# hardware_distance_pair=all_pairs_distances(N_qubits, simple_30_qubit_coupling_map())
DIST_MATRIX = np.asarray(hardware_distance_pair, dtype=float)
fake_backend=construct_fake_ibm_torino()


data_hardware_ratio=0.8
#The maximum distance between a data qubit and a helper qubit
max_data_helper_distance=10

# N_qubits=10
# hardware_distance_pair=all_pairs_distances(N_qubits, simple_10_qubit_coupling_map())






def calculate_intro_cost_for_process(process_list: List[process],proc_mapping: Dict[int, Dict[int, int]]) -> float:
    """
    Calculate the intra-process cost for all processes in the batch given the mapping
    The intro cost is defined as the layout mapping cost within each process
    """
    intro_cost=0.0
    #First step, calculate the intra cost of all process
    for proc in process_list:
        intro_cost+=proc.intro_costs(proc_mapping[proc.get_process_id()],DIST_MATRIX)
        # intro_cost+=proc.intro_costs(proc_mapping[proc.get_process_id()],hardware_distance_pair)

    return intro_cost




def calculate_helper_cost_for_process(process_list: List[process],proc_mapping: Dict[int, Dict[int, int]],helper_qubit_list: List[int]) -> float:
    """
    Calculate the helper qubit cost for all processes in the batch given the mapping
    This is defined as the total distance between data qubits and their nearest helper qubits, weighted by the helper qubit weight
    """
    helper_cost=0.0

    if len(helper_qubit_list)==0:
        return float('inf')


    helpers = np.asarray(helper_qubit_list, dtype=int)


    min_dist_to_helper = DIST_MATRIX[:, helpers].min(axis=1)

    #Accumulate helper cost
    helper_cost = 0.0
    for proc in process_list:
        pid = proc.get_process_id()
        mapping = proc_mapping[pid]
        topo = proc.get_topology()
        num_data = proc.get_num_data_qubits()

        for data_qubit in range(num_data):
            helper_weight = topo.get_data_helper_weight(data_qubit)
            if not helper_weight:
                continue
            phys = mapping[data_qubit]
            helper_cost += helper_weight * min_dist_to_helper[phys]

    return helper_cost




def calculate_inter_cost_for_process(process_list: List[process],proc_mapping: Dict[int, Dict[int, int]]) -> float:
    """
    Calculate the inter-process cost for all processes in the batch given the mapping
    The cost is defined as the distance between data qubits of different processes
    """

    #Precompute the list of physical qubits per process
    #Don't look up the dict in the inner most loop
    proc_ids: List[int] = []
    proc_phys_lists: List[np.ndarray] = []

    for proc in process_list:
        pid = proc.get_process_id()
        mapping = proc_mapping[pid]
        num_data = proc.get_num_data_qubits()
        #Create numpy array of phys indices in data-qubit order
        phys_arr = np.fromiter([mapping[dq] for dq in range(num_data)], dtype=int, count=num_data)
        proc_ids.append(pid)
        proc_phys_lists.append(phys_arr)


    inter_cost=0.0
    n = len(process_list)

    for i in range(n):
        proc_i = proc_phys_lists[i]
        if proc_i.size == 0:
            continue

        for j in range(i + 1, n):
            proc_j = proc_phys_lists[j]
            if proc_j.size == 0:
                continue
            # Compute pairwise distances between data qubits of proc_i and proc_j
            sub = DIST_MATRIX[np.ix_(proc_i, proc_j)]
            total_distance = float(sub.sum())
            count = sub.size
            if count > 0:
                inter_cost += total_distance / count


    return inter_cost



def calculate_mapping_cost(process_list: List[process],mapping: Dict[int, tuple[int, int]]) -> float:
    """
    Input:
        mapping: Dict[int, tuple[int, int]]
            A dictionary where the key is physical, and the value is a tuple (process_id, data_qubit_index)
        
        Output:
            A float representing the total mapping cost, calculated as:
            cost = alpha * intro_cost - beta * inter_cost + gamma * helper_cost + delta * compact_cost
    """


    used_phys = set(mapping.keys())
    helper_qubit_list=[phys for phys in range(N_qubits) if phys not in used_phys]


    #----Convert to per-process mapping format----
    proc_mapping: Dict[int, Dict[int, int]] = {pid:{} for pid in [proc.get_process_id() for proc in process_list]}
    for phys,(pid,data_qubit) in mapping.items():
        proc_mapping[pid][data_qubit]=phys


    intro_cost = calculate_intro_cost_for_process(process_list,proc_mapping)
    inter_cost = calculate_inter_cost_for_process(process_list,proc_mapping)
    helper_cost= calculate_helper_cost_for_process(process_list,proc_mapping,helper_qubit_list)


    return alpha * intro_cost - beta * inter_cost + gamma * helper_cost

class MappingCostState:
    """
    FULL general incremental cost engine supporting ALL local changes:
    - swap
    - move to helper
    - move back from helper
    - reassign arbitrary phys

    Normalized components:
      intro_norm   ~ [0, 1]   (intra-process layout cost)
      inter_norm   ~ [0, 1]   (mean distance between processes)
      helper_norm  ~ [0, 1]   (data→helper distances, weighted)
      compact_norm ~ [0, 1]   (cluster tightness per process)
      path_norm    ~ [0, 1]   (NEW: foreign qubits along shortest paths)

    Total cost:
      cost = alpha * intro_norm
           - beta  * inter_norm
           + gamma * helper_norm
           + delta * compact_norm
           + eps   * path_norm
    """

    def __init__(self, process_list, mapping, n_qubits, dist_matrix):
        self.process_list = process_list
        self.n_qubits = n_qubits
        self.dist = dist_matrix

        # phys -> (pid, dq)
        self.mapping = dict(mapping)

        # Process metadata
        self.pid_to_proc = {p.get_process_id(): p for p in process_list}
        self.pids = [p.get_process_id() for p in process_list]
        self.num_proc = len(self.pids)

        # Logical→physical per process
        self.proc_mapping = {pid: {} for pid in self.pids}
        for phys, (pid, dq) in self.mapping.items():
            self.proc_mapping[pid][dq] = phys

        # Used vs helper phys
        used_mask = np.zeros(n_qubits, dtype=bool)
        for phys in self.mapping:
            used_mask[phys] = True
        self.used_mask = used_mask
        self.helper_mask = ~used_mask
        self.helper_qubits = np.where(self.helper_mask)[0]

        # Which process occupies each phys?  -1 means helper/unused
        self.pid_at_phys = np.full(self.n_qubits, -1, dtype=int)
        for phys, (pid, dq) in self.mapping.items():
            self.pid_at_phys[phys] = pid

        # Build adjacency & canonical shortest paths on the hardware graph
        self._build_adjacency()
        self._precompute_shortest_paths()

        # Normalization bounds
        self._init_normalization_bounds()

        # Cost caches (unnormalized sums)
        self.proc_intro_cost   = {}
        self.proc_helper_cost  = {}
        self.proc_compact_cost = {}
        self.inter_cost_pair   = {}

        self.sum_intro   = 0.0
        self.sum_helper  = 0.0
        self.sum_compact = 0.0
        self.sum_inter   = 0.0

        # NEW: path-conflict (unnormalized)
        self.path_conflict_sum = 0.0

        # Precompute helper distances for INITIAL state
        self._compute_min_dist_to_helper()

        # Initialize all pieces
        self._init_process_costs()
        self._init_inter_costs()
        self._compute_path_conflict_full()

    # ------------------------------------------------------------
    # Adjacency + shortest paths on hardware graph
    # ------------------------------------------------------------
    def _build_adjacency(self):
        """Adjacency list from dist == 1 (physical couplings)."""
        self.adj = [[] for _ in range(self.n_qubits)]
        for u in range(self.n_qubits):
            neighs = np.where(self.dist[u] == 1)[0]
            for v in neighs:
                self.adj[u].append(v)

    def _precompute_shortest_paths(self):
        """
        For every pair (u,v), u < v, store one canonical shortest path
        as a list of vertices [u, ..., v].
        """
        from collections import deque

        self.shortest_paths = {}
        for s in range(self.n_qubits):
            prev = [-1] * self.n_qubits
            q = deque([s])
            prev[s] = -2  # root marker

            # BFS
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if prev[v] == -1:
                        prev[v] = u
                        q.append(v)

            # Recover paths s→t
            for t in range(s + 1, self.n_qubits):
                if prev[t] == -1:
                    continue  # disconnected (shouldn't happen in your hardware)
                path = []
                cur = t
                while cur != -2:
                    path.append(cur)
                    cur = prev[cur]
                path.append(s)
                path.reverse()
                self.shortest_paths[(s, t)] = path

    # ------------------------------------------------------------
    # Normalization bounds
    # ------------------------------------------------------------
    def _init_normalization_bounds(self):
        """
        Compute heuristic upper bounds for intro, helper, inter, compact,
        and path-conflict so we can normalize each term to ~[0,1].
        """
        max_dist = float(self.dist.max()) if self.dist.size > 0 else 1.0
        self.max_dist = max_dist

        # intro_max: complete graph over data qubits per process
        intro_max = 0.0
        for p in self.process_list:
            d = p.get_num_data_qubits()
            num_edges = d * (d - 1) / 2.0
            intro_max += num_edges * max_dist
        if intro_max <= 0.0:
            intro_max = 1.0
        self.intro_max = intro_max

        # helper_max: total helper weight * max_dist
        total_helper_weight = 0.0
        for p in self.process_list:
            topo = p.get_topology()
            for dq in range(p.get_num_data_qubits()):
                w = topo.get_data_helper_weight(dq)
                if w:
                    total_helper_weight += w
        helper_max = total_helper_weight * max_dist
        if helper_max <= 0.0:
            helper_max = 1.0
        self.helper_max = helper_max

        # inter_max: mean distance between any two processes
        self.inter_max = max_dist if self.num_proc > 1 else 1.0

        # compact_max: each process has diameter ~max_dist
        self.compact_max = self.num_proc * max_dist if self.num_proc > 0 else 1.0

        # path_conflict_max: worst case every intermediate node is foreign
        # for every pair of data qubits in each process.
        max_intermediate = max(0.0, max_dist - 1.0)
        path_max = 0.0
        for p in self.process_list:
            d = p.get_num_data_qubits()
            num_pairs = d * (d - 1) / 2.0
            path_max += num_pairs * max_intermediate
        if path_max <= 0.0:
            path_max = 1.0
        self.path_conflict_max = path_max

    # ------------------------------------------------------------
    # Helper distances
    # ------------------------------------------------------------
    def _compute_min_dist_to_helper(self):
        if self.helper_qubits.size > 0:
            self.min_dist_to_helper = self.dist[:, self.helper_qubits].min(axis=1)
        else:
            self.min_dist_to_helper = np.zeros(self.n_qubits)

    # ------------------------------------------------------------
    # Compactness & initial process costs
    # ------------------------------------------------------------
    def _compute_compact_for_pid(self, pid):
        """
        Compactness cost for one process:
          mean pairwise distance between all its data qubits.
        """
        proc = self.pid_to_proc[pid]
        num_data = proc.get_num_data_qubits()
        if num_data <= 1:
            return 0.0

        phys_arr = np.fromiter(
            (self.proc_mapping[pid][dq] for dq in range(num_data)),
            dtype=int,
            count=num_data,
        )
        if phys_arr.size <= 1:
            return 0.0

        sub = self.dist[np.ix_(phys_arr, phys_arr)]
        total = sub.sum()
        n = phys_arr.size
        num_pairs = n * (n - 1)  # each unordered pair counted twice
        if num_pairs == 0:
            return 0.0
        return float(total / num_pairs)

    def _init_process_costs(self):
        for pid in self.pids:
            proc = self.pid_to_proc[pid]
            L = self.proc_mapping[pid]

            # intro
            intro = proc.intro_costs(L, self.dist)
            self.proc_intro_cost[pid] = intro
            self.sum_intro += intro

            # helper
            topo = proc.get_topology()
            hcost = 0.0
            for dq in range(proc.get_num_data_qubits()):
                w = topo.get_data_helper_weight(dq)
                if w == 0:
                    continue
                phys = L[dq]
                hcost += w * self.min_dist_to_helper[phys]
            self.proc_helper_cost[pid] = hcost
            self.sum_helper += hcost

            # compact
            ccost = self._compute_compact_for_pid(pid)
            self.proc_compact_cost[pid] = ccost
            self.sum_compact += ccost

    # ------------------------------------------------------------
    # Inter-process distance
    # ------------------------------------------------------------
    def _init_inter_costs(self):
        for i in range(self.num_proc):
            pid_i = self.pids[i]
            proc_i = self.pid_to_proc[pid_i]
            phys_i = np.array(
                [self.proc_mapping[pid_i][dq] for dq in range(proc_i.get_num_data_qubits())]
            )

            for j in range(i + 1, self.num_proc):
                pid_j = self.pids[j]
                proc_j = self.pid_to_proc[pid_j]
                phys_j = np.array(
                    [self.proc_mapping[pid_j][dq] for dq in range(proc_j.get_num_data_qubits())]
                )

                if phys_i.size == 0 or phys_j.size == 0:
                    val = 0.0
                else:
                    sub = self.dist[np.ix_(phys_i, phys_j)]
                    val = float(sub.mean())

                self.inter_cost_pair[(pid_i, pid_j)] = val
                self.sum_inter += val

    # ------------------------------------------------------------
    # Path-conflict (foreign qubits along shortest paths)
    # ------------------------------------------------------------
    def _compute_path_conflict_full(self):
        """
        Recompute path_conflict_sum from scratch.

        For each process pid and each pair of its data-qubit physical
        locations (u, v), we look up a precomputed shortest path.
        We count intermediate vertices whose pid != pid and pid != -1.

        Sum over all processes & pairs.
        """
        total = 0.0
        for pid in self.pids:
            proc = self.pid_to_proc[pid]
            num_data = proc.get_num_data_qubits()
            if num_data <= 1:
                continue

            phys_arr = [
                self.proc_mapping[pid][dq] for dq in range(num_data)
            ]

            for i in range(num_data):
                u = phys_arr[i]
                for j in range(i + 1, num_data):
                    v = phys_arr[j]
                    if u < v:
                        key = (u, v)
                    else:
                        key = (v, u)
                    path = self.shortest_paths.get(key, None)
                    if path is None or len(path) <= 2:
                        continue

                    # skip endpoints; count foreign occupants in between
                    for w in path[1:-1]:
                        pid_w = self.pid_at_phys[w]
                        if pid_w != -1 and pid_w != pid:
                            total += 1.0

        self.path_conflict_sum = total

    # ------------------------------------------------------------
    # Recompute helpers after a local change
    # ------------------------------------------------------------
    def _recompute_process(self, pid):
        proc = self.pid_to_proc[pid]
        L = self.proc_mapping[pid]

        # intro
        old_intro = self.proc_intro_cost[pid]
        new_intro = proc.intro_costs(L, self.dist)
        self.proc_intro_cost[pid] = new_intro
        self.sum_intro += new_intro - old_intro

        # helper
        old_helper = self.proc_helper_cost[pid]
        topo = proc.get_topology()
        hcost = 0.0
        for dq in range(proc.get_num_data_qubits()):
            w = topo.get_data_helper_weight(dq)
            if w == 0:
                continue
            phys = L[dq]
            hcost += w * self.min_dist_to_helper[phys]
        self.proc_helper_cost[pid] = hcost
        self.sum_helper += hcost - old_helper

        # compact
        old_compact = self.proc_compact_cost[pid]
        new_compact = self._compute_compact_for_pid(pid)
        self.proc_compact_cost[pid] = new_compact
        self.sum_compact += new_compact - old_compact

    def _recompute_inter_for(self, pid_x, pid_y):
        if pid_x == pid_y:
            return
        if pid_x < pid_y:
            key = (pid_x, pid_y)
        else:
            key = (pid_y, pid_x)

        old = self.inter_cost_pair[key]

        proc_x = self.pid_to_proc[pid_x]
        proc_y = self.pid_to_proc[pid_y]
        phys_x = np.array(
            [self.proc_mapping[pid_x][dq] for dq in range(proc_x.get_num_data_qubits())]
        )
        phys_y = np.array(
            [self.proc_mapping[pid_y][dq] for dq in range(proc_y.get_num_data_qubits())]
        )

        if phys_x.size == 0 or phys_y.size == 0:
            new = 0.0
        else:
            sub = self.dist[np.ix_(phys_x, phys_y)]
            new = float(sub.mean())

        self.inter_cost_pair[key] = new
        self.sum_inter += new - old

    # ------------------------------------------------------------
    # GENERAL LOCAL MOVE: reassign
    # ------------------------------------------------------------
    def reassign(self, phys_src, phys_dst):
        """
        Move the data-qubit at phys_src to phys_dst.
        If phys_dst was occupied, we perform a swap.
        If phys_dst was helper, phys_src becomes helper afterwards.
        """
        src_pid, src_dq = self.mapping[phys_src]
        dst_was_used = self.used_mask[phys_dst]

        if dst_was_used:
            # SWAP CASE
            dst_pid, dst_dq = self.mapping[phys_dst]

            # swap mapping
            self.mapping[phys_src], self.mapping[phys_dst] = \
                (dst_pid, dst_dq), (src_pid, src_dq)

            self.proc_mapping[src_pid][src_dq] = phys_dst
            self.proc_mapping[dst_pid][dst_dq] = phys_src

            affected = {src_pid, dst_pid}

        else:
            # MOVE-TO-HELPER CASE
            self.mapping[phys_dst] = (src_pid, src_dq)
            del self.mapping[phys_src]

            # update proc mapping
            self.proc_mapping[src_pid][src_dq] = phys_dst

            # update used/helper masks
            self.used_mask[phys_src] = False
            self.helper_mask[phys_src] = True
            self.used_mask[phys_dst] = True
            self.helper_mask[phys_dst] = False

            self.helper_qubits = np.where(self.helper_mask)[0]
            self._compute_min_dist_to_helper()

            affected = {src_pid}

        # recompute process costs (intro, helper, compact) for affected
        for pid in affected:
            self._recompute_process(pid)

        # recompute intercost pairs
        for pid in self.pids:
            if pid in affected:
                continue
            for pid_x in affected:
                self._recompute_inter_for(pid_x, pid)
        if len(affected) == 2:
            a, b = tuple(affected)
            self._recompute_inter_for(a, b)

        # update pid_at_phys from mapping
        self.pid_at_phys[:] = -1
        for phys, (pid, dq) in self.mapping.items():
            self.pid_at_phys[phys] = pid

        # and recompute global path-conflict
        self._compute_path_conflict_full()

    # ------------------------------------------------------------
    # cost
    # ------------------------------------------------------------
    def get_cost(self):
        """
        Return normalized total cost.
        """
        intro_norm  = self.sum_intro   / self.intro_max
        helper_norm = self.sum_helper  / self.helper_max
        inter_norm  = self.sum_inter   / self.inter_max
        compact_norm = self.sum_compact / self.compact_max
        path_norm   = self.path_conflict_sum / self.path_conflict_max

        return (
            alpha * intro_norm
            - beta  * inter_norm
            + gamma * helper_norm
            + delta * compact_norm
            + eps   * path_norm
        )



# NEW: limit how far a move/swap can go on the hardware graph
LOCAL_RADIUS = 2   # 1 = strictly nearest-neighbour, 2 = slightly more flexible

def propose_move(
    state: MappingCostState,
    move_prob: float = 0.3,
) -> Optional[Tuple[int, int, bool]]:
    """
    Propose a *local* move on the given MappingCostState.

    Returns:
        (phys_src, phys_dst, dst_was_used) or None if no move is possible.

    Semantics:
        - If dst_was_used == True: this is a SWAP between two mapped phys,
          but ONLY within the same process and within LOCAL_RADIUS.
        - If dst_was_used == False: this is MOVE-TO-HELPER: phys_src (used) -> phys_dst (helper),
          and phys_dst must be within LOCAL_RADIUS of phys_src.

    IMPORTANT:
        - We NEVER swap qubits belonging to different processes.
        - All motion is local (bounded by LOCAL_RADIUS in the hardware distance).
    """
    used_phys = list(state.mapping.keys())
    num_used = len(used_phys)

    if num_used == 0:
        return None

    helper_qubits = state.helper_qubits
    has_helpers = helper_qubits.size > 0

    # Pick a random source data qubit
    phys_src = random.choice(used_phys)
    src_pid, src_dq = state.mapping[phys_src]

    # Precompute distances from src to all phys
    d_from_src = state.dist[phys_src]

    # --------------------------------------------------------
    # Helper candidates: local helpers only
    # --------------------------------------------------------
    local_helpers = []
    if has_helpers:
        for h in helper_qubits:
            if d_from_src[h] <= LOCAL_RADIUS:
                local_helpers.append(int(h))

    # --------------------------------------------------------
    # Decide whether to attempt a move-to-helper or a swap
    # --------------------------------------------------------
    do_move = (len(local_helpers) > 0) and (random.random() < move_prob)

    if do_move:
        # ---- LOCAL MOVE TO HELPER ----
        phys_dst = random.choice(local_helpers)
        return phys_src, phys_dst, False

    # --------------------------------------------------------
    # LOCAL SWAP *within the same process* only
    # --------------------------------------------------------
    same_pid_local_targets: List[int] = []
    for phys in used_phys:
        if phys == phys_src:
            continue
        pid, dq = state.mapping[phys]
        if pid != src_pid:
            continue  # different process => forbidden
        if d_from_src[phys] <= LOCAL_RADIUS:
            same_pid_local_targets.append(phys)

    if not same_pid_local_targets:
        # No valid local swap partner; fall back to local move if possible
        if local_helpers:
            phys_dst = random.choice(local_helpers)
            return phys_src, phys_dst, False
        # No valid move at all
        return None

    # Do a local, intra-process swap
    phys_dst = random.choice(same_pid_local_targets)
    return phys_src, phys_dst, True


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
    Greedy, *cluster-by-process* placement.

    Strategy:
    1) Choose one seed physical qubit per process.
       - First process gets phys 0.
       - Each subsequent process gets the unused phys that is farthest
         (maximin distance) from all previously chosen seeds.
       This spreads processes across the chip.

    2) For each process, grow its own cluster around its seed:
       - For each remaining data qubit of that process, pick the unused
         physical qubit that is closest to *that process's existing cluster*
         (not to the global cluster).
       This keeps each process contiguous but avoids interleaving processes.

    Remaining physical qubits are left unused (potential helper zone).
    """
    total_data_qubits = sum(p.get_num_data_qubits() for p in process_list)
    if total_data_qubits > n_qubits:
        raise ValueError("Not enough physical qubits for all data qubits")

    # numpy distance matrix (n_qubits x n_qubits)
    dist_mat = np.asarray(distance, dtype=float)

    mapping: Dict[int, tuple[int, int]] = {}
    used_mask = np.zeros(n_qubits, dtype=bool)

    # ------------------------------------------------------------
    # Step 1: choose a "seed" phys for each process, far apart
    # ------------------------------------------------------------
    seeds: Dict[int, int] = {}
    cluster_phys: Dict[int, List[int]] = {}

    for idx, proc in enumerate(process_list):
        pid = proc.get_process_id()

        if idx == 0:
            # First process: seed at 0
            seed_phys = 0
        else:
            # Other processes: pick the unused phys that maximizes
            # distance to the set of existing seeds (maximin).
            candidates = np.where(~used_mask)[0]
            if candidates.size == 0:
                raise ValueError("No free physical qubits left for new process seed.")

            existing_seeds = np.fromiter(seeds.values(), dtype=int)
            # distances from each candidate to nearest seed
            # shape: (num_candidates, num_seeds) -> min over axis=1
            cand_dists = dist_mat[np.ix_(candidates, existing_seeds)].min(axis=1)
            # pick candidate with largest min distance to seeds
            seed_idx = int(cand_dists.argmax())
            seed_phys = int(candidates[seed_idx])

        # Assign first data qubit (dq=0) of this process to the seed
        mapping[seed_phys] = (pid, 0)
        used_mask[seed_phys] = True
        seeds[pid] = seed_phys
        cluster_phys[pid] = [seed_phys]

    # ------------------------------------------------------------
    # Step 2: grow each process's cluster around its own seed
    # ------------------------------------------------------------
    for proc in process_list:
        pid = proc.get_process_id()
        num_data = proc.get_num_data_qubits()

        # dq = 0 already placed at the seed
        for dq in range(1, num_data):
            # Current cluster for this process
            cluster = np.array(cluster_phys[pid], dtype=int)

            # For every physical qubit, compute distance to this cluster
            # (min distance to any qubit in the cluster)
            dists_to_cluster = dist_mat[:, cluster].min(axis=1)

            # Can't reuse occupied qubits
            dists_to_cluster = np.where(used_mask, float("inf"), dists_to_cluster)

            best_phys = int(dists_to_cluster.argmin())
            if not np.isfinite(dists_to_cluster[best_phys]):
                raise ValueError("No available physical site found during greedy mapping.")

            mapping[best_phys] = (pid, dq)
            used_mask[best_phys] = True
            cluster_phys[pid].append(best_phys)

    # ------------------------------------------------------------
    # Optional: visualize the initial territories
    # ------------------------------------------------------------
    plot_process_schedule_on_torino(
        coupling_edges=torino_coupling_map(),
        process_list=process_list,
        mapping=mapping,
        out_png="greedy_initial.png",
    )

    return mapping



def iteratively_find_the_best_mapping_for_data(
    process_list: List[process],
    n_qubits: int,
    n_restarts: int = 30,
    steps_per_restart: int = 800,
    move_prob: float = 0.3,
    interrupt_event: Optional[threading.Event] = None,
) -> Dict[int, Tuple[int, int]]:
    """
    Heuristic search for a good mapping using simulated annealing
    with multiple random restarts.

    This version uses MappingCostState to maintain the cost incrementally.
    Local moves:
      - swap two mapped physical qubits
      - move a mapped qubit to a helper qubit (changing helper zone)

    If `interrupt_event` is provided and gets set(), the search will
    stop early and return the best mapping found so far.
    """
    global_best_mapping: Optional[Dict[int, Tuple[int, int]]] = None
    global_best_cost = float("inf")

    # Geometric cooling rate: smaller -> more aggressive cooling
    cooling_rate = 0.995  # try 0.99 if you want even more aggressive cooling

    for r in range(n_restarts):
        # --- Early exit: if interrupted, stop all remaining restarts ---
        if interrupt_event is not None and interrupt_event.is_set():
            print(f"[MAPPING] Interrupt received before restart {r}; "
                  f"returning best mapping found so far: cost={global_best_cost}")
            break

        # 1) Greedy initial mapping (already reasonably compact)
        init_mapping = greedy_initial_mapping(
            process_list, n_qubits, hardware_distance_pair
        )

        # 2) Build incremental cost state
        state = MappingCostState(process_list, init_mapping, n_qubits, DIST_MATRIX)
        current_cost = state.get_cost()

        # If we haven't recorded any global best yet, use this as a baseline
        if global_best_mapping is None or current_cost < global_best_cost:
            global_best_cost = current_cost
            global_best_mapping = dict(state.mapping)

        # Initial temperature scaled to cost magnitude
        T0 = max(1.0, abs(current_cost) * 0.1)

        for step in range(steps_per_restart):
            # --- Early exit inside restart: interrupt while annealing ---
            if interrupt_event is not None and interrupt_event.is_set():
                print(f"[MAPPING] Interrupt received during restart {r}, step {step}; "
                      f"returning best mapping found so far: cost={global_best_cost}")
                # We don't need to do anything else: global_best_* already
                # tracks the best we have seen across all restarts/steps.
                if global_best_mapping is None:
                    # Fallback: use current state's mapping if for some reason
                    # we never updated the global_best_mapping.
                    global_best_mapping = dict(state.mapping)
                    global_best_cost = current_cost
                # Break out of both loops via flag
                break

            # Geometric cooling schedule
            T = T0 * (cooling_rate ** step) + 1e-9  # avoid T == 0

            # 3) Propose a local move
            move = propose_move(state, move_prob=move_prob)
            if move is None:
                # No valid move (should be very rare)
                break

            phys_src, phys_dst, dst_was_used = move

            old_cost = current_cost

            # 4) Apply move
            state.reassign(phys_src, phys_dst)
            new_cost = state.get_cost()
            delta = new_cost - old_cost

            # 5) Acceptance rule (simulated annealing)
            if delta <= 0:
                accept = True
            else:
                accept = math.exp(-delta / T) > random.random()

            if accept:
                current_cost = new_cost

                # Update global best
                if new_cost < global_best_cost:
                    global_best_cost = new_cost
                    # Copy the current mapping (phys -> (pid, dq))
                    global_best_mapping = dict(state.mapping)
            else:
                # 6) Reject: revert the move
                if dst_was_used:
                    # Swap again restores original
                    state.reassign(phys_src, phys_dst)
                else:
                    # Move-to-helper was src->dst; revert by dst->src
                    state.reassign(phys_dst, phys_src)
                # current_cost stays old_cost

        else:
            # inner loop ended normally (no interrupt break)
            print(f"[Restart {r}] best so far: {global_best_cost}")
            continue  # go to next restart

        # If we hit the interrupt break, we land here and should break outer loop
        if interrupt_event is not None and interrupt_event.is_set():
            break

        print(f"[Restart {r}] best so far: {global_best_cost}")

    print("Final best cost:", global_best_cost)
    if global_best_mapping is None:
        # Should not happen unless there were no valid mappings
        raise RuntimeError("No valid mapping found during annealing.")
    return global_best_mapping



def load_ibm_api_key_from_file(filename: str) -> str:
    """
    Load the IBMQ API key from a file
    """
    with open(filename, "r") as f:
        api_key = f.read().strip()
    return api_key


APIKEY = load_ibm_api_key_from_file("apikey")

class jobManager:
    """
    This class manage the job and the job result.

    The job include:
    1. Manage the interface with IBMQ or simulator
    2. Decode the job result and distribution to process output
    """
    def __init__(self, ibmkey: str = APIKEY, use_simulator: bool = False):
        self._result_queue = queue.Queue()
        self._ibmkey = ibmkey
        self._use_simulator = use_simulator
        self._virtual_measurement_size = 1



    def execute_on_hardware(self, shots: int, measurement_size: int,measurement_to_process_map: Dict[int, int], scheduled_instructions: List[instruction]):
        """
        Send the scheduled instructions to hardware.
        Use the jobManager class to submit the job to IBMQ or simulator

        Input:
            measurement_to_process_map: Dict[int, int]
                The mapping from measurement index to process id

        Return:

            Result of the execution, which is a dictionary of all process results
            For example, if there are two processes in the batch, the result is:
            {
                process_id_1: {"result_key_1": result_value_1, ...},
                process_id_2: {"result_key_2": result_value_2, ...}
            }
        """
        self._virtual_measurement_size = measurement_size
        if self._use_simulator:
            # Wait for 2 second to simulate the job submission time
            time.sleep(7)
            raw_result = self.submit_job_to_simulator(shots, scheduled_instructions)
        else:
            raw_result = self.submit_job_to_ibmq(shots,scheduled_instructions)


        # print("Measurement to process map:", measurement_to_process_map)
        # print("Raw result from hardware/simulator:", raw_result)

        redistributed_result = self.redistribute_job_result(measurement_to_process_map, raw_result)
        return redistributed_result


    def submit_job_to_ibmq(self, shots: int,  scheduled_instructions: List[instruction]) -> Dict[str, int]:
        """
        Submit the scheduled instructions to IBMQ or simulator
        Get the raw result back
        
        Input:
            scheduled_instructions: List[instruction]
                The scheduled instruction list to be submitted
        """

        qiskit_circuit = construct_qiskit_circuit_for_hardware_instruction(self._virtual_measurement_size, scheduled_instructions)
        fake_hardware = fake_torino_backend
        initial_layout = [i for i in range(N_qubits)]  # logical i -> physical i

        transpiled = transpile(
            qiskit_circuit,
            backend= fake_hardware,
            initial_layout=initial_layout,
            optimization_level=3,
        )

        service = QiskitRuntimeService(channel="ibm_cloud",token=self._ibmkey)
        
        #backend = service.least_busy(simulator=False, operational=True)

        backend = service.backend("ibm_torino")

        # Convert to an ISA circuit and layout-mapped observables.
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        isa_circuit = pm.run(transpiled)
        
        # run SamplerV2 on the chosen backend
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots

        print("[Submitted] Job to IBMQ backend!", backend)


        job = sampler.run([isa_circuit])
        # Ensure it’s done so metrics/timestamps are populated
        job.wait_for_final_state()

         # --- results ---
        pub = job.result()[0]  # first (and only) PUB result
        counts = pub.join_data().get_counts()

        return counts


    def redistribute_job_result(self, measurement_to_process_map: Dict[int, int], raw_result: Dict[str, int]) -> Dict[int, Dict[str, int]]:
        """
        Redistribute the raw result to process output.

        For example, if the count result from hardware is:
        {
            "00": 500,
            "01": 300,
            "10": 200
        }
        Where the first bit(Reverse order) belongs to process 1, and the second bit(Reverse order) belongs to process 2.
        Then the ouput is:
        {
            process_id_1: {"0": 700, "1": 300},
            process_id_2: {"0": 800, "1": 200}
        }

        
        When count result from hardware is:
        {
            "001": 500,
            "010": 300,
            "101": 200
        }
        Where the first two bits(Reverse order) belongs to process 1, and the third bit(Reverse order) belongs to process 2.
        Then the ouput is:
        {
            process_id_1: {"01": 500, "10": 300,"01":200},
            process_id_2: {"0": 800, "1": 200}
        }


        Input:
            raw_result: Any
                The raw result returned from IBMQ or simulator
        """
        redistributed_result = {}
        for bitstring, count in raw_result.items():
            # Reverse the bitstring to match measurement order
            bitstring = bitstring[::-1]

            """
            First, we get the distributed set of bitstrings for all processes
            """
            proc_string_map = {}
            for meas_index, proc_id in measurement_to_process_map.items():
                if proc_id not in proc_string_map:
                    proc_string_map[proc_id] = bitstring[meas_index]
                else:
                    proc_string_map[proc_id] += bitstring[meas_index]
            """
            Reverse the bitstring for each process to match the original order
            """
            for proc_id in proc_string_map:
                proc_string_map[proc_id] = proc_string_map[proc_id][::-1]
            """
            Update the count for each process
            """
            for proc_id, proc_bitstring in proc_string_map.items():
                if proc_id not in redistributed_result:
                    redistributed_result[proc_id] = {}
                if proc_bitstring not in redistributed_result[proc_id]:
                    redistributed_result[proc_id][proc_bitstring] = count
                else:
                    redistributed_result[proc_id][proc_bitstring] += count


        return redistributed_result



    def submit_job_to_simulator(self, shots, scheduled_instructions)-> Dict[str, int]:
        """
        Submit the scheduled instructions to simulator

        Input:
            scheduled_instructions: List[instruction]
                The scheduled instruction list to be submitted
        """
        sim = AerSimulator()
        qiskit_circuit = construct_qiskit_circuit_for_hardware_instruction(self._virtual_measurement_size, scheduled_instructions)
        
        #qiskit_circuit.draw("mpl").savefig("simulated_circuit.png")
        
        tqc = transpile(qiskit_circuit, sim)
        # Run with 1000 shots
        result = sim.run(tqc, shots=shots).result()

        counts = result.get_counts(tqc)


        # print("result:", counts)
        return counts


class haloScheduler:
    """
    The scheduler in our quantum operating system.

    It should maintain a process Queue from the user
    """
    def __init__(self, use_simulator: bool = False):
        self._process_queue = []

        self._lock = threading.Lock()

        self._scheduled_job_queue = queue.Queue()
        self._in_flight_processes = set()
        self._execution_thread: Optional[threading.Thread] = None
        

        # the stop event
        self._start_time=0
        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None   
        # Log of the history
        self._log=[]
        #Store the process waiting time
        #The process fidelity
        #The total running time
        #The finished process count
        self._process_source_id={}
        self._process_start_time={}
        self._process_end_time={}
        self._process_waiting_time={}
        self._process_fidelity={}
        self._total_running_time=0.0
        self._finished_process_count=0
        self._jobmanager=jobManager(use_simulator=use_simulator)


        # Store the statistics in batch level
        self._num_process_in_batch = []
        self._qubit_utilization_in_batch = [] 
        self._batch_shared_ratio = []



        # NEW: event to interrupt the mapping / annealing algorithm
        self._mapping_interrupt_event = threading.Event()

    def store_log(self, filename: str):
        """
        Store the log to a file
        """
        with open(filename, 'w') as f:
            f.write("----------------------Log:---------------------------------\n")
            for entry in self._log:
                f.write(entry + '\n')
            """
            Also store the statistics in the same file:
            """
            f.write("------------------Statistics:---------------------------------")
            f.write(f"Total running time: {self._total_running_time}\n")
            f.write(f"Finished process count: {self._finished_process_count}\n")
            f.write("Throughput: \n")
            if self._total_running_time > 0:
                throughput = self._finished_process_count / self._total_running_time
                f.write(f"{throughput}\n")
            else:
                f.write("N/A\n")

            f.write("----------------------Batch statistics-----------------------:\n")
            # If they are lists:
            for i, (num_proc, qubit_util) in enumerate(
                zip(self._num_process_in_batch, self._qubit_utilization_in_batch)
            ):
                f.write(
                    f"Batch {i}: Number of processes: {num_proc}, "
                    f"Qubit utilization: {qubit_util}\n"
                )


            for i, shared_ratio in enumerate(self._batch_shared_ratio):
                f.write(f"Batch {i}: Shared ratio: {shared_ratio}\n")
            # Averages
            if self._batch_shared_ratio:
                average_shared_ratio = sum(self._batch_shared_ratio) / len(self._batch_shared_ratio)
            else:                
                average_shared_ratio = 0.0
            f.write(f"Average shared ratio per batch: {average_shared_ratio}\n")



            if self._num_process_in_batch and self._qubit_utilization_in_batch:
                average_num_process = sum(self._num_process_in_batch) / len(self._num_process_in_batch)
                average_qubit_utilization = (
                    sum(self._qubit_utilization_in_batch) / len(self._qubit_utilization_in_batch)
                )
            else:
                average_num_process = 0.0
                average_qubit_utilization = 0.0
            f.write(f"Average number of processes per batch: {average_num_process}\n")
            f.write(f"Average qubit utility per batch: {average_qubit_utilization}\n")

            f.write("----------------------Process waiting times-----------------------:\n")
            average_waiting_time = 0.0
            for process_id, waiting_time in self._process_waiting_time.items():
                f.write(f"Process {process_id}: {waiting_time}\n")
                average_waiting_time += waiting_time
            if self._process_waiting_time:
                average_waiting_time /= len(self._process_waiting_time)
            f.write(f"Average waiting time across all processes: {average_waiting_time}\n")

            f.write("----------------------Process fidelities-----------------------:\n")
            for process_id, fidelity in self._process_fidelity.items():
                f.write(f"Process {process_id}: {fidelity}\n")
            average_fidelity = sum(self._process_fidelity.values()) / len(self._process_fidelity) if self._process_fidelity else 0.0    
            f.write(f"Average fidelity across all processes: {average_fidelity}\n")




    def _execution_worker(self):
        """
        Consumer Thread: Pulls jobs, executes them, and updates state.
        Wrapped in try/except to prevent silent thread death.
        """
        print("[SYSTEM] Execution Worker started.")
        while not self._stop_event.is_set():
            try:
                # Wait for a job
                try:
                    job_data = self._scheduled_job_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Unpack
                shots, total_meas, meas_map, insts, process_ids = job_data
                
                try:
                    # --- EXECUTE (Crucial Step) ---
                    result = self._jobmanager.execute_on_hardware(shots, total_meas, meas_map, insts)
                    self._log.append(f"[HARDWARE RESULT] {result}")

                    # --- UPDATE STATE ---
                    with self._lock:
                        self.update_process_queue(shots, result)


                    # NEW: signal that a batch has finished, so any ongoing
                    # mapping/annealing for the next batch should stop asap.
                    self._mapping_interrupt_event.set()
                    # (We will clear this in allocate_data_qubit before
                    # starting the next anneal.)

                except Exception as e:
                    print(f"\n[CRITICAL ERROR] Execution Worker failed on batch {process_ids}: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    # --- CLEANUP (Must happen no matter what) ---
                    # Always remove these IDs from in-flight, or the program will never halt.
                    with self._lock:
                        for pid in process_ids:
                            self._in_flight_processes.discard(pid)
                    
                    self._scheduled_job_queue.task_done()

            except Exception as outer_e:
                print(f"[FATAL] Execution thread crashed completely: {outer_e}")
                break  


    def start(self):
        """
        Start the scheduling thread
        """
        if Scheduling_Option == SchedulingOptions.BASELINE_SEQUENTIAL:
            self._scheduler_thread = threading.Thread(target=self.base_line_sequential_scheduling, daemon=True)
        elif Scheduling_Option == SchedulingOptions.NO_SHARING:
            self._scheduler_thread = threading.Thread(target=self.halo_scheduling, daemon=True)
        elif Scheduling_Option == SchedulingOptions.HALO:
            self._scheduler_thread = threading.Thread(target=self.halo_scheduling, daemon=True)
        elif Scheduling_Option == SchedulingOptions.SHOT_UNAWARE:
            assert False, "Shot unaware scheduling not implemented yet."

        self._scheduler_thread.start()
        self._execution_thread = threading.Thread(target=self._execution_worker, daemon=True)
        self._execution_thread.start()


    def wait_until_done(self):
        print("[SYSTEM] Waiting for all jobs to complete...")
        last_print_time = 0
        
        while True:
            with self._lock:
                q_len = len(self._process_queue)
                f_len = len(self._in_flight_processes)
                j_empty = self._scheduled_job_queue.empty()
            
            # Exit condition
            if q_len == 0 and f_len == 0 and j_empty:
                print("\n[SYSTEM] All jobs completed. Queue empty. Flight empty.")
                break
            
            # Print status every 3 seconds so you aren't guessing
            if time.time() - last_print_time > 3:
                print(f"[WAITING] Queue: {q_len} | In-Flight: {f_len} | Jobs Pending: {not j_empty}")
                if f_len > 0:
                    print(f"   -> Waiting for PIDs: {list(self._in_flight_processes)}")
                last_print_time = time.time()
            
            time.sleep(0.5)

    def stop(self):
            """
            Stop the scheduling thread AND the execution thread.
            """
            print("[SYSTEM] Stopping threads...")
            self._stop_event.set()
            
            # 1. Join the Scheduler (Producer)
            if self._scheduler_thread is not None:
                self._scheduler_thread.join()
                print("[SYSTEM] Scheduler thread stopped.")

            # 2. Join the Executor (Consumer)
            if self._execution_thread is not None:
                # The execution thread might be stuck waiting on queue.get()
                # We send a dummy/None item to unblock it if it's waiting
                if self._execution_thread.is_alive():
                    # Only put if empty to avoid messing up valid jobs, 
                    # though the stop_event check in the worker handles this too.
                    pass 
                
                self._execution_thread.join()
                print("[SYSTEM] Execution thread stopped.")

            self._log.append(f"[STOP] Scheduling stopped at time {time.time()-self._start_time}.")


    def add_process(self, process, source_id: int = -1):
        """
        Add another process in the currecnt queue
        The source id is used to identify the circuit in the benchmark suit
        """
        self._process_queue.append(process)
        self._process_start_time[process.get_process_id()] = time.time()-self._start_time
        self._process_source_id[process.get_process_id()] = source_id
        self._log.append(f"[ADD] Process {process.get_process_id()} added to the queue at time {self._process_start_time[process.get_process_id()] }, shots: {process.get_remaining_shots()}.")



    def get_next_batch_baseline(self)-> Optional[Tuple[int, List[process]]]:
        """
        Get the next process batch for baseline scheduling.
        Just return the next process in the queue.
        Return: List[process]
        Just need to make sure the total data qubits in the batch is less than N_qubits
        """
        if len(self._process_queue) == 0:
            return None
        process = self._process_queue[0]
        if process.get_num_data_qubits() > int(N_qubits * data_hardware_ratio):
            assert False, f"Process {process.get_process_id()} requires {process.get_num_data_qubits()} data qubits, which exceeds the maximum allowed {int(N_qubits * data_hardware_ratio)}."
        return process





    def get_next_batch(self, force_return: bool = False, shot_awareness: bool= True) -> Optional[Tuple[int, List[process]]]:
        """
        Get the next process batch.
        Return: List[process]
        Just need to make sure the total data qubits in the batch is less than N_qubits

        1. Try to fill up the data qubit Zone as much as possible
        2. If the force_return is True, return the batch even when the total data qubit usage is less than 50%
        3. If the total data qubit usage is less than 50%, return None

        """
        with self._lock: # Protect the read
            if len(self._process_queue) == 0:
                return None
            
            # Filter out processes that are currently being executed
            available_processes = [
                p for p in self._process_queue 
                if p.get_process_id() not in self._in_flight_processes
            ]
            
            if not available_processes:
                return None

            total_utility=0.0
            used_qubits=0
            remain_qubits=int(N_qubits*data_hardware_ratio)
            process_batch=[]

            for proc in available_processes:
                if proc.get_num_data_qubits()<=remain_qubits:
                    process_batch.append(proc)
                    used_qubits+=proc.get_num_data_qubits()
                    remain_qubits-=proc.get_num_data_qubits()
                else:
                    continue

    
            """
            Only return the batch when the total data qubit usage is 100%
            """
            total_utility=used_qubits/(N_qubits*data_hardware_ratio)
            print("Total utility of selected batch:", total_utility)


            # if total_utility<=1 and not force_return:
            #     return None
            min_shots= min([proc.get_remaining_shots() for proc in process_batch])  
            self._log.append(f"[BATCH] Process batch with {[proc.get_process_id() for proc in process_batch]} selected with min shots {min_shots}.")
        return min_shots,process_batch
    


    def allocate_data_qubit(self, process_batch: List[process]) -> Dict[int, tuple[int, int]]:
        """
        Allocate data qubit territory for all processes in the batch

        Return a Dict[int, tuple[int, int]] represent the data qubit mapping
        """
        self._mapping_interrupt_event.clear()
        """
        For the very first batch, we are waiting to nothing so we can do less restarts
        """
        n_restarts = 30
        best_mapping=iteratively_find_the_best_mapping_for_data(process_batch,N_qubits,n_restarts=n_restarts,interrupt_event=self._mapping_interrupt_event)
        return best_mapping
    



    def sequential_scheduling(self, L: Dict[int, tuple[int, int]], process_batch: List[process])-> Tuple[Dict[int, int], List[instruction]]:
        """
        Sequentially schedule instructions for all processes in the batch.
        We just run one process after another.
        Thus is helpful for debugging and baseline comparison.
        Return:
            scheduled_instructions: List[instruction]
                The scheduled instruction list
            Also, all instructions in the process are updated with helper qubit qubit assignment
        """
        final_scheduled_instructions=[]
        measurement_to_process_map = {}
        for proc in process_batch:

            scheduled_insts, is_finished, meas_to_proc_map = proc.schedule_all_instructions(L)
            final_scheduled_instructions.extend(scheduled_insts)
            measurement_to_process_map.update(meas_to_proc_map)
        return measurement_to_process_map, final_scheduled_instructions



    def scheduling_without_sharing(self, L: Dict[int, tuple[int, int]], process_batch: List[process])-> Tuple[Dict[int, int], List[instruction]]:
        """
        Not sharing helper qubits across processes
        Here L stands for the mapping of all qubits, including helper qubits.
        """
        pass




    def dynamic_helper_scheduling(self, L: Dict[int, tuple[int, int]], process_batch: List[process])-> Tuple[float,float, int, Tuple[Dict[int, int], List[instruction]]]:
        """
        Dynamically assigne helper qubits and schedule instructions
        Return:

            utilization: float
                The qubit utilization in this batch, which is defined as the total data qubit usage divided by total qubit number

            total_measurement_size: int
                The total measurement size across all processes

            scheduled_instructions: List[instruction]
                The scheduled instruction list
            Also, all instructions in the process are updated with helper qubit qubit assignment
        """
        final_scheduled_instructions=[]
        num_finished_process=0 
        process_finish_map = {i: False for i in process_batch}

        used_qubit = set(L.keys())
        utility = 0.0

        """
        Reform the mapping L to per process form
        """
        process_data_qubit_map = {proc.get_process_id(): {} for proc in process_batch}
        for phy, (proc_id, data_qubit) in L.items():
            process_data_qubit_map[proc_id][data_qubit] = phy


        current_measurement_index=0
        measurement_to_process_map = {}


        #All the remaining qubits are helper qubits
        #The helper qubit can be taken, but will also be released after used.
        #TODO: Not only assign the nearest helper qubits, but also bound the maximum distance between data qubit and helper qubit
        helper_qubit_zone = {phys for phys in range(N_qubits)} - set(L.keys())
        helper_qubit_available = {phys: True for phys in helper_qubit_zone}

        helper_qubit_used_count = {phys: 0 for phys in helper_qubit_zone}
        #Store the current helper qubit assignment for each process:
        #For example, {1:{"s0":3, "s1":5}, 2:{"s0":10, "s1":12}}
        all_proc_id = {proc.get_process_id() for proc in process_batch}
        current_helper_qubit_map = {proc_id: {} for proc_id in all_proc_id}
        num_available_helper_qubits = len(helper_qubit_zone)


        # prev_scheduled_instruction_count=0
        while num_finished_process < len(process_batch):
            """
            Round robin style instruction scheduling.

            Take turn to each process in the batch to schedule its next instruction.
            Assign helper qubits

            Make sure progress must be made in each round, otherwise deadlock happens.
            """
            # print("Scheduling round robin, finished processes:", num_finished_process)
            # print("Helper qubit available:", num_available_helper_qubits)
            # print("final_scheduled_instructions", len(final_scheduled_instructions))
            # print(final_scheduled_instructions)

            # if len(final_scheduled_instructions) == prev_scheduled_instruction_count:
            #     print("No progress in scheduling, deadlock!")
            #     count+=1
            
            # if count>100:
            #     break
            for proc in process_batch:
                if process_finish_map[proc]:
                    continue
                current_proc_id = proc.get_process_id()
                next_inst = proc.get_next_instruction()
                # --- DEBUGGING BLOCK END ---
                """
                If the process is already finished, update the finish map
                Release all data qubits
                Put these qubits back to the helper qubit pool
                """
                if proc.is_finished():
                    process_finish_map[proc] = True
                    num_finished_process += 1
                    all_data_qubit_addresses = [x for x in L.keys() if L[x][0] == current_proc_id]
                    #Release all helper qubits assigned to this process
                    #Add reset instruction for each released helper qubit
                    for phy in all_data_qubit_addresses:
                        if phy in current_helper_qubit_map[current_proc_id]:
                            helper_qubit_available[phy] = True
                            num_available_helper_qubits += 1                            
                            final_scheduled_instructions.append(instruction(Instype.RESET, qubitaddress=[], reset_address=phy))
                    continue

                """
                No matter what is the next instruction, we need to update the scheduled mapped address for all data qubits
                """
                next_inst = proc.get_next_instruction()
                all_data_qubit_addresses = next_inst.get_all_data_qubit_addresses()
                all_helper_qubit_addresses = next_inst.get_all_helper_qubit_addresses()
                for d_addr in all_data_qubit_addresses:
                    d_index=int(d_addr[1:])
                    next_inst.set_scheduled_mapped_address(d_addr, process_data_qubit_map[current_proc_id][d_index])
                for h_addr in all_helper_qubit_addresses:
                    if h_addr in current_helper_qubit_map[current_proc_id]:
                        next_inst.set_scheduled_mapped_address(h_addr, current_helper_qubit_map[current_proc_id][h_addr])


                """
                Get the next instruction from the current process
                We have several cases:
                1. The instruction is a measurement, we need to record the measurement to process mapping
                2. The instruction is a release helper qubit instruction, we need to release the helper qubits
                3. The instruction needs no helper qubit at all, just schedule it directly
                4. The instruction needs helper qubits, there are also enough helper qubits available, assign them and schedule the instruction
                5. The instruction needs helper qubits, but there are not enough helper qubits available, skip this process this round
                """    


                """
                Case1: Measurement instruction
                We need to record the measurement to process mapping
                For example, measurement address 0 -> process id 1, measurement address 1 -> process id 2
                The mapping is:
                measurement_to_process_map:  {0: 1, 1: 2}

                We need to initialize a new measurement instruction
                """
                if next_inst.is_measurement():
                    """
                    If might be possible that user directly measure a helper qubit!
                    In that case, we also have to assign the scheduled mapped address for the helper qubit
                    """
                    helper_qubit_address = next_inst.get_all_helper_qubit_addresses()

                    if len(helper_qubit_address)==1:
                        num_needed_helper_qubits = 0
                        h_addr=helper_qubit_address[0]
                        if h_addr not in current_helper_qubit_map[current_proc_id]:
                            num_needed_helper_qubits = 1


                        if num_needed_helper_qubits==1:
                            if num_available_helper_qubits==0:
                                proc.set_status(ProcessStatus.WAIT_FOR_HELPER)
                                continue
                            available_helper_qubit_list=[phys for phys, available in helper_qubit_available.items() if available]
                            selected_helper_qubit = available_helper_qubit_list[0]
                            next_inst.set_scheduled_mapped_address(h_addr, selected_helper_qubit)
                            current_helper_qubit_map[current_proc_id][h_addr]=selected_helper_qubit
                            helper_qubit_available[selected_helper_qubit]=False
                            used_qubit.add(selected_helper_qubit)
                            helper_qubit_used_count[selected_helper_qubit] += 1

                            
                            
                    next_inst.set_scheduled_classical_address(current_measurement_index)
                    final_scheduled_instructions.append(next_inst)
                    measurement_to_process_map[current_measurement_index] = current_proc_id
                    current_measurement_index += 1
                    proc.execute_next_instruction()
                    continue


                """
                Case2: A deallocating helper qubit instruction
                Add a reset instruction to release the helper qubit!
                """
                if next_inst.is_release_helper_qubit():
                    helper_qubit_address = next_inst.get_all_helper_qubit_addresses()
                    for h_addr in helper_qubit_address:
                        if h_addr in current_helper_qubit_map[current_proc_id]:
                            phys = current_helper_qubit_map[current_proc_id][h_addr]
                            helper_qubit_available[phys] = True
                            helper_qubit_zone.add(phys)
                            num_available_helper_qubits += 1
                            del current_helper_qubit_map[current_proc_id][h_addr]
                            final_scheduled_instructions.append(instruction(Instype.RESET, qubitaddress=[], reset_address=phys))
                    proc.execute_next_instruction()
                    continue



                helper_qubit_needed=next_inst.get_helper_qubit_count()
                """
                Case3: No helper qubit needed. all qubits are data qubits
                We should update the scheduled mapped address for all data qubits
                """
                if helper_qubit_needed==0:
                    final_scheduled_instructions.append(next_inst)
                    proc.execute_next_instruction()
                    continue



                all_helper_qubit=next_inst.get_all_helper_qubit_addresses()
                # print("All helper qubit needed:", all_helper_qubit)
                # print("Current helper qubit map:", current_helper_qubit_map[current_proc_id])
                # print("Current inst:", next_inst)
                """
                Case3.5: Helper qubit needed but already assigned.
                But need to update the instruction with the current mapping
                """
                for h_addr in list(all_helper_qubit):
                    if h_addr not in current_helper_qubit_map[current_proc_id].keys():
                        # print("Helper qubit not assigned yet:", h_addr)
                        continue
                    else:
                        # print("Helper qubit already assigned:", h_addr)
                        all_helper_qubit.remove(h_addr)
                        helper_qubit_needed-=1

                if helper_qubit_needed==0:
                    for h_addr in all_helper_qubit:
                        next_inst.set_scheduled_mapped_address(current_helper_qubit_map[current_proc_id][h_addr])    
                    final_scheduled_instructions.append(next_inst)
                    proc.execute_next_instruction()
                    continue

                # print(f"DEBUG: Proc {current_proc_id} | Inst: {next_inst} | Helpers Needed: {helper_qubit_needed}")
                #print("Current helper qubit map:", current_helper_qubit_map)
                """
                Case4: Helper qubit needed and available.
                Find the nearest available helper qubits.

                Two subcases here:
                4.1: The gate has one data qubit only, and another helper qubit
                4.2: The gate has two helper qubits
                """

                availble_helper_qubit_list=[phys for phys, available in helper_qubit_available.items() if available]

                if helper_qubit_needed<= num_available_helper_qubits:
                    # print("Have enough helper qubits available.")
                    topology=proc.get_topology()
                    data_qubit_physical_addresses = process_data_qubit_map[current_proc_id]
                    #Assign helper qubits
                    data_qubit_addresses = next_inst.get_all_data_qubit_addresses()
                    """
                    Case 4.1: We find the nearest helper qubit for the specific data qubit
                    """
                    if helper_qubit_needed==1:
                        helper_qubit_index=int(all_helper_qubit[0][1:])


                        selected_helper_qubit = topology.best_helper_qubit_location(hardware_distance=hardware_distance_pair,
                                                                                  data_qubit_mapping=data_qubit_physical_addresses,
                                                                                  available_helper_qubits=availble_helper_qubit_list,
                                                                                  helper_qubit_index=helper_qubit_index)
                        helper_qubit_used_count[selected_helper_qubit] += 1


                        #Assign the helper qubit
                        if selected_helper_qubit is not None:

                            #Update the used qubit set
                            used_qubit.add(selected_helper_qubit)

                            next_inst.set_scheduled_mapped_address(all_helper_qubit[0],selected_helper_qubit)
                            current_helper_qubit_map[current_proc_id][all_helper_qubit[0]]=selected_helper_qubit
                            final_scheduled_instructions.append(next_inst)
                            helper_qubit_available[selected_helper_qubit]=False
                            num_available_helper_qubits-=1
                            proc.execute_next_instruction()
                            continue
                        else:
                            raise ValueError("No available helper qubit found, but should be available.")


                    """
                    This is a rare case, but is not impossible

                    For example, when preparing a cat state, we need add gates on two helper qubits first,
                    before any data qubit is involved.

                    Case 4.2: We find the nearest set of helper qubits to the entire process data qubit territory
                    """
                    if len(data_qubit_addresses)==0 and helper_qubit_needed==2:
                        helper_qubit_index1=int(all_helper_qubit[0][1:])
                        selected_helper_qubit_1 = topology.best_helper_qubit_location(hardware_distance=hardware_distance_pair,
                                                                                  data_qubit_physical_addresses=data_qubit_physical_addresses,
                                                                                  available_helper_qubits=availble_helper_qubit_list,
                                                                                  helper_qubit_index=helper_qubit_index1)
                        
                        helper_qubit_used_count[selected_helper_qubit_1] += 1
                        used_qubit.add(selected_helper_qubit_1)
                        current_helper_qubit_map[current_proc_id][all_helper_qubit[0]]=selected_helper_qubit_1
                        next_inst.set_scheduled_mapped_address(all_helper_qubit[0],selected_helper_qubit_1)

                        helper_qubit_available[selected_helper_qubit_1]=False
                        num_available_helper_qubits-=1
                        availble_helper_qubit_list.remove(selected_helper_qubit_1)


                        helper_qubit_index2=int(all_helper_qubit[1][1:])
                        selected_helper_qubit_2 = topology.best_helper_qubit_location(hardware_distance=hardware_distance_pair,
                                                                                  data_qubit_physical_addresses=data_qubit_physical_addresses,
                                                                                  available_helper_qubits=availble_helper_qubit_list,
                                                                                  helper_qubit_index=helper_qubit_index2)
                        

                        used_qubit.add(selected_helper_qubit_2)
                        helper_qubit_used_count[selected_helper_qubit_2] += 1
                        current_helper_qubit_map[current_proc_id][all_helper_qubit[1]]=selected_helper_qubit_2
                        next_inst.set_scheduled_mapped_address(all_helper_qubit[1],selected_helper_qubit_2)


                        helper_qubit_available[selected_helper_qubit_2]=False
                        num_available_helper_qubits-=1


                        final_scheduled_instructions.append(next_inst)
                        proc.execute_next_instruction()
                        continue

                    assert False, "Unhandled case in helper qubit assignment."

                # print("Not enough helper qubits available!!!")
                """
                Case4: Helper qubit needed and not available. The process has to wait
                """
                proc.set_status(ProcessStatus.WAIT_FOR_HELPER)


                # Release helper qubits

        if current_measurement_index==0:
            for proc in process_batch:
                print(f"[ERROR] Process {proc.get_process_id()} has instructions but no measurement scheduled.")
            for proc in process_batch:
                print("Instructions:")
                for inst in final_scheduled_instructions:
                    print(inst)
            raise ValueError("No measurement scheduled in this batch, something is wrong.")



        utility = len(used_qubit)/N_qubits


        """
        Calculate the number of shared helper qubits
        """

        shared_helper_qubits = sum(1 for count in helper_qubit_used_count.values() if count > 1)
        shared_ratio = shared_helper_qubits / len(helper_qubit_zone) if len(helper_qubit_zone) > 0 else 0.0
        
        
        #Verify all instructions have scheduled mapped address
        for inst in final_scheduled_instructions:
            if not inst.is_scheduled():
                raise ValueError(f"Instruction {inst} has no scheduled mapped address for all qubits.")
        
        
        
        return shared_ratio,utility, current_measurement_index, measurement_to_process_map, final_scheduled_instructions



    def show_queue(self, add_to_log: bool = True):
        """
        Show the current process queue. 

        For example, process 1: num_data_qubits=3, num_helper_qubits=2, ramaining_shots=100,  process 2: num_data_qubits=3, num_helper_qubits=2, ramaining_shots=500

        Then print:
            [Process Queue] P1: DQ=3, HQ=2, Shots=100----P2: DQ=3, HQ=2, Shots=500
        """
        temp_list = []
        tmp_str = "[PROCESS QUEUE] "
        for proc in self._process_queue:
            temp_list.append(proc)
            tmp_str += f"P{proc.get_process_id()}: DQ={proc.get_num_data_qubits()}, HQ={proc.get_num_helper_qubits()}, Shots={proc._remaining_shots}----"

        if add_to_log:
            self._log.append(f"[QUEUE STATUS] {tmp_str}")


    def base_line_sequential_scheduling(self):
        """
        The baseline scheduling algorithm.
        1) Update the process queue
        2) Get the next process, allocate qubit for that single process
        3) Schedule all instructions for that single process
        4) Send the scheduled instructions to hardware
        5) Update the process queue after one process execution
        """
        start_time=time.time()
        self._start_time=start_time
        while not self._stop_event.is_set() or not len(self._process_queue)==0:


            # Step 1: Get the next batch of processes
            if len(self._process_queue) == 0:
                continue

            next_proc=self._process_queue[0]

            shots = next_proc.get_remaining_shots()

            if shots ==0:
                continue


            # Step 2: Allocate data qubit territory for all processes
            L=self.allocate_data_qubit([next_proc])

            # Step 2.5: Update the data qubit mapping in each process
            # for proc in process_batch:
            #     proc.update_data_qubit_mapping(L)


            # Step 3: Dynamically assign helper qubits and schedule instructions
            shared_ratio,utility ,total_measurements,measurement_to_process_map, scheduled_instructions = self.dynamic_helper_scheduling(L,[next_proc])

            self._batch_shared_ratio.append(shared_ratio)
            self._qubit_utilization_in_batch.append(utility)
            self._num_process_in_batch.append(1)
            self._log.append(f"[UTILITY] New batch has utility {utility}.")
            self._log.append(f"[PROCESS COUNT] New batch has 1 processes.")
            self._log.append(f"[SHARING RATIO] New batch has helper qubit sharing ratio {shared_ratio}.")

            # Step 4: Send the scheduled instructions to hardware
            result=self._jobmanager.execute_on_hardware(shots,total_measurements,measurement_to_process_map,scheduled_instructions)


            self._log.append(f"[HARDWARE RESULT] {result}")
            # Step 5: Update the process queue after one batch execution
            self.update_process_queue(shots,result)


            #Clear the queue
            #self._process_queue.remove(next_proc)


            print("[FINISH] Finish current process.")

            self.show_queue(add_to_log=True)
        end_time=time.time()
        self._total_running_time=end_time-start_time


    def halo_scheduling(self):
        """
        The main scheduling algorithm (PRODUCER).
        Improved version:
          - The scheduler only compiles a new batch when either:
              (1) It has been spinning for more than `max_spinning_time`, OR
              (2) The queue is rich enough that we can fill a large fraction
                  of the physical data-qubit space (target_batch_utilization).
        """
        start_time = time.time()
        self._start_time = start_time

        # --- Tunable knobs ---
        max_spinning_time = 40.0      # seconds: condition (1)
        target_batch_utilization = 0.8  # e.g. 80% of data-qubit zone: condition (2)

        is_spinning = False
        spinning_start_time = 0.0

        while (
            not self._stop_event.is_set()
            or len(self._process_queue) > 0
            or len(self._in_flight_processes) > 0
        ):
            # ------------------------------------------------------------------
            # 0. Fast exit if no queued work but jobs are still running.
            # ------------------------------------------------------------------
            with self._lock:
                queue_empty = (len(self._process_queue) == 0)
                in_flight = len(self._in_flight_processes)



            if queue_empty:
                # Nothing new to schedule; reset spinning and wait for arrivals
                is_spinning = False
                time.sleep(0.01)
                continue

            # ------------------------------------------------------------------
            # 1. Estimate the *best possible* data-qubit utilization
            #    we could achieve from the current queue if we packed optimally.
            # ------------------------------------------------------------------
            with self._lock:
                # Only consider processes that are NOT currently in flight.
                available_processes = [
                    p for p in self._process_queue
                    if p.get_process_id() not in self._in_flight_processes
                ]

            if not available_processes:
                # All processes already in flight; just wait for them to finish.
                is_spinning = False
                time.sleep(1)

                print("[SCHEDULER] All processes are currently in flight; waiting.")
                continue

            # Capacity of the data-qubit region
            data_qubit_capacity = int(N_qubits * data_hardware_ratio)
            if data_qubit_capacity <= 0:
                raise ValueError("Data-qubit capacity is non-positive; check N_qubits and data_hardware_ratio.")

            # Greedy upper-bound: sort processes by size (descending) and pack
            # until we hit the capacity. This approximates the *maximum* util.
            # This is only used for the decision "is it worth scheduling now?"
            total_data_qubits = 0
            for proc in sorted(available_processes, key=lambda p: p.get_num_data_qubits(), reverse=True):
                dq = proc.get_num_data_qubits()
                if dq > data_qubit_capacity:
                    # This process alone is too big; get_next_batch will assert on it.
                    continue
                if total_data_qubits + dq > data_qubit_capacity:
                    continue
                total_data_qubits += dq

            potential_utilization = total_data_qubits / data_qubit_capacity if data_qubit_capacity > 0 else 0.0

            # ------------------------------------------------------------------
            # 2. Decide whether we are allowed to schedule the next batch:
            #    - Condition (2): potential_utilization >= target_batch_utilization
            #    - Condition (1): spinning_time >= max_spinning_time
            # ------------------------------------------------------------------
            now = time.time()

            # Condition (2): enough stuff in the queue to "fill the machine"
            ready_due_to_util = (potential_utilization >= target_batch_utilization)

            # Condition (1): we have spun for too long waiting for more jobs
            if not is_spinning and not ready_due_to_util:
                # Start timing a spin interval
                is_spinning = True
                spinning_start_time = now

            ready_due_to_spin = False
            if is_spinning:
                spinning_time = now - spinning_start_time
                if spinning_time >= max_spinning_time:
                    ready_due_to_spin = True

            # If neither condition is met, keep waiting and DO NOT schedule yet.
            if not ready_due_to_util and not ready_due_to_spin:
                time.sleep(0.1)
                continue

            # At this point we are allowed to schedule:
            #   - If only "spin" condition fired, we will force a batch even if
            #     utilization isn't great (to avoid starvation).
            #   - If util condition fired, we try to get a high-utility batch.
            force_return = ready_due_to_spin and not ready_due_to_util

            if force_return:
                print(f"[SCHEDULER] Forcing batch due to spin timeout after {spinning_time:.2f}s.")
            else:
                print(f"[SCHEDULER] Scheduling new batch")
            batchresult = self.get_next_batch(force_return=force_return)

            if batchresult is None:
                print("[SCHEDULER] Could not form a valid batch yet. Keep spinning / waiting.")
                # Could not form a valid batch yet. Keep spinning / waiting.
                # (This can happen due to get_next_batch's internal 50% util threshold.)
                if not is_spinning:
                    is_spinning = True
                    spinning_start_time = time.time()
                time.sleep(0.01)
                continue

            # We successfully formed a batch; reset spinning state.
            is_spinning = False

            shots, process_batch = batchresult
            if len(process_batch) == 0 or shots == 0:
                # Nothing meaningful to run; just continue.
                print("[SCHEDULER] Empty batch returned; continuing.")
                time.sleep(0.01)
                continue

            # ------------------------------------------------------------------
            # 3. Mark processes as in-flight so they won't be picked again
            #    until the Execution Worker updates their remaining shots.
            # ------------------------------------------------------------------
            with self._lock:
                for proc in process_batch:
                    self._in_flight_processes.add(proc.get_process_id())

            # Clear any old mapping state in the processes
            for proc in process_batch:
                proc.reset_all_mapping()

            # ------------------------------------------------------------------
            # 4. Allocate Data Qubit Territory
            # ------------------------------------------------------------------
            L = self.allocate_data_qubit(process_batch)

            print(f"[SCHEDULER] Allocated data qubits for batch {[p.get_process_id() for p in process_batch]}: {L}")

            # ------------------------------------------------------------------
            # 5. Dynamic Helper Scheduling (Compilation)
            # ------------------------------------------------------------------
            shared_ratio,utility, total_measurements, measurement_to_process_map, scheduled_instructions = \
                self.dynamic_helper_scheduling(L, process_batch)

            self._batch_shared_ratio.append(shared_ratio)
            print(f"[SCHEDULER] Compiled batch {[p.get_process_id() for p in process_batch]} with utility {utility:.3f} and {total_measurements} measurements.")

            self._qubit_utilization_in_batch.append(utility)
            self._num_process_in_batch.append(len(process_batch))
            self._log.append(f"[UTILITY] New batch has utility {utility}.")
            self._log.append(f"[PROCESS COUNT] New batch has {len(process_batch)} processes.")
            self._log.append(f"[SHARING RATIO] New batch has helper qubit sharing ratio {shared_ratio}.")
            # ------------------------------------------------------------------
            # 6. Dispatch to Execution Queue (Non-Blocking)
            # ------------------------------------------------------------------
            job_package = (
                shots,
                total_measurements,
                measurement_to_process_map,
                scheduled_instructions,
                [p.get_process_id() for p in process_batch],
            )
            self._scheduled_job_queue.put(job_package)

            # Loop repeats; we do NOT immediately grab a new batch anymore.
            # We will only do so again once conditions (1) or (2) are met.

        # End of main loop
        end_time = time.time()
        self._total_running_time = end_time - start_time
        print(f"[SCHEDULER] Scheduling thread finished. Total active time: {self._total_running_time:.2f}s")


    # def halo_scheduling(self):
    #     """
    #     The main scheduling algorithm (PRODUCER).
    #     It decouples compilation from execution. 
    #     It compiles batches and pushes them to a job queue for the background thread.
    #     """
    #     start_time = time.time()
    #     self._start_time = start_time

    #     previous_spinning_start_time = 0
    #     is_spinning = False
    #     max_spinning_time = 1  # Seconds to wait before forcing a batch

    #     # Loop until stopped AND no work remains
    #     # We check _in_flight_processes to ensure we don't exit while jobs are running
    #     while not self._stop_event.is_set() or len(self._process_queue) > 0 or len(self._in_flight_processes) > 0:

    #         # ---------------------------------------------------------
    #         # Step 1: Get the next batch of processes
    #         # ---------------------------------------------------------
            
    #         # We use a Lock because the Consumer thread is modifying _process_queue (removing finished jobs)
    #         # and _in_flight_processes (removing completed IDs)
    #         with self._lock:
    #             # If queue is empty but jobs are running, we just wait.
    #             if len(self._process_queue) == 0:
    #                 time.sleep(0.01)
    #                 continue

    #         # Determine if we should force a batch return
    #         force = False
    #         if is_spinning:
    #             spinning_time = time.time() - previous_spinning_start_time
    #             if spinning_time >= max_spinning_time:
    #                 force = True
            
    #         # Get batch (Note: get_next_batch must be updated to ignore _in_flight_processes)
    #         batchresult = self.get_next_batch(force_return=force)



    #         if batchresult is None:
    #             if not is_spinning:
    #                 previous_spinning_start_time = time.time()
    #             is_spinning = True
    #             # CRITICAL: Sleep briefly to prevent CPU burn while waiting for the 
    #             # Consumer thread to finish jobs and free up resources.
    #             time.sleep(0.01) 
    #             continue

    #         # We have a batch!
    #         is_spinning = False
    #         shots, process_batch = batchresult

    #         if len(process_batch) == 0:
    #             continue

    #         self._num_process_in_batch.append(len(process_batch) if batchresult else 0)            

    #         for proc in process_batch:
    #             proc.reset_all_mapping()

    #         # ---------------------------------------------------------
    #         # Step 1.5: Mark processes as "In Flight"
    #         # ---------------------------------------------------------
    #         # We must mark them NOW so the next loop iteration doesn't pick them up
    #         with self._lock:
    #             for proc in process_batch:
    #                 self._in_flight_processes.add(proc.get_process_id())

    #         # ---------------------------------------------------------
    #         # Step 2: Allocate Data Qubit Territory
    #         # ---------------------------------------------------------
    #         L = self.allocate_data_qubit(process_batch)

    #         # ---------------------------------------------------------
    #         # Step 3: Dynamic Helper Scheduling (Compilation)
    #         # ---------------------------------------------------------
    #         # print(f"[SCHEDULER] Compiling batch {[p.get_process_id() for p in process_batch]}...")
            
    #         utility, total_measurements, measurement_to_process_map, scheduled_instructions = \
    #             self.dynamic_helper_scheduling(L, process_batch)


    #         self._qubit_utilization_in_batch.append(utility)
    #         self._num_process_in_batch.append(len(process_batch))
    #         self._log.append(f"[UTILITY] New batch has utility {utility}.")
    #         self._log.append(f"[PROCESS COUNT] New batch has {len(process_batch)} processes.")

    #         # ---------------------------------------------------------
    #         # Step 4: Dispatch to Execution Queue (Non-Blocking)
    #         # ---------------------------------------------------------
    #         # Instead of waiting for hardware, we bundle the job and push it to the queue.
    #         # The Execution Worker thread will handle the waiting.
            
    #         job_package = (
    #             shots, 
    #             total_measurements, 
    #             measurement_to_process_map, 
    #             scheduled_instructions, 
    #             [p.get_process_id() for p in process_batch] # Pass IDs for tracking
    #         )
            
    #         self._scheduled_job_queue.put(job_package)
            
    #         # print(f"[SCHEDULER] Pushed batch to Execution Queue. Queue Size: {self._scheduled_job_queue.qsize()}")

    #         # ---------------------------------------------------------
    #         # Step 5: Reset Mappings for Next Compilation
    #         # ---------------------------------------------------------
    #         # We reset the mappings here so the Process objects are clean for the 
    #         # next time they are picked up (after the Consumer thread updates their shots).


    #         # LOOP RESTART: Immediately go back to Step 1 to schedule the next batch!
            
        
    #     # End of Loop
    #     end_time = time.time()
    #     self._total_running_time = end_time - start_time
    #     print(f"[SCHEDULER] Scheduling thread finished. Total active time: {self._total_running_time:.2f}s")


    # def halo_scheduling(self):
    #     """
    #     The main scheduling algorithm. There are multiple steps:
    #     1) Get the next batch of processes
    #     2) Allocate data qubit territory for all processes
    #     3) Dynamically assign helper qubits and schedule instructions
    #     4) Send the scheduled instructions to hardware
    #     5) Update the process queue after one batch execution
    #     6) Repeat until the process queue is empty
    #     """
    #     start_time=time.time()
    #     self._start_time=start_time

    #     """
    #     Keep track of how long the scheduler has been spinning without doing any work
    #     We need to make sure the scheduler can force return a batch after a long spinning time
    #     """
    #     previous_spinning_start_time=0
    #     is_spinning=False
    #     max_spinning_time=1  # Set a default max spinning time (in seconds)
    #     while not self._stop_event.is_set() or not len(self._process_queue)==0:


    #         # Step 1: Get the next batch of processes
    #         # If the next_batch doesn't use enough qubits, wait for the next round
    #         #print("Starting to get the next batch...")
    #         if is_spinning:
    #             spinning_time=time.time()-previous_spinning_start_time
    #             if spinning_time>=max_spinning_time:
    #                 batchresult=self.get_next_batch(force_return=True)
    #         else:
    #             batchresult=self.get_next_batch()

    #         if batchresult is None:
    #             if not is_spinning:
    #                 previous_spinning_start_time=time.time()
    #             is_spinning=True
    #             continue

    #         shots, process_batch = batchresult

    #         if shots ==0:
    #             break

    #         if len(process_batch) == 0:
    #             if not is_spinning:
    #                 previous_spinning_start_time=time.time()
    #             is_spinning=True
    #             continue


    #         is_spinning=False


    #         # Step 2: Allocate data qubit territory for all processes
    #         L=self.allocate_data_qubit(process_batch)

    #         # Step 2.5: Update the data qubit mapping in each process
    #         # for proc in process_batch:
    #         #     proc.update_data_qubit_mapping(L)


    #         # Step 3: Dynamically assign helper qubits and schedule instructions
    #         print(f"[BATCH START] Scheduling batch with processes {[proc.get_process_id() for proc in process_batch]} for {shots} shots.")
    #         total_measurements,measurement_to_process_map, scheduled_instructions = self.dynamic_helper_scheduling(L,process_batch)
    #         print(f"[BATCH END] Finished scheduling batch with processes {[proc.get_process_id() for proc in process_batch]} for {shots} shots.")
    #         # Step 4: Send the scheduled instructions to hardware
    #         result=self._jobmanager.execute_on_hardware(shots,total_measurements,measurement_to_process_map,scheduled_instructions)


    #         self._log.append(f"[HARDWARE RESULT] {result}")
    #         # Step 5: Update the process queue after one batch execution
    #         self.update_process_queue(shots,result)


    #         # Clear the queue
    #         for proc in list(self._process_queue):  # iterate over a shallow copy
    #             if proc.finish_all_shots():
    #                 self._process_queue.remove(proc)


    #         # Step 6: Reset for the next batch
    #         for proc in process_batch:
    #             proc.reset_all_mapping()

    #         print("[FINISH] BATCHFINISH.")

    #         self.show_queue(add_to_log=True)
    #     end_time=time.time()
    #     self._total_running_time=end_time-start_time

    def update_process_queue(self, shots: int ,result: Dict[int, Dict[str, int]]):
        """
        Update the process queue after one batch execution.
        
        NOTE: This function is called by the Execution Thread. 
        It assumes the caller holds self._lock to protect self._process_queue modification.
        """

        """
        First, update the count stored in each process
        1) Find the process in the current process queue
        2) Update the process result with the result from hardware
        """
        # We iterate over the results returned by hardware
        for proc_id, proc_result in result.items():
            # Find the matching process object in our queue
            for proc in self._process_queue:
                if proc.get_process_id() == proc_id:
                    proc.update_result(shots, proc_result)
                    break # Optimization: Found the process, move to next result

        """
        Second, check if the process is finished
        If finished, remove it from the process queue
        Also, calculate the waiting time and fidelity for statistics
        """
        # CRITICAL FIX: Iterate over list(self._process_queue) to create a copy.
        # We cannot iterate over the list while removing items from it simultaneously.
        for proc in list(self._process_queue):
            if proc.finish_all_shots():

                print(f"[PROCESS FINISH] Process {proc.get_process_id()} finished all shots!")
                
                # Safe to remove now because we are iterating over a copy
                self._process_queue.remove(proc)

                # Update process waiting time
                current_time = time.time()
                self._process_end_time[proc.get_process_id()] = current_time - self._start_time
                self._process_waiting_time[proc.get_process_id()] = self._process_end_time[proc.get_process_id()] - self._process_start_time[proc.get_process_id()]

                # Update the process fidelity
                benchmark_id = self._process_source_id[proc.get_process_id()]
                
                # Check if we have an ID to load ideal results (benchmarks usually have IDs >= 0)
                if benchmark_id >= 0:
                    try:
                        ideal_result = load_ideal_count_output(benchmark_Option, benchmark_id)
                        self._process_fidelity[proc.get_process_id()] = distribution_fidelity(ideal_result, proc.get_result_counts())
                    except Exception as e:
                        print(f"[WARNING] Could not calculate fidelity for P{proc.get_process_id()}: {e}")
                        self._process_fidelity[proc.get_process_id()] = 0.0
                else:
                    self._process_fidelity[proc.get_process_id()] = 0.0

                print(f"[PROCESS FIDELITY] Process {proc.get_process_id()} fidelity: {self._process_fidelity[proc.get_process_id()]}")
                print(f"[PROCESS WAITING TIME] Process {proc.get_process_id()} waiting time: {self._process_waiting_time[proc.get_process_id()]} seconds")

                self._log.append(f"[PROCESS FINISH] Process {proc.get_process_id()} finished all shots at time {self._process_end_time[proc.get_process_id()]}.")
                self._log.append(f"[PROCESS FIDELITY] Process {proc.get_process_id()} fidelity: {self._process_fidelity[proc.get_process_id()]}")
                self._log.append(f"[PROCESS WAITING TIME] Process {proc.get_process_id()} waiting time: {self._process_waiting_time[proc.get_process_id()]} seconds")
                
                self._finished_process_count += 1

    # def update_process_queue(self, shots: int ,result: Dict[int, Dict[str, int]]):
    #     """
    #     Update the process queue after one batch execution.
    #     The result has a clear form such as:
    #     {
    #         process_id_1: {"result_key_1": result_value_1, ...},
    #         process_id_2: {"result_key_2": result_value_2, ...}
    #     }
    #     """


    #     """
    #     First, update the count stored in each process
    #     1) Find the process in the current process queue
    #     2) Update the process result with the result from hardware
    #     """
    #     for proc_id, proc_result in result.items():
    #         for proc in self._process_queue:
    #             if proc.get_process_id() == proc_id:
    #                 proc.update_result(shots, proc_result)


    #     """
    #     Second, check if the process is finished
    #     If finished, remove it from the process queue
    #     Also, calculate the waiting time and fidelity for statistics
    #     """
    #     for proc in self._process_queue:
    #         if proc.finish_all_shots():

    #             print(f"[PROCESS FINISH] Process {proc.get_process_id()} finished all shots!")
    #             self._process_queue.remove(proc)


    #             # Update process waiting time
    #             self._process_end_time[proc.get_process_id()] = time.time()-self._start_time
    #             self._process_waiting_time[proc.get_process_id()] = self._process_end_time[proc.get_process_id()] - self._process_start_time[proc.get_process_id()]


    #             # Update the process fidelity
    #             benchmark_id = self._process_source_id[proc.get_process_id()]
    #             ideal_result = load_ideal_count_output(benchmark_Option,benchmark_id)
    #             self._process_fidelity[proc.get_process_id()] = distribution_fidelity(ideal_result, proc.get_result_counts())
    #             print(f"[PROCESS FIDELITY] Process {proc.get_process_id()} fidelity: {self._process_fidelity[proc.get_process_id()]}")
    #             print(f"[PROCESS WAITING TIME] Process {proc.get_process_id()} waiting time: {self._process_waiting_time[proc.get_process_id()]} seconds")

    #             self._log.append(f"[PROCESS FINISH] Process {proc.get_process_id()} finished all shots at time {self._process_end_time[proc.get_process_id()]}.")
    #             self._log.append(f"[PROCESS FIDELITY] Process {proc.get_process_id()} fidelity: {self._process_fidelity[proc.get_process_id()]}")
    #             self._log.append(f"[PROCESS WAITING TIME] Process {proc.get_process_id()} waiting time: {self._process_waiting_time[proc.get_process_id()]} seconds")
    #             self._finished_process_count += 1




def construct_qiskit_circuit_for_hardware_instruction(num_measurements:int, instruction_list: List[instruction]) -> QuantumCircuit:
    """
    Construct a qiskit circuit from the instruction list.
    Also help to visualize the circuit.
    """
    dataqubit = QuantumRegister(N_qubits, "q")


    # Classical registers (optional, if you want measurements)
    classicalbits = ClassicalRegister(num_measurements, "c")

    # Combine them into one circuit
    qiskit_circuit = QuantumCircuit(dataqubit, classicalbits)

    for inst in instruction_list:
        if inst.is_system_call():
            continue
        addresses = inst.get_qubitaddress()
        qiskitaddress=[]
        for addr in addresses:
            phy_index = inst.get_scheduled_mapped_address(addr)
            qiskitaddress.append(dataqubit[phy_index])

        match inst.get_type():
            case Instype.H:
                qiskit_circuit.h(qiskitaddress[0])
            case Instype.X:
                qiskit_circuit.x(qiskitaddress[0]) 
            case Instype.Y:
                qiskit_circuit.y(qiskitaddress[0]) 
            case Instype.Z:
                qiskit_circuit.z(qiskitaddress[0])  
            case Instype.T:
                qiskit_circuit.t(qiskitaddress[0])
            case Instype.Tdg:
                qiskit_circuit.tdg(qiskitaddress[0])
            case Instype.S:
                qiskit_circuit.s(qiskitaddress[0])
            case Instype.Sdg:
                qiskit_circuit.sdg(qiskitaddress[0])
            case Instype.SX:
                qiskit_circuit.sx(qiskitaddress[0])
            case Instype.RZ:
                params=inst.get_params()
                qiskit_circuit.rz(params[0], qiskitaddress[0])
            case Instype.RX:
                params=inst.get_params()
                qiskit_circuit.rx(params[0], qiskitaddress[0])
            case Instype.RY:
                params=inst.get_params()
                qiskit_circuit.ry(params[0], qiskitaddress[0])
            case Instype.U3:
                params=inst.get_params()
                qiskit_circuit.u(params[0], params[1], params[2], qiskitaddress[0])
            case Instype.U:
                params=inst.get_params()
                qiskit_circuit.u(params[0], params[1], params[2], qiskitaddress[0])
            case Instype.Toffoli:
                qiskit_circuit.ccx(qiskitaddress[0], qiskitaddress[1], qiskitaddress[2])
            case Instype.CNOT:
                qiskit_circuit.cx(qiskitaddress[0], qiskitaddress[1])
            case Instype.CH:
                qiskit_circuit.ch(qiskitaddress[0], qiskitaddress[1])
            case Instype.SWAP:
                qiskit_circuit.swap(qiskitaddress[0], qiskitaddress[1])
            case Instype.CSWAP:
                qiskit_circuit.cswap(qiskitaddress[0], qiskitaddress[1], qiskitaddress[2])
            case Instype.CP:
                params=inst.get_params()
                qiskit_circuit.cp(params[0], qiskitaddress[0], qiskitaddress[1])
            case Instype.RESET:
                qiskit_circuit.reset(dataqubit[inst.get_reset_address()])
            case Instype.MEASURE:
                scheduled_classical_address=inst.get_scheduled_classical_address()
                qiskit_circuit.measure(qiskitaddress[0], scheduled_classical_address)

    return qiskit_circuit



def random_arrival_generator(scheduler: haloScheduler,
                             arrival_rate: float = 1,
                             max_time: float = 100.0,
                             share_qubit: bool = True):
    """
    Producer thread: generate processes according to a Poisson process
    (exponential inter-arrival times with mean 1/arrival_rate).

    All processes are generated from one benchmark suit if the option is not mixed.

    TODO: Support mixed benchmark suit in the future. Just need to randomly select a benchmark suit for each arrival.
    """
    start = time.time()  
    pid = 0
    while time.time() - start < max_time:
        # Wait random time until next arrival
        wait = random.expovariate(arrival_rate)  # mean 1/lambda
        time.sleep(wait)

        # Generate a random process

        #shots = random.choice(np.arange(500, 2500, 100))
        shots = 1000
        #Generate a random process from benchmark suit
        if benchmark_Option!=benchmarktype.MIX:
            benchmark_suit = benchmark_type_to_benchmark[benchmark_Option]
            benchmark_root_path = benchmark_file_path[benchmark_Option]
            benchmark_id = random.randint(0, len(benchmark_suit) - 1)
        else:
            benchmark_suit = random.choice(list(benchmark_type_to_benchmark.keys()))
            benchmark_root_path = benchmark_file_path[benchmark_suit]
            benchmark_id = random.randint(0, len(benchmark_suit) - 1)

            
        proc = generate_process_from_benchmark(benchmark_root_path, benchmark_suit, benchmark_id, pid, shots, share_qubit=share_qubit)
        print(f"[ARRIVAL] New process {benchmark_suit[benchmark_id]} arriving, pid: {pid}, shots: {shots}")
        scheduler.add_process(proc, source_id=benchmark_id)
        pid += 1

    print("[STOP] Finished generating processes.")
    



def generate_process_from_benchmark(benchmark_root_path:str,benchmark_suit: Dict[int,str], benchmark_id: int, pid: int, shots: int, share_qubit=True) -> process:
    """
    Generate a process from the benchmark suit, given the pid and shots
    1. Load the benchmark data from the file
    2. Create a process instance
    3. Return the process instance
    """
    if not share_qubit:
        file_path = f"{benchmark_root_path}nohelper//{benchmark_suit[benchmark_id]}"
    else:
        file_path = f"{benchmark_root_path}{benchmark_suit[benchmark_id]}"
    # Load the benchmark data from the file
    (inst_list, data_n, syn_n, measure_n)=parse_program_from_file(file_path)
    proc = process(pid, data_n, syn_n, shots, inst_list)
    return proc



def test_scheduling():
    """
    A simple test function for the haloScheduler
    """
    process1 = generate_process_from_benchmark(0, 1, 100)
    process2 = generate_process_from_benchmark(1, 2, 200)
    process3 = generate_process_from_benchmark(2, 3, 300)
    process4 = generate_process_from_benchmark(3, 4, 100)
    process5 = generate_process_from_benchmark(4, 5, 200)
    process6 = generate_process_from_benchmark(5, 6, 300)


    print("Start testing haloScheduler...")
    scheduler = haloScheduler()
    scheduler.add_process(process1)
    scheduler.add_process(process2)
    scheduler.add_process(process3)
    scheduler.add_process(process4)
    scheduler.add_process(process5)
    scheduler.add_process(process6)

 

    print("Start getting next batch...")
    shots, next_batch = scheduler.get_next_batch()


    
    L = scheduler.allocate_data_qubit(next_batch)


    plot_process_schedule_on_torino(
        torino_coupling_map(),
        next_batch,
        L,
        out_png="best_torino_mapping_6proc.png",
    )

    utility, total_measurements, measurement_to_process_map, scheduled_instructions = scheduler.dynamic_helper_scheduling(L, next_batch)


    print("Instructions:")
    for inst in scheduled_instructions:
        print(inst)


    qiskit_circuit = construct_qiskit_circuit_for_hardware_instruction(total_measurements, scheduled_instructions)



    



# if __name__ == "__main__":
#     test_scheduling()








# if __name__ == "__main__":

#     random.seed(42)


#     haloScheduler_instance=haloScheduler(use_simulator=False)
#     haloScheduler_instance.start()


#     producer_thread = threading.Thread(
#         target=random_arrival_generator,
#         args=(haloScheduler_instance, 0.4, 25.0, True),
#         daemon=False
#     )
#     producer_thread.start()


#     simulation_time = 40  # seconds
#     time.sleep(simulation_time)


#     # Wait for producer to finish generating all processes
#     producer_thread.join()


#     haloScheduler_instance.stop()
#     print("Simulation finished.")


#     haloScheduler_instance.store_log("halo_multi_thread.txt")


def run_experiment_general(log_path: str, lambda_: float, simulation_time: float = 40.0, seed: int = 42):
    """
    The generate function for running experiments with different parameters
    """

    random.seed(seed)
    print("[MAIN] Starting Scheduler...")
    haloScheduler_instance = haloScheduler(use_simulator=False)
    haloScheduler_instance.start()


    share_helper_qubit = True
    match Scheduling_Option:
        case SchedulingOptions.NO_SHARING :
            share_helper_qubit = False
        case SchedulingOptions.BASELINE_SEQUENTIAL:
            share_helper_qubit = False
        case _:
            share_helper_qubit = True

    # 2. Start Producer (Process Generator)
    print("[MAIN] Starting Producer...")
    producer_thread = threading.Thread(
        target=random_arrival_generator,
        args=(haloScheduler_instance, lambda_, simulation_time, share_helper_qubit),
        daemon=False
    )
    producer_thread.start()

    # 3. Wait for Simulation Time
    print(f"[MAIN] Sleeping for simulation time...")
    try:
        # We loop with short sleeps so we can interrupt with Ctrl+C if needed
        for _ in range(int(simulation_time / 10)): # 4 * 10s = 40s
            time.sleep(10)
            print(f"[MAIN] ... Simulation still running ...")
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user!")

    # 4. Wait for Producer to finish generating
    print("[MAIN] Waiting for Producer thread to join...")
    producer_thread.join()
    print("[MAIN] Producer finished.")

    # 5. Wait for Scheduler to finish processing the backlog
    # (This is where it looked stuck before - now it will tell you what it's doing)
    print("[MAIN] Waiting for Scheduler to clear the job queue...")
    haloScheduler_instance.wait_until_done()

    # 6. Stop Threads cleanly
    print("[MAIN] Stopping Scheduler threads...")
    haloScheduler_instance.stop()

    # 7. Save Log
    print("[MAIN] Saving log...")
    haloScheduler_instance.store_log(log_path)
    print(f"[MAIN] Done. Log saved to {log_path}")




def run_experiment_on_multiX_small():
    """
    Run an experiment with multiple controlled-X benchmark on small hardware

    We should run baseline, with helper, without helper qubit, and without shot-ware scheduling versions for comparison

    The result log should be stored to the corresponding folder
    """
    # 1. Setup the algorithm

    global Scheduling_Option
    global benchmark_Option


    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    benchmark_Option = benchmarktype.MULTI_CONTROLLED_X_SMALL
    log_path = "resultlog/multiXsmall/halo.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the baseline sequential scheduling
    # Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    # benchmark_Option = benchmarktype.MULTI_CONTROLLED_X_SMALL
    # log_path = "resultlog/multiXsmall/baseline_sequential.txt"
    # run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the no helper qubit scheduling
    Scheduling_Option = SchedulingOptions.NO_SHARING
    benchmark_Option = benchmarktype.MULTI_CONTROLLED_X_SMALL
    log_path = "resultlog/multiXsmall/no_helper.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)




def run_experiment_on_multiX_medium():
    """
    Run an experiment with multiple controlled-X benchmark on medium hardware
    """
    # 1. Setup the algorithm

    global Scheduling_Option
    global benchmark_Option


    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    benchmark_Option = benchmarktype.MULTI_CONTROLLED_X_MEDIUM
    log_path = "resultlog/multiXmedium/halo.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the baseline sequential scheduling
    Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    benchmark_Option = benchmarktype.MULTI_CONTROLLED_X_MEDIUM
    log_path = "resultlog/multiXmedium/baseline_sequential.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the no helper qubit scheduling
    Scheduling_Option = SchedulingOptions.NO_SHARING
    benchmark_Option = benchmarktype.MULTI_CONTROLLED_X_MEDIUM
    log_path = "resultlog/multiXmedium/no_helper.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)





def run_experiment_on_qec_small():
    """
    Run an experiment with QEC benchmark on small hardware
    """
    # 1. Setup the algorithm

    global Scheduling_Option
    global benchmark_Option


    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    benchmark_Option = benchmarktype.STABILIZER_MEASUREMENT_SMALL
    log_path = "resultlog/qecsmall/halo.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    # #Run the baseline sequential scheduling
    # Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    # benchmark_Option = benchmarktype.STABILIZER_MEASUREMENT_SMALL
    # log_path = "resultlog/qecsmall/baseline_sequential.txt"
    # run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the no helper qubit scheduling
    # Scheduling_Option = SchedulingOptions.NO_SHARING
    # benchmark_Option = benchmarktype.STABILIZER_MEASUREMENT_SMALL
    # log_path = "resultlog/qecsmall/no_helper.txt"
    # run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)




def run_experiment_on_qec_medium():
    """
    Run an experiment with QEC benchmark on medium hardware
    """
    # 1. Setup the algorithm

    global Scheduling_Option
    global benchmark_Option


    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    benchmark_Option = benchmarktype.STABILIZER_MEASUREMENT_MEDIUM
    log_path = "resultlog/qecmedium/halo.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the baseline sequential scheduling
    Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    benchmark_Option = benchmarktype.STABILIZER_MEASUREMENT_MEDIUM
    log_path = "resultlog/qecmedium/baseline_sequential.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the no helper qubit scheduling
    Scheduling_Option = SchedulingOptions.NO_SHARING
    benchmark_Option = benchmarktype.STABILIZER_MEASUREMENT_MEDIUM
    log_path = "resultlog/qecmedium/no_helper.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)





def run_experiment_on_random_small():
    """
    Run an experiment with random benchmark on small hardware
    """
    # 1. Setup the algorithm

    global Scheduling_Option
    global benchmark_Option


    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    benchmark_Option = benchmarktype.RANDOM_SMALL
    log_path = "resultlog/randomsmall/halo.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the baseline sequential scheduling
    # Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    # benchmark_Option = benchmarktype.RANDOM_SMALL
    # log_path = "resultlog/randomsmall/baseline_sequential.txt"
    # run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the no helper qubit scheduling
    Scheduling_Option = SchedulingOptions.NO_SHARING
    benchmark_Option = benchmarktype.RANDOM_SMALL
    log_path = "resultlog/randomsmall/no_helper.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)





def run_experiment_on_random_medium():
    # 1. Setup the algorithm

    global Scheduling_Option
    global benchmark_Option


    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    benchmark_Option = benchmarktype.RANDOM_MEDIUM
    log_path = "resultlog/randommedium/halo.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the baseline sequential scheduling
    Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    benchmark_Option = benchmarktype.RANDOM_MEDIUM
    log_path = "resultlog/randommedium/baseline_sequential.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    #Run the no helper qubit scheduling
    Scheduling_Option = SchedulingOptions.NO_SHARING
    benchmark_Option = benchmarktype.RANDOM_MEDIUM
    log_path = "resultlog/randommedium/no_helper.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)





def run_experiment_on_arithmetic_small():
    global Scheduling_Option
    global benchmark_Option


    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    benchmark_Option = benchmarktype.CLASSICAL_LOGIC_SMALL
    log_path = "resultlog/arithsmall/halo.txt"  
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)


    Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    log_path = "resultlog/arithsmall/baseline_sequential.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)

    #Run the no helper qubit scheduling
    Scheduling_Option = SchedulingOptions.NO_SHARING
    log_path = "resultlog/arithsmall/no_helper.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)




def run_experiment_on_arithmetic_medium():
    global Scheduling_Option
    global benchmark_Option

    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    benchmark_Option = benchmarktype.CLASSICAL_LOGIC_MEDIUM
    log_path = "resultlog/arithmedium/halo.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)

    Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    log_path = "resultlog/arithmedium/baseline_sequential.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)

    #Run the no helper qubit scheduling
    Scheduling_Option = SchedulingOptions.NO_SHARING
    log_path = "resultlog/arithmedium/no_helper.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=40.0, seed=42)





def run_experiment_on_mix():
    global Scheduling_Option
    global benchmark_Option

    benchmark_Option = benchmarktype.MIX


    #Run the HALO scheduling first
    Scheduling_Option = SchedulingOptions.HALO
    log_path = "resultlog/mix/halo.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=80.0, seed=42)


    Scheduling_Option = SchedulingOptions.BASELINE_SEQUENTIAL
    log_path = "resultlog/mix/baseline_sequential.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=80.0, seed=42)


    #Run the no helper qubit scheduling
    Scheduling_Option = SchedulingOptions.NO_SHARING
    log_path = "resultlog/mix/no_helper.txt"
    run_experiment_general(log_path=log_path, lambda_=0.6, simulation_time=80.0, seed=42)











def run_all_experiments():
    run_experiment_on_multiX_small()
    run_experiment_on_multiX_medium()
    run_experiment_on_qec_small()
    run_experiment_on_qec_medium()
    run_experiment_on_random_small()
    run_experiment_on_random_medium()
    run_experiment_on_arithmetic_small()
    run_experiment_on_arithmetic_medium()
    run_experiment_on_mix()





if __name__ == "__main__":
    
    # run_experiment_on_random_medium()
    # run_experiment_on_multiX_small()
    # run_experiment_on_qec_small()

    #run_experiment_on_multiX_medium()
    #run_experiment_on_random_small()
    run_experiment_on_qec_medium()

    #run_experiment_on_arithmetic_small()


    #run_experiment_on_random_medium()
    # import sys

    # # 1. Setup
    # random.seed(42)
    # print("[MAIN] Starting Scheduler...")
    # haloScheduler_instance = haloScheduler(use_simulator=False)
    # haloScheduler_instance.start()

    # # 2. Start Producer (Process Generator)
    # print("[MAIN] Starting Producer...")
    # producer_thread = threading.Thread(
    #     target=random_arrival_generator,
    #     args=(haloScheduler_instance, 0.6, 35.0, True),
    #     daemon=False
    # )
    # producer_thread.start()

    # # 3. Wait for Simulation Time
    # print(f"[MAIN] Sleeping for simulation time...")
    # try:
    #     # We loop with short sleeps so we can interrupt with Ctrl+C if needed
    #     for _ in range(4): # 4 * 10s = 40s
    #         time.sleep(10)
    #         print(f"[MAIN] ... Simulation still running ...")
    # except KeyboardInterrupt:
    #     print("\n[MAIN] Interrupted by user!")

    # # 4. Wait for Producer to finish generating
    # print("[MAIN] Waiting for Producer thread to join...")
    # producer_thread.join()
    # print("[MAIN] Producer finished.")

    # # 5. Wait for Scheduler to finish processing the backlog
    # # (This is where it looked stuck before - now it will tell you what it's doing)
    # print("[MAIN] Waiting for Scheduler to clear the job queue...")
    # haloScheduler_instance.wait_until_done()

    # # 6. Stop Threads cleanly
    # print("[MAIN] Stopping Scheduler threads...")
    # haloScheduler_instance.stop()

    # # 7. Save Log
    # print("[MAIN] Saving log...")
    # haloScheduler_instance.store_log("no_helper.txt")
    # print("[MAIN] Done. Log saved to no_helper.txt")