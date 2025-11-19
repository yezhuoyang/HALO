from hardwares import torino_coupling_map
from process import all_pairs_distances, process
from typing import Dict, List
import random
import math
import queue


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




class haloScheduler:
    """
    The scheduler in our quantum operating system.

    It should maintain a process Queue from the user
    """
    def __init__(self):
        self._process_queue = queue.Queue()




    def add_process(self, process):
        """
        Add another process in the currecnt queue
        """
        self._process_queue.put(process)




    def get_next_batch(self):
        """
        Get the next process batch
        """
        pass



    def allocate_data_qubit(self):
        """
        Allocate data qubit territory for all processes
        """
        pass



    def dynamic_helper_scheduling(self):
        """
        Dynamically assigne helper qubits and schedule instructions
        """
        pass



    def halo_scheduling(self):
        """
        The main scheduling algorithm. There are multiple steps:
        1) Get the next batch of processes
        2) Allocate data qubit territory for all processes
        3) Dynamically assigne helper qubits and schedule instructions
        4) Send the scheduled instructions to hardware
        5) Update the process queue after one batch execution
        6) Repeat until the process queue is empty
        """
        pass



    def send_to_hardware(self):
        """
        Send the scheduled instructions to hardware

        Return:

            Result of the execution
        """
        pass



    def update_process_queue(self):
        """
        Update the process queue after one batch execution
        """
        pass




class jobManager:
    """
    This class manage the job and the job result.

    The job include:
    1. Manage the interface with IBMQ or simulator
    2. Decode the job result and distribution to process output
    """
    def __init__(self):
        self._result_queue = queue.Queue()


    def submit_job_to_ibmq(self, scheduled_instructions):
        """
        Submit the scheduled instructions to IBMQ or simulator

        Input:
            scheduled_instructions: List[instruction]
                The scheduled instruction list to be submitted
        """
        pass



    def submit_job_to_simulator(self, scheduled_instructions):
        """
        Submit the scheduled instructions to simulator

        Input:
            scheduled_instructions: List[instruction]
                The scheduled instruction list to be submitted
        """
        pass

    