import json
from hardwares import torino_coupling_map
from instruction import instruction, Instype, parse_program_from_file
from process import all_pairs_distances, plot_process_schedule_on_torino, process, ProcessStatus
from typing import Dict, List, Optional, Tuple
import random
import math
import queue
import threading
import time



benchmark_suit_file_root="C://Users//yezhu//Documents//HALO//benchmark//"
benchmark_suit={
    0:"cat_state_prep_n4",
    1:"cat_state_verification_n4",
    2:"repetition_code_distance3_n3",
    3:"shor_parity_measurement_n4",
    4:"shor_stabilizer_XZZX_n3",
    5:"shor_stabilizer_ZZZZ_n4",
    6:"syndrome_extraction_surface_n3"
}





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



alpha=0.3
beta=100
gamma=300
delta=200
N_qubits=133
hardware_distance_pair=all_pairs_distances(N_qubits, torino_coupling_map())
data_hardware_ratio=0.8
#The maximum distance between a data qubit and a helper qubit
max_data_helper_distance=10

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
                                      n_restarts: int = 100,
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
            f.write("Average waiting time per process:\n")
            if self._finished_process_count > 0:
                avg_waiting_time = self._total_running_time / self._finished_process_count
                f.write(f"{avg_waiting_time}\n")
            else:
                f.write("N/A\n")
            f.write("Throughput: \n")
            if self._total_running_time > 0:
                throughput = self._finished_process_count / self._total_running_time
                f.write(f"{throughput}\n")
            else:
                f.write("N/A\n")
            f.write("Process average fidelities:\n")
            for process_id, fidelity in self._process_fidelity.items():
                f.write(f"Process {process_id}: {fidelity}\n")


    def start(self):
        """
        Start the scheduling thread
        """
        self._scheduler_thread = threading.Thread(target=self.halo_scheduling, daemon=True)
        self._scheduler_thread.start()
    

    def stop(self):
        """
        Stop the scheduling thread
        """
        self._stop_event.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join()
        self._log.append(f"[STOP] Scheduling stopped at time {self._total_running_time}.")


    def add_process(self, process, source_id: int = -1):
        """
        Add another process in the currecnt queue
        The source id is used to identify the circuit in the benchmark suit
        """
        self._process_queue.put(process)
        self._process_start_time[process.get_process_id()] = time.time()-self._start_time
        self._process_source_id[process.get_process_id()] = source_id
        self._log.append(f"[ADD] Process {process.get_process_id()} added to the queue at time {self._process_start_time[process.get_process_id()] }.")



    def get_next_batch(self)-> List[process]:
        """
        Get the next process batch.
        Return: List[process]
        Just need to make sure the total data qubits in the batch is less than N_qubits
        """
        if self._process_queue.empty():
            return None
        remain_qubits=int(N_qubits*data_hardware_ratio)
        process_batch=[]
        for proc in list(self._process_queue.queue):
            if proc.get_num_data_qubits()<=remain_qubits:
                process_batch.append(proc)
                remain_qubits-=proc.get_num_data_qubits()
            else:
                self._process_queue.put(proc)
                break
        return process_batch
    


    def allocate_data_qubit(self, process_batch: List[process]) -> Dict[int, tuple[int, int]]:
        """
        Allocate data qubit territory for all processes in the batch

        Return a Dict[int, tuple[int, int]] represent the data qubit mapping
        """
        best_mapping=iteratively_find_the_best_mapping(process_batch,N_qubits)



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
        """
        pass




    def dynamic_helper_scheduling(self, L: Dict[int, tuple[int, int]], process_batch: List[process])-> Tuple[Dict[int, int], List[instruction]]:
        """
        Dynamically assigne helper qubits and schedule instructions
        Return:
            scheduled_instructions: List[instruction]
                The scheduled instruction list
            Also, all instructions in the process are updated with helper qubit qubit assignment
        """
        final_scheduled_instructions=[]
        num_finished_process=0 
        process_finish_map = {i: False for i in process_batch}



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


        #Store the current helper qubit assignment for each process:
        #For example, {1:{"s0":3, "s1":5}, 2:{"s0":10, "s1":12}}
        all_proc_id = {proc.get_process_id() for proc in process_batch}
        current_helper_qubit_map = {proc_id: {} for proc_id in all_proc_id}
        num_available_helper_qubits = len(helper_qubit_zone)

        while num_finished_process < len(process_batch):
            """
            Round robin style instruction scheduling.

            Take turn to each process in the batch to schedule its next instruction.
            Assign helper qubits
            """
            for proc in process_batch:
                if process_finish_map[proc]:
                    continue
                current_proc_id = proc.get_process_id()
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
                for d_addr in all_data_qubit_addresses:
                    d_index=int(d_addr[1:])
                    next_inst.set_scheduled_mapped_address(d_addr, process_data_qubit_map[current_proc_id][d_index])


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
                """
                if next_inst.is_measurement():
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

                """
                Case3.5: Helper qubit needed but already assigned.
                But need to update the instruction with the current mapping
                """
                for h_addr in all_helper_qubit:
                    if h_addr not in current_helper_qubit_map[current_proc_id]:
                        break
                    else:
                        all_helper_qubit.remove(h_addr)
                        helper_qubit_needed-=1

                if helper_qubit_needed==0:
                    for h_addr in all_helper_qubit:
                        next_inst.set_scheduled_mapped_address(current_helper_qubit_map[current_proc_id][h_addr])    
                    final_scheduled_instructions.append(next_inst)
                    proc.execute_next_instruction()
                    continue


                """
                Case4: Helper qubit needed and available.
                Find the nearest available helper qubits.

                Two subcases here:
                4.1: The gate has one data qubit only, and another helper qubit
                4.2: The gate has two helper qubits
                """


                if helper_qubit_needed< num_available_helper_qubits:
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
                                                                                  available_helper_qubits=helper_qubit_available,
                                                                                  helper_qubit_index=helper_qubit_index)
                        


                        #Assign the helper qubit
                        if selected_helper_qubit is not None:
                            next_inst.set_scheduled_mapped_address(all_helper_qubit[0],selected_helper_qubit)
                            final_scheduled_instructions.append(next_inst)
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
                                                                                  available_helper_qubits=helper_qubit_available,
                                                                                  helper_qubit_index=helper_qubit_index1)

                        next_inst.set_scheduled_mapped_address(all_helper_qubit[0],selected_helper_qubit_1)
                        helper_qubit_index2=int(all_helper_qubit[1][1:])
                        selected_helper_qubit_2 = topology.best_helper_qubit_location(hardware_distance=hardware_distance_pair,
                                                                                  data_qubit_physical_addresses=data_qubit_physical_addresses,
                                                                                  available_helper_qubits=helper_qubit_available,
                                                                                  helper_qubit_index=helper_qubit_index2)
                        next_inst.set_scheduled_mapped_address(all_helper_qubit[1],selected_helper_qubit_2)
                        final_scheduled_instructions.append(next_inst)
                        proc.execute_next_instruction()
                        continue


                """
                Case4: Helper qubit needed and not available. The process has to wait
                """
                process.set_status(ProcessStatus.WAIT_FOR_HELPER)


                # Release helper qubits

                
        return measurement_to_process_map, final_scheduled_instructions



    def show_queue(self):
        """
        Show the current process queue. 

        For example, process 1: num_data_qubits=3, num_helper_qubits=2, ramaining_shots=100,  process 2: num_data_qubits=3, num_helper_qubits=2, ramaining_shots=500

        Then print:
            [Process Queue] P1: DQ=3, HQ=2, Shots=100----P2: DQ=3, HQ=2, Shots=500
        """
        temp_list = []
        queue_size = self._process_queue.qsize()
        for _ in range(queue_size):
            proc = self._process_queue.get()
            temp_list.append(proc)
            print(f"P{proc.get_process_id()}: DQ={proc.get_num_data_qubits()}, HQ={proc.get_num_helper_qubits()}, Shots={proc._shots}", end="----")
        for proc in temp_list:
            self._process_queue.put(proc)




    def halo_scheduling(self):
        """
        The main scheduling algorithm. There are multiple steps:
        1) Get the next batch of processes
        2) Allocate data qubit territory for all processes
        3) Dynamically assign helper qubits and schedule instructions
        4) Send the scheduled instructions to hardware
        5) Update the process queue after one batch execution
        6) Repeat until the process queue is empty
        """
        start_time=time.time()
        self._start_time=start_time
        while not self._stop_event.is_set() or not self._process_queue.empty():
            # Step 1: Get the next batch of processes
            process_batch = self.get_next_batch()
            # if not process_batch:
            #     continue

            # Step 2: Allocate data qubit territory for all processes
            L=self.allocate_data_qubit(process_batch)

            # Step 2.5: Update the data qubit mapping in each process
            for proc in process_batch:
                proc.update_data_qubit_mapping(L)


            # Step 3: Dynamically assign helper qubits and schedule instructions
            measurement_to_process_map, scheduled_instructions = self.dynamic_helper_scheduling(L,process_batch)

            # Step 4: Send the scheduled instructions to hardware
            result=self.execute_on_hardware(measurement_to_process_map,scheduled_instructions)


            # Step 5: Update the process queue after one batch execution
            self.update_process_queue(result)

            time.sleep(1)  # small delay to prevent busy waiting
            print("[FINISH] BATCHFINISH.")

            self.show_queue()
        end_time=time.time()
        self._total_running_time=end_time-start_time


    def execute_on_hardware(self, measurement_to_process_map: Dict[int, int], scheduled_instructions: List[instruction]):
        """
        Send the scheduled instructions to hardware.
        Use the jobManager class to submit the job to IBMQ or simulator

        Return:

            Result of the execution
        """
        pass



    def update_process_queue(self, result):
        """
        Update the process queue after one batch execution
        """
        proc = self._process_queue.get()
        # Update process based on result
        self._process_end_time[proc.get_process_id()] = time.time()-self._start_time
        self._process_waiting_time[proc.get_process_id()] = self._process_end_time[proc.get_process_id()] - self._process_start_time[proc.get_process_id()]
        self._process_fidelity[proc.get_process_id()] = 0.99  # Placeholder for actual fidelity calculation
        self._finished_process_count += 1




class jobManager:
    """
    This class manage the job and the job result.

    The job include:
    1. Manage the interface with IBMQ or simulator
    2. Decode the job result and distribution to process output
    """
    def __init__(self, ibmkey: str = "", use_simulator: bool = True):
        self._result_queue = queue.Queue()
        self._ibmkey = ibmkey
        self._use_simulator = use_simulator


    def submit_job_to_ibmq(self, scheduled_instructions):
        """
        Submit the scheduled instructions to IBMQ or simulator
        Get the raw result back
        
        Input:
            scheduled_instructions: List[instruction]
                The scheduled instruction list to be submitted
        """
        pass


    def redistribute_job_result(self, raw_result):
        """
        Redistribute the raw result to process output

        Input:
            raw_result: Any
                The raw result returned from IBMQ or simulator
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



def random_arrival_generator(scheduler: haloScheduler,
                             arrival_rate: float = 1,
                             max_time: float = 30.0):
    """
    Producer thread: generate processes according to a Poisson process
    (exponential inter-arrival times with mean 1/arrival_rate).
    """
    start = time.time()
    pid = 0
    while time.time() - start < max_time:
        # Wait random time until next arrival
        wait = random.expovariate(arrival_rate)  # mean 1/lambda
        time.sleep(wait)
        print("[ARRIVAL] New process arriving.")
        # Generate a random process
        num_dq = random.randint(1, 5)
        num_hq = random.randint(0, 4)
        shots = random.choice([100, 200, 500, 1000])
        proc = process(pid, num_dq, num_hq, shots,[])
        scheduler.add_process(proc)
        pid += 1

    print("[ARRIVAL] Finished generating processes.")
    




def generate_process_from_benchmark(benchmark_id: int, pid: int, shots: int) -> process:
    """
    Generate a process from the benchmark suit, given the pid and shots
    1. Load the benchmark data from the file
    2. Create a process instance
    3. Return the process instance
    """
    file_path = f"{benchmark_suit_file_root}{benchmark_suit[benchmark_id]}"
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

    print("Start testing haloScheduler...")
    scheduler = haloScheduler()
    scheduler.add_process(process1)
    scheduler.add_process(process2)
    scheduler.add_process(process3)



    print("Start getting next batch...")
    next_batch = scheduler.get_next_batch()
    L = scheduler.allocate_data_qubit(next_batch)


    plot_process_schedule_on_torino(
        torino_coupling_map(),
        next_batch,
        L,
        out_png="best_torino_mapping_3proc.png",
    )

    measurement_to_process_map, scheduled_instructions = scheduler.dynamic_helper_scheduling(L, next_batch)


    print("Scheduled Instructions:")
    for inst in scheduled_instructions:
        print(inst)
    print("Measurement to Process Map:")
    for meas, pid in measurement_to_process_map.items():
        print(f"Measurement {meas} -> Process {pid}")
    






if __name__ == "__main__":
    test_scheduling()








# if __name__ == "__main__":

#     random.seed(42)


#     haloScheduler_instance=haloScheduler()
#     haloScheduler_instance.start()



#     producer_thread = threading.Thread(
#         target=random_arrival_generator,
#         args=(haloScheduler_instance, 2, 10.0),
#         daemon=False
#     )
#     producer_thread.start()


#     simulation_time = 10.0  # seconds
#     time.sleep(simulation_time)


#     # Wait for producer to finish generating all processes
#     producer_thread.join()


#     haloScheduler_instance.stop()
#     print("Simulation finished.")


#     haloScheduler_instance.store_log("halo_scheduler_log.txt")

