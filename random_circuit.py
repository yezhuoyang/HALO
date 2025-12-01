from instruction import instruction, Instype, get_gate_type_name, generate_program_without_helper_qubit
from typing import List, Tuple
import random



def generate_random_program(data_n: int, syn_n: int, gate_count: int,seed: int=None) -> Tuple[List[instruction], str]:
    """
    Generate a completely random program with the given parameters.
    """
    if seed is not None:
        random.seed(seed)

    program_string = ""
    inst_list = []
    program_string += f"q = alloc_data({data_n})\n"
    program_string += f"s = alloc_helper({syn_n})\n"



    """
    We maintain a set of active helper qubits.
    When we release a helper qubit, we remove it from the set.
    """
    active_helper_qubits = set()
    for i in range(syn_n):
        active_helper_qubits.add(f"s{i}")
    active_data_qubits = set()
    for i in range(data_n):
        active_data_qubits.add(f"q{i}")

    forbidden = {Instype.RESET,
                Instype.CSWAP, Instype.Toffoli, Instype.RELEASE}
    allowed = [g for g in Instype if g not in forbidden]



    #Randomize the location of release instructions
    release_locations = random.sample(range(gate_count)[2:], k=syn_n)
    current_gate_count = 0

    current_measure_index = 0
    for _ in range(gate_count):
        # Randomly choose gate type and qubits

        if current_gate_count in release_locations and active_helper_qubits:
            random_qubit = random.choice(list(active_helper_qubits))
            inst = instruction(Instype.MEASURE, [random_qubit], classical_address=current_measure_index)
            inst_list.append(inst)
            program_string += f"c{current_measure_index} = MEASURE {random_qubit}\n"
            current_measure_index += 1
            inst = instruction(Instype.RELEASE, [random_qubit])
            active_helper_qubits.remove(random_qubit)
            program_string += f"deallocate_helper({random_qubit})\n"
            inst_list.append(inst)
            current_gate_count += 1
            continue

        gate_type = random.choice(allowed)
        """
        Randomly generate two qubit,
        either qk or sk, k is the qubit index
        We can only choose from active helper qubits
        """
        if active_data_qubits==set() and active_helper_qubits==set():
            break


        match gate_type:
            case Instype.H | Instype.X | Instype.Y | Instype.Z | Instype.T | Instype.Tdg | Instype.S | Instype.Sdg | Instype.SX:
                random_qubit = random.choice(list(active_data_qubits)+list(active_helper_qubits))
                inst = instruction(gate_type, [random_qubit])          
                inst_list.append(inst)      
                program_string += f"{get_gate_type_name(gate_type)} {random_qubit}\n"
            case Instype.RZ | Instype.RX | Instype.RY:
                random_qubit = random.choice(list(active_data_qubits)+list(active_helper_qubits))
                angle = random.uniform(0, 2 * 3.141592653589793)
                inst = instruction(gate_type, [random_qubit], params=[angle])
                inst_list.append(inst)
                program_string += f"{get_gate_type_name(gate_type)}({angle}) {random_qubit} \n"
            case Instype.U3 | Instype.U:
                random_qubit = random.choice(list(active_data_qubits)+list(active_helper_qubits))
                angles = [random.uniform(0, 2 * 3.141592653589793) for _ in range(3)]
                inst = instruction(gate_type, [random_qubit], params=angles)
                inst_list.append(inst)
                program_string += f"{get_gate_type_name(gate_type)}({angles[0]}, {angles[1]}, {angles[2]}) {random_qubit} \n"
            case Instype.CNOT | Instype.CH | Instype.SWAP:
                qubits = random.sample(list(active_data_qubits)+list(active_helper_qubits), k=2)
                inst = instruction(gate_type, qubits)
                inst_list.append(inst)
                program_string += f"{get_gate_type_name(gate_type)} {qubits[0]}, {qubits[1]}\n"
            case Instype.CP:
                qubits = random.sample(list(active_data_qubits)+list(active_helper_qubits), k=2)
                angle = random.uniform(0, 2 * 3.141592653589793)
                inst = instruction(gate_type, qubits, params=[angle])
                inst_list.append(inst)
                program_string += f"{get_gate_type_name(gate_type)}({angle}) {qubits[0]}, {qubits[1]}\n"
            case Instype.MEASURE:
                random_qubit = random.choice(list(active_data_qubits)+list(active_helper_qubits))
                inst = instruction(gate_type, [random_qubit], classical_address=current_measure_index)
                inst_list.append(inst)
                program_string += f"c{current_measure_index} = MEASURE {random_qubit}\n"
                current_measure_index += 1
                # Remove the measured qubit from active sets
                if random_qubit.startswith('q'):
                    active_data_qubits.remove(random_qubit)
                else:
                    active_helper_qubits.remove(random_qubit)
                    #Add a release instruction
                    inst = instruction(Instype.RELEASE, [random_qubit])
                    inst_list.append(inst)
                    program_string += f"deallocate_helper({random_qubit})\n"


        current_gate_count += 1

    program_string += "deallocate_data(q)\n"
    return inst_list, program_string



def generate_random_benchmark_small(data_n: int, syn_n: int, gate_count: int,label: int, seed: int=None):
    """
    Generate the random benchmark program and store it in 'benchmark/randomsmall/'.
    """
    inst_list, program_string = generate_random_program(data_n, syn_n, gate_count, seed)

    file_path = f"benchmark/randomsmall/data_{data_n}_syn_{syn_n}_gc_{gate_count}_{label}"
    with open(file_path, 'w') as f:
        f.write(program_string)


    program_no_helper= generate_program_without_helper_qubit(data_n, syn_n, inst_list)
    file_path_no_helper = f"benchmark/randomsmall/nohelper/data_{data_n}_syn_{syn_n}_gc_{gate_count}_{label}"
    with open(file_path_no_helper, 'w') as f:
        f.write(program_no_helper)    



def generate_random_benchmark_medium(data_n: int, syn_n: int, gate_count: int,label: int, seed: int=None):
    """
    Generate the random benchmark program and store it in 'benchmark/randommedium/'.
    """
    inst_list, program_string = generate_random_program(data_n, syn_n, gate_count, seed)
    file_path = f"benchmark/randommedium/data_{data_n}_syn_{syn_n}_gc_{gate_count}_{label}"
    with open(file_path, 'w') as f:
        f.write(program_string)


    program_no_helper= generate_program_without_helper_qubit(data_n, syn_n, inst_list)
    file_path_no_helper = f"benchmark/randommedium/nohelper/data_{data_n}_syn_{syn_n}_gc_{gate_count}_{label}"
    with open(file_path_no_helper, 'w') as f:
        f.write(program_no_helper)        

