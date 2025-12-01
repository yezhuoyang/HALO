
from instruction import instruction, Instype, generate_program_without_helper_qubit
from typing import List, Tuple
from instruction import instruction, Instype
from toffoli_decomposition import toffoli_decomposition

def generate_multi_controlled_x(num_controls: int, input_str: str, decompose_toffoli: bool) -> Tuple[List[instruction], str]:
    """
    Generate a program of multi-controlled X gate using ancilla qubits.

    The input is a string of I and X, representing the state of the control qubits

    Returns a list of instructions implementing the multi-controlled X gate.
    """
    program_string = ""
    inst_list = []
    program_string += f"q = alloc_data({num_controls+1})\n"
    program_string += f"s = alloc_helper({num_controls-1})\n"


    #Compute the input
    for i in range(num_controls):
        if input_str[i] == 'X':
            inst_list.append(instruction(Instype.X, [f"q{i}"]))
            program_string += f"X q{i}\n"


    if decompose_toffoli:
        toffoli_insts, toffoli_str = toffoli_decomposition(f"q0", f"q1", f"s0")
        inst_list.extend(toffoli_insts)
        program_string += toffoli_str
    else:
        inst_list.append(instruction(Instype.Toffoli, [f"q0", f"q1", f"s0"]))
        program_string += f"Toffoli q0, q1, s0\n"

    for i in range(2, num_controls):
        if decompose_toffoli:
            toffoli_insts, toffoli_str = toffoli_decomposition(f"q{i}", f"s{i-2}", f"s{i-1}")
            inst_list.extend(toffoli_insts)
            program_string += toffoli_str
        else:
            inst_list.append(instruction(Instype.Toffoli, [f"q{i}", f"s{i-2}", f"s{i-1}"]))
            program_string += f"Toffoli q{i}, s{i-2}, s{i-1}\n"

    inst_list.append(instruction(Instype.CNOT, [f"s{num_controls-2}", f"q{num_controls}"]))
    program_string += f"CNOT s{num_controls-2}, q{num_controls}\n"

    #Uncompute
    for i in range(num_controls-1, 1, -1):
        if decompose_toffoli:
            toffoli_insts, toffoli_str = toffoli_decomposition(f"q{i}", f"s{i-2}", f"s{i-1}")
            inst_list.extend(toffoli_insts)
            program_string += toffoli_str
        else:
            inst_list.append(instruction(Instype.Toffoli, [f"q{i}", f"s{i-2}", f"s{i-1}"]))
            program_string += f"Toffoli q{i}, s{i-2}, s{i-1}\n"
        #Release helper qubit
        inst_list.append(instruction(Instype.RELEASE, [f"s{i-1}"]))
        program_string += f"deallocate_helper(s{i-1})\n"

    if decompose_toffoli:
        toffoli_insts, toffoli_str = toffoli_decomposition(f"q0", f"q1", f"s0")
        inst_list.extend(toffoli_insts)
        program_string += toffoli_str
    else:
        inst_list.append(instruction(Instype.Toffoli, [f"q0", f"q1", f"s0"]))
        program_string += f"Toffoli q0, q1, s0\n"
    #Release helper qubit
    inst_list.append(instruction(Instype.RELEASE, [f"s0"]))
    program_string += f"deallocate_helper(s0)\n"


    measure_index = 0
    #Measure all data qubits
    for i in range(num_controls):
        inst_list.append(instruction(Instype.MEASURE, qubitaddress=[f"q{i}"], classical_address=measure_index))
        program_string += f"c{i} = MEASURE q{i}\n"
        measure_index += 1

    program_string += f"deallocate_data(q)\n"
    return inst_list, program_string



def generate_mcx_benchmark_small(num_controls: int):
    """
    Generate a benchmark program for multi-controlled X gate.

    Store the problem in the folder 'benchmark//multiXsmall//'.

    Also generate the corresponding program without considering helper qubit.
    """

    assert num_controls <= 10, "Number of controls must be at most 10."

    inst_list, program_string = generate_multi_controlled_x(num_controls,"I"*num_controls ,decompose_toffoli=True)
    file_path = f"benchmark/multiXsmall/mcx_{num_controls}_0"
    with open(file_path, 'w') as f:
        f.write(program_string)


    program_no_helper= generate_program_without_helper_qubit(num_controls+1, num_controls-1, inst_list)
    file_path_no_helper = f"benchmark/multiXsmall/nohelper/mcx_{num_controls}_0"
    with open(file_path_no_helper, 'w') as f:
        f.write(program_no_helper)

    
    inst_list, program_string = generate_multi_controlled_x(num_controls,"X"*num_controls ,decompose_toffoli=True)
    file_path = f"benchmark/multiXsmall/mcx_{num_controls}_1"
    with open(file_path, 'w') as f:
        f.write(program_string)


    program_no_helper= generate_program_without_helper_qubit(num_controls+1, num_controls-1, inst_list)
    file_path_no_helper = f"benchmark/multiXsmall/nohelper/mcx_{num_controls}_1"
    with open(file_path_no_helper, 'w') as f:
        f.write(program_no_helper)    





def generate_mcx_benchmark_medium(num_controls: int):
    assert num_controls > 7, "Number of controls must be at least 9."

    inst_list, program_string = generate_multi_controlled_x(num_controls,"I"*num_controls, decompose_toffoli=True)
    file_path = f"benchmark/multiXmedium/mcx_{num_controls}_0"
    with open(file_path, 'w') as f:
        f.write(program_string)


    program_no_helper= generate_program_without_helper_qubit(num_controls+1, num_controls-1, inst_list)
    file_path_no_helper = f"benchmark/multiXmedium/nohelper/mcx_{num_controls}_0"
    with open(file_path_no_helper, 'w') as f:
        f.write(program_no_helper)    


    inst_list, program_string = generate_multi_controlled_x(num_controls,"X"*num_controls, decompose_toffoli=True)
    file_path = f"benchmark/multiXmedium/mcx_{num_controls}_1"
    with open(file_path, 'w') as f:
        f.write(program_string)


    program_no_helper= generate_program_without_helper_qubit(num_controls+1, num_controls-1, inst_list)
    file_path_no_helper = f"benchmark/multiXmedium/nohelper/mcx_{num_controls}_1"
    with open(file_path_no_helper, 'w') as f:
        f.write(program_no_helper)    




