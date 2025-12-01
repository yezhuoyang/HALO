from instruction import instruction, Instype, generate_program_without_helper_qubit
from typing import List, Tuple
import random



def generate_stabilizer_circuit_with_error(number_data_qubit: int,stabilizer_list: List[str], num_round: int, injected_error: str) -> Tuple[List[instruction], str]:
    """
    Generate a five-qubit code circuit with the specified round and injected error.

    Injected error is a string of length 5, each character is I, X, Y, or Z.
    """

    program_string = ""
    inst_list = []
    program_string += f"q = alloc_data({number_data_qubit})\n"
    program_string += f"s = alloc_helper({num_round*len(stabilizer_list)})\n"


    if len(injected_error) != len(stabilizer_list[0]):
        raise ValueError("injected_error must be a 5-character string of I/X/Y/Z")

    #Inject the error
    for qubit_index, error in enumerate(injected_error):
        data_qubit = f"q{qubit_index}"
        match error:
            case 'X':
                inst_list.append(instruction(Instype.X, [data_qubit]))
                program_string += f"X {data_qubit}\n"
            case 'Y':
                inst_list.append(instruction(Instype.Y, [data_qubit]))
                program_string += f"Y {data_qubit}\n"
            case 'Z':
                inst_list.append(instruction(Instype.Z, [data_qubit]))
                program_string += f"Z {data_qubit}\n"
            case 'I':
                continue
            case _:
                raise ValueError(f"Invalid error operator '{error}' for qubit '{data_qubit}'")


    #Compile many rounds of stabilizer measurement
    for r in range(num_round):
        for stab_index, stabilizer in enumerate(stabilizer_list):
            ancilla_qubit = f"s{stab_index + r*len(stabilizer_list)}"


            inst_list.append(instruction(Instype.H, [ancilla_qubit]))
            program_string += f"H {ancilla_qubit}\n"

            for data_index, pauli in enumerate(stabilizer):
                data_qubit = f"q{data_index}"
                match pauli:
                    case 'X':
                        inst_list.append(instruction(Instype.CNOT, [ancilla_qubit,data_qubit]))
                        program_string += f"CNOT {ancilla_qubit}, {data_qubit}\n"
                    case 'Z':
                        inst_list.append(instruction(Instype.H, [data_qubit]))
                        program_string += f"H {data_qubit}\n"
                        inst_list.append(instruction(Instype.CNOT, [ancilla_qubit,data_qubit]))
                        program_string += f"CNOT {ancilla_qubit}, {data_qubit}\n"
                        inst_list.append(instruction(Instype.H, [data_qubit]))
                        program_string += f"H {data_qubit}\n"
                    case 'I':
                        continue
                    case _:
                        raise ValueError(f"Invalid Pauli operator '{pauli}' in stabilizer '{stabilizer}'")

            # Final Hadamard on ancilla qubit before measurement
            inst_list.append(instruction(Instype.H, [ancilla_qubit]))
            program_string += f"H {ancilla_qubit}\n"

            # Measure the ancilla qubit
            classical_address = stab_index + r * len(stabilizer_list)
            inst_list.append(instruction(Instype.MEASURE, [ancilla_qubit], classical_address=classical_address))
            program_string += f"c{classical_address} = MEASURE {ancilla_qubit}\n"


            #Release the used ancilla qubit
            inst_list.append(instruction(Instype.RELEASE, [ancilla_qubit]))
            program_string += f"deallocate_helper({ancilla_qubit})\n"


    program_string += f"deallocate_data(q)\n"
    return inst_list, program_string



def generate_five_qubit_code_benchmark(num_round: int, injected_error: str) -> Tuple[List[instruction], str]:
    """
    Five-qubit [[5,1,3]] code:
    X stabilizers and Z stabilizers (5 data qubits).
    Qubit order: q0..q4
    """
    number_data_qubit = 5

    five_qubit_stabilizers = [
        # X-type
        "XZZXI",  # X0 Z1 Z2 X3 I4
        "IXZZX",  # I0 X1 Z2 Z3 X4
        "XIXZZ",  # X0 I1 X2 Z3 Z4
        "ZXIXZ",  # Z0 X1 I2 X3 Z4
        # Z-type
        "ZXXZI",  # Z0 X1 X2 Z3 I4
        "IZXXZ",  # I0 Z1 X2 X3 Z4
        "ZIZXX",  # Z0 I1 Z2 X3 X4
        "XZIZX",  # X0 Z1 I2 Z3 X4
    ]

    return generate_stabilizer_circuit_with_error(
        number_data_qubit,
        five_qubit_stabilizers,
        num_round,
        injected_error,
    )





def generate_steane_code_benchmark(num_round: int, injected_error: str) -> Tuple[List[instruction], str]:
    """
    Steane [[7,1,3]] CSS code:
    X stabilizers and Z stabilizers (7 data qubits).
    Qubit order: q0..q6
    """

    number_data_qubit = 7

    steane_stabilizers = [
        # X-type
        "XXXXIII",  # X0 X1 X2 X3
        "XXIIXXI",  # X0 X1 X4 X5
        "XIXIXIX",  # X0 X2 X4 X6
        # Z-type
        "ZZZZIII",  # Z0 Z1 Z2 Z3
        "ZZIIZZI",  # Z0 Z1 Z4 Z5
        "ZIZIZIZ",  # Z0 Z2 Z4 Z6
    ]

    return generate_stabilizer_circuit_with_error(
        number_data_qubit,
        steane_stabilizers,
        num_round,
        injected_error,
    )



def generate_shor_code_benchmark(num_round: int, injected_error: str) -> Tuple[List[instruction], str]:
    """
    Shor [[9,1,3]] code (bit-flip + phase-flip):
    9 data qubits.
    Qubit order: q0..q8
    """
    number_data_qubit = 9

    shor_stabilizers = [
        # Z-type (pairwise ZZ within each 3-qubit block)
        "ZZIIIIIII",  # Z0 Z1
        "IZZIIIIII",  # Z1 Z2
        "IIIZZIIII",  # Z3 Z4
        "IIIIZZIII",  # Z4 Z5
        "IIIIIIZZI",  # Z6 Z7
        "IIIIIIIZZ",  # Z7 Z8
        # X-type (XXXXXXIII and IIIXXXXXX to tie blocks)
        "XXXXXXIII",  # X0..X5
        "IIIXXXXXX",  # X3..X8
    ]

    return generate_stabilizer_circuit_with_error(
        number_data_qubit,
        shor_stabilizers,
        num_round,
        injected_error,
    )


def generate_surface_code_d3_benchmark(num_round: int, injected_error: str) -> Tuple[List[instruction], str]:
    """
    Generate the benchmark program for the surface code (d=3) with the specified round and injected error.
    """
    """
    Rotated surface-code-like distance-3 patch with 9 data qubits (3x3 grid).
    Qubit order (row-major): 
        q0 q1 q2
        q3 q4 q5
        q6 q7 q8
    4 X-type and 4 Z-type plaquette stabilizers (weight-4).
    """
    number_data_qubit = 9

    surface_d3_stabilizers = [
        # X-type plaquettes
        "XXIXXIIII",  # X0 X1 X3 X4
        "IXXIXXIII",  # X1 X2 X4 X5
        "IIIXXIXXI",  # X3 X4 X6 X7
        "IIIIXXIXX",  # X4 X5 X7 X8
        # Z-type plaquettes
        "ZZIZZIIII",  # Z0 Z1 Z3 Z4
        "IZZIZZIII",  # Z1 Z2 Z4 Z5
        "IIIZZIZZI",  # Z3 Z4 Z6 Z7
        "IIIIZZIZZ",  # Z4 Z5 Z7 Z8
    ]

    return generate_stabilizer_circuit_with_error(
        number_data_qubit,
        surface_d3_stabilizers,
        num_round,
        injected_error,
    )


def generate_repetition_code_d3_benchmark(num_round: int, injected_error: str) -> Tuple[List[instruction], str]:
    """
    Generate the benchmark program for the repetition code (d=3) with the specified round and injected error.
    """
    """
    Repetition code of distance 3 (3 data qubits).
    Bit-flip code: ZZ checks + global X stabilizer.
    Qubit order: q0 q1 q2
    """
    number_data_qubit = 3

    repetition_d3_stabilizers = [
        "ZZI",  # Z0 Z1
        "IZZ",  # Z1 Z2
        "XXX",  # X0 X1 X2
    ]

    return generate_stabilizer_circuit_with_error(
        number_data_qubit,
        repetition_d3_stabilizers,
        num_round,
        injected_error,
    )


def generate_repetition_code_d5_benchmark(num_round: int, injected_error: str) -> Tuple[List[instruction], str]:
    """
    Generate the benchmark program for the repetition code (d=5) with the specified round and injected error.
    """
    """
    Repetition code of distance 5 (5 data qubits).
    Bit-flip code: chain of ZZ checks + global X stabilizer.
    Qubit order: q0..q4
    """
    number_data_qubit = 5

    repetition_d5_stabilizers = [
        "ZZIII",  # Z0 Z1
        "IZZII",  # Z1 Z2
        "IIZZI",  # Z2 Z3
        "IIIZZ",  # Z3 Z4
        "XXXXX",  # X0..X4
    ]

    return generate_stabilizer_circuit_with_error(
        number_data_qubit,
        repetition_d5_stabilizers,
        num_round,
        injected_error,
    )






def generate_qec_benchmark_medium():
    """
    Generate medium-sized QEC benchmark programs and store them in the folder 'benchmark//qecmedium//'.
    """


    #Five qubit code
    for num_round in range(2,4):
        #Sample 3 different error patterns with 1 error
        for k in range(3):
            #Randomly generate an error pattern with 1 errors
            error_pattern = ['I']*5
            error_positions = random.sample(range(5), k=1)
            error_pattern[error_positions[0]] = random.choice(['X','Y','Z'])
            error_pattern = ''.join(error_pattern)

            inst_list, program_string = generate_five_qubit_code_benchmark(num_round, error_pattern)
            file_path = f"benchmark/qecmedium/fivequbit_round_{num_round}_error_{error_pattern}"
            with open(file_path, 'w') as f:
                f.write(program_string)

            #Generate the program without helper qubit
            program_no_helper= generate_program_without_helper_qubit(5, num_round*4, inst_list)
            file_path_no_helper = f"benchmark/qecmedium/nohelper/fivequbit_round_{num_round}_error_{error_pattern}"
            with open(file_path_no_helper, 'w') as f:
                f.write(program_no_helper)


    #Steane code
    for num_round in range(2,4):
        #Sample 3 different error patterns with 1 error
        for k in range(3):
            #Randomly generate an error pattern with 1 errors
            error_pattern = ['I']*7
            error_positions = random.sample(range(7), k=1)
            error_pattern[error_positions[0]] = random.choice(['X','Y','Z'])
            error_pattern = ''.join(error_pattern)
            #print(f"Generating Steane code benchmark: round {num_round}, error {error_pattern}")
            inst_list, program_string = generate_steane_code_benchmark(num_round, error_pattern)
            file_path = f"benchmark/qecmedium/steane_round_{num_round}_error_{error_pattern}"
            with open(file_path, 'w') as f:
                f.write(program_string)

            #Generate the program without helper qubit
            program_no_helper= generate_program_without_helper_qubit(7, num_round*6, inst_list)
            file_path_no_helper = f"benchmark/qecmedium/nohelper/steane_round_{num_round}_error_{error_pattern}"
            with open(file_path_no_helper, 'w') as f:
                f.write(program_no_helper)


    #Shor code
    for num_round in range(2,4):
        #Sample 3 different error patterns with 1 error
        for k in range(3):
            #Randomly generate an error pattern with 1 errors
            error_pattern = ['I']*9
            error_positions = random.sample(range(9), k=1)
            error_pattern[error_positions[0]] = random.choice(['X','Y','Z'])
            error_pattern = ''.join(error_pattern)

            inst_list, program_string = generate_shor_code_benchmark(num_round, error_pattern)
            file_path = f"benchmark/qecmedium/shor_round_{num_round}_error_{error_pattern}"
            with open(file_path, 'w') as f:
                f.write(program_string)

            #Generate the program without helper qubit
            program_no_helper= generate_program_without_helper_qubit(9, num_round*8, inst_list)
            file_path_no_helper = f"benchmark/qecmedium/nohelper/shor_round_{num_round}_error_{error_pattern}"
            with open(file_path_no_helper, 'w') as f:
                f.write(program_no_helper)

    
    #Surface code d3
    for num_round in range(2,4):
        #Sample 3 different error patterns with 1 error
        for k in range(3):
            #Randomly generate an error pattern with 1 errors
            error_pattern = ['I']*9
            error_positions = random.sample(range(9), k=1)
            error_pattern[error_positions[0]] = random.choice(['X','Y','Z'])
            error_pattern = ''.join(error_pattern)

            inst_list, program_string = generate_surface_code_d3_benchmark(num_round, error_pattern)
            file_path = f"benchmark/qecmedium/surface_d3_round_{num_round}_error_{error_pattern}"
            with open(file_path, 'w') as f:
                f.write(program_string)

            #Generate the program without helper qubit
            program_no_helper= generate_program_without_helper_qubit(9, num_round*8, inst_list)
            file_path_no_helper = f"benchmark/qecmedium/nohelper/surface_d3_round_{num_round}_error_{error_pattern}"
            with open(file_path_no_helper, 'w') as f:
                f.write(program_no_helper)


    #Repetition code d3
    for num_round in range(2,4):
        #Sample 3 different error patterns with 1 error
        for k in range(3):
            #Randomly generate an error pattern with 1 errors
            error_pattern = ['I']*3
            error_positions = random.sample(range(3), k=1)
            error_pattern[error_positions[0]] = random.choice(['X','Y','Z'])
            error_pattern = ''.join(error_pattern)

            inst_list, program_string = generate_repetition_code_d3_benchmark(num_round, error_pattern)
            file_path = f"benchmark/qecmedium/repetition_d3_round_{num_round}_error_{error_pattern}"
            with open(file_path, 'w') as f:
                f.write(program_string)

            #Generate the program without helper qubit
            program_no_helper= generate_program_without_helper_qubit(3, num_round*3, inst_list)
            file_path_no_helper = f"benchmark/qecmedium/nohelper/repetition_d3_round_{num_round}_error_{error_pattern}"
            with open(file_path_no_helper, 'w') as f:
                f.write(program_no_helper)


    #Repetition code d5
    for num_round in range(2,4):
        #Sample 3 different error patterns with 1 error
        for k in range(3):
            #Randomly generate an error pattern with 1 errors
            error_pattern = ['I']*5
            error_positions = random.sample(range(5), k=1)
            error_pattern[error_positions[0]] = random.choice(['X','Y','Z'])
            error_pattern = ''.join(error_pattern)

            inst_list, program_string = generate_repetition_code_d5_benchmark(num_round, error_pattern)
            file_path = f"benchmark/qecmedium/repetition_d5_round_{num_round}_error_{error_pattern}"
            with open(file_path, 'w') as f:
                f.write(program_string)

            #Generate the program without helper qubit
            program_no_helper= generate_program_without_helper_qubit(5, num_round*4, inst_list)
            file_path_no_helper = f"benchmark/qecmedium/nohelper/repetition_d5_round_{num_round}_error_{error_pattern}"
            with open(file_path_no_helper, 'w') as f:
                f.write(program_no_helper)

