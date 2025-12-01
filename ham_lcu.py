from instruction import instruction, Instype, get_gate_type_name
from typing import List, Tuple
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli

# --- 1. Memory Management / Compiler ---

class LCUCompiler:
    """
    Manages qubit allocation and instruction generation.
    Distinguishes between 'Select' qubits (persistent) and 'Work' qubits (transient).
    """
    def __init__(self, num_data: int):
        self.inst_list: List[instruction] = []
        self.program_string = ""
        self.num_data = num_data
        
        # Helper ID tracking
        self.next_helper_id = 0
        self.active_helpers = set()
        self.max_helper_usage = 0

    def alloc_helper(self) -> str:
        """Allocates a new helper qubit."""
        # Find the first available ID (simple increment strategy for unique names, 
        # but in a real allocator we might reuse IDs. Here we increment to keep unique names 
        # for the timeline, but track max simultaneous usage).
        
        # To simulate re-usable memory, we typically just increment a counter 
        # but track how many are active simultaneously.
        
        hid = self.next_helper_id
        self.next_helper_id += 1
        
        name = f"s{hid}"
        self.active_helpers.add(name)
        
        current_usage = len(self.active_helpers)
        if current_usage > self.max_helper_usage:
            self.max_helper_usage = current_usage
            
        return name

    def release_helper(self, name: str):
        """Releases a helper qubit immediately."""
        if name in self.active_helpers:
            self.active_helpers.remove(name)
            
            # Emit RELEASE instruction
            self.inst_list.append(instruction(Instype.RELEASE, [name]))
            self.program_string += f"deallocate_helper({name})\n"
        else:
            raise ValueError(f"Attempted to release unallocated helper {name}")

    def emit(self, inst_type: Instype, qubits: List[str]):
        self.inst_list.append(instruction(inst_type, qubits))
        q_str = ", ".join(qubits)
        cmd = get_gate_type_name(inst_type)
        self.program_string += f"{cmd} {q_str}\n"

# --- 2. Gate Logic ---

def emit_mcx_with_release(
    controls: List[str], 
    target: str, 
    compiler: LCUCompiler
):
    """
    Generates a Multi-Controlled X gate using Toffoli decomposition (V-chain).
    Allocates work qubits on the fly and RELEASES them immediately after uncomputation.
    """
    num_ctrl = len(controls)
    
    # 
    
    # Base cases
    if num_ctrl == 0:
        compiler.emit(Instype.X, [target])
        return
    if num_ctrl == 1:
        compiler.emit(Instype.CNOT, [controls[0], target])
        return
    if num_ctrl == 2:
        compiler.emit(Instype.Toffoli, [controls[0], controls[1], target])
        return

    # Recursive step / Ladder decomposition for n > 2
    # We need n - 2 work qubits.
    num_work_needed = num_ctrl - 2
    
    # 1. Allocate Work Qubits
    work_qubits = []
    for _ in range(num_work_needed):
        work_qubits.append(compiler.alloc_helper())
        
    # 2. Compute Ladder (Forward)
    # Tof(c0, c1, w0)
    compiler.emit(Instype.Toffoli, [controls[0], controls[1], work_qubits[0]])
    
    for i in range(1, num_work_needed):
        # Tof(c_{i+1}, w_{i-1}, w_{i})
        c_idx = i + 1
        w_target = work_qubits[i]
        w_ctrl = work_qubits[i-1]
        compiler.emit(Instype.Toffoli, [controls[c_idx], w_ctrl, w_target])

    # 3. Target Operation
    # Tof(c_{last}, w_{last}, target)
    compiler.emit(Instype.Toffoli, [controls[-1], work_qubits[num_work_needed-1], target])

    # 4. Uncompute Ladder (Backward)
    for i in range(num_work_needed - 1, 0, -1):
        c_idx = i + 1
        w_target = work_qubits[i]
        w_ctrl = work_qubits[i-1]
        compiler.emit(Instype.Toffoli, [controls[c_idx], w_ctrl, w_target])
        
    compiler.emit(Instype.Toffoli, [controls[0], controls[1], work_qubits[0]])

    # 5. Release Work Qubits (Immediate Release)
    # We release in reverse order (stack discipline), though sets don't strictly care.
    for w in reversed(work_qubits):
        compiler.release_helper(w)


def generate_lcu_benchmark(hamiltonian_terms: List[str]) -> Tuple[List[instruction], str]:
    """
    Generates a quantum program implementing the LCU lemma.
    Uses dynamic allocation to minimize peak qubit usage.
    """
    if not hamiltonian_terms:
        return [], ""

    num_data = len(hamiltonian_terms[0])
    num_terms = len(hamiltonian_terms)
    
    compiler = LCUCompiler(num_data)
    
    # Calculate required Select bits
    num_select = math.ceil(math.log2(num_terms))
    if num_select == 0: num_select = 1 

    # 

    # --- Header ---
    # Note: We construct the header at the end or update a placeholder, 
    # but here we build string iteratively. We will prepend allocs later or just output logic.
    header_str = f"q = alloc_data({num_data})\n"
    
    # 1. Allocate Select Qubits
    # These MUST persist for the entire duration of the operator sum.
    select_qubits = []
    for _ in range(num_select):
        select_qubits.append(compiler.alloc_helper())
        
    # PREP (Hadamard on Select)
    for s_q in select_qubits:
        compiler.emit(Instype.H, [s_q])

    # 2. Iterate Terms (SELECT + Controlled-Operations)
    for k, term in enumerate(hamiltonian_terms):
        if len(term) != num_data:
            raise ValueError("All Hamiltonian terms must have the same length.")
        
        # 2a. Condition on index k
        flipped_indices = []
        for bit_idx in range(num_select):
            # Check bit at bit_idx is 0 in k
            if not ((k >> bit_idx) & 1):
                s_q = select_qubits[bit_idx]
                compiler.emit(Instype.X, [s_q])
                flipped_indices.append(bit_idx)
                
        # 2b. Apply Controlled-Paulis
        # Control is the entire select register
        for data_idx, pauli_char in enumerate(term):
            target_q = f"q{data_idx}"
            
            if pauli_char == 'I':
                continue
                
            elif pauli_char == 'X':
                emit_mcx_with_release(select_qubits, target_q, compiler)
                
            elif pauli_char == 'Z':
                compiler.emit(Instype.H, [target_q])
                emit_mcx_with_release(select_qubits, target_q, compiler)
                compiler.emit(Instype.H, [target_q])
                
            elif pauli_char == 'Y':
                compiler.emit(Instype.Sdg, [target_q])
                emit_mcx_with_release(select_qubits, target_q, compiler)
                compiler.emit(Instype.S, [target_q])

        # 2c. Un-condition
        for bit_idx in flipped_indices:
            s_q = select_qubits[bit_idx]
            compiler.emit(Instype.X, [s_q])

    # 3. Finalize Select Register (Hdag -> Measure -> Release)
    for s_q in select_qubits:
        compiler.emit(Instype.H, [s_q])
        
    for i, s_q in enumerate(select_qubits):
        compiler.inst_list.append(instruction(Instype.MEASURE, [s_q], classical_address=i))
        compiler.program_string += f"c{i} = MEASURE {s_q}\n"
        
        # Now we can release the select qubits
        compiler.release_helper(s_q)

    # Final Cleanup string
    compiler.program_string += "deallocate_data(q)\n"
    
    # Prepend the resource allocation to the string
    full_program = header_str
    full_program += f"s = alloc_helper({compiler.max_helper_usage}) # Max simultaneous helpers\n"
    full_program += compiler.program_string
    
    return compiler.inst_list, full_program


# --- 3. Testing Infrastructure (Updated for compatibility) ---

def norm_2_distance(matrix1: np.ndarray, matrix2: np.ndarray):
    return np.sum(np.abs(matrix1 - matrix2) ** 2)

class LCUTester:
    def __init__(self, terms: list[str]):
        self.terms = terms
        self.num_data = len(terms[0])
        self.num_terms = len(terms)
        import math
        self.num_select = math.ceil(math.log2(self.num_terms))
        if self.num_select == 0: self.num_select = 1
        self.alpha = 2 ** self.num_select

    def instructions_to_qiskit_unitary(self, inst_list, num_data, total_helpers) -> np.ndarray:
        total_qubits = num_data + total_helpers
        qc = QuantumCircuit(total_qubits)

        def get_idx(addr: str):
            if addr.startswith('q'):
                return int(addr[1:])
            elif addr.startswith('s'):
                return num_data + int(addr[1:])
            else:
                raise ValueError(f"Unknown register {addr}")

        for inst in inst_list:
            t = inst.get_type()
            qs = [get_idx(a) for a in inst.get_qubitaddress()]

            if t == Instype.H: qc.h(qs[0])
            elif t == Instype.X: qc.x(qs[0])
            elif t == Instype.Y: qc.y(qs[0])
            elif t == Instype.Z: qc.z(qs[0])
            elif t == Instype.S: qc.s(qs[0])
            elif t == Instype.Sdg: qc.sdg(qs[0])
            elif t == Instype.CNOT: qc.cx(qs[0], qs[1])
            elif t == Instype.Toffoli: qc.ccx(qs[0], qs[1], qs[2])
            # Measurements and Release ignored for Unitary construction
            elif t == Instype.MEASURE or t == Instype.RELEASE:
                continue 
            else:
                print(f"Warning: instruction {t} not implemented in tester adapter.")

        return np.array(Operator(qc))

    def get_exact_operator(self) -> np.ndarray:
        dim = 2 ** self.num_data
        total_op = np.zeros((dim, dim), dtype=complex)
        for term in self.terms:
            qiskit_str = term[::-1] 
            mat = Pauli(qiskit_str).to_matrix()
            total_op += mat
        return total_op

    def extract_block_encoding(self, W: np.ndarray, num_data, total_helpers) -> np.ndarray:
        dim_data = 2 ** num_data
        return W[0:dim_data, 0:dim_data]

    def run_test(self):
        print(f"Testing LCU for terms: {self.terms}")
        
        # 1. Generate Circuit Instructions
        inst_list, prog_str = generate_lcu_benchmark(self.terms)
        
        # Scan for max index to determine size of simulator
        max_s = -1
        for inst in inst_list:
            for q in inst.get_qubitaddress():
                if q.startswith('s'):
                    idx = int(q[1:])
                    if idx > max_s: max_s = idx
        total_helpers = max_s + 1

        print(f" > Max Helper Qubit Index used: {max_s}")

        # 2. Get Circuit Unitary
        W = self.instructions_to_qiskit_unitary(inst_list, self.num_data, total_helpers)
        
        # 3. Project and Compare
        projected_op = self.extract_block_encoding(W, self.num_data, total_helpers)
        exact_H = self.get_exact_operator()
        scaled_exact = exact_H / self.alpha
        
        dist = norm_2_distance(projected_op, scaled_exact)
        
        print(f" > Normalization Factor (alpha): {self.alpha}")
        print(f" > 2-Norm Distance: {dist:.6e}")
        
        if dist < 1e-10:
            print("[PASS] LCU Implementation Correct.\n")
        else:
            print("[FAIL] Discrepancy detected.\n")

if __name__ == "__main__":
    # Test Case 1: Simple



# Define the Hamiltonian terms (Pauliz strings)
    terms = ["XXX", "ZZZ"]
    
    # Generate the program
    instructions, program_str = generate_lcu_benchmark(terms)
    
    # Print the resulting program string
    print(program_str)
    
    # tester = LCUTester(["XXX", "ZZZ", "YZY"])
    # tester.run_test()
    
    # # Test Case 2: Mixed
    # tester2 = LCUTester(["XIXI", "IZZZ","IIIY"])
    # tester2.run_test()