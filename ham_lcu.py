from instruction import instruction, Instype, get_gate_type_name
from typing import List, Tuple
import math
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli
from qiskit.circuit.library import XGate, HGate, ZGate, SGate, SdgGate, CXGate, CCXGate


def get_required_helpers_for_mcx(num_controls: int) -> int:
    """
    Calculate required workspace (ladder) qubits for a multi-controlled X gate.
    For n controls (n > 2), we need n - 2 ancillas for a V-chain decomposition.
    """
    if num_controls <= 2:
        return 0
    return num_controls - 2

def emit_mcx(
    controls: List[str], 
    target: str, 
    work_qubits: List[str], 
    inst_list: List[instruction], 
    program_string: str
) -> Tuple[List[instruction], str]:
    """
    Generates a Multi-Controlled X gate using Toffoli decomposition (V-chain).
    """
    num_ctrl = len(controls)
    
    # Base cases
    if num_ctrl == 0:
        inst_list.append(instruction(Instype.X, [target]))
        program_string += f"X {target}\n"
        return inst_list, program_string
    if num_ctrl == 1:
        inst_list.append(instruction(Instype.CNOT, [controls[0], target]))
        program_string += f"CNOT {controls[0]}, {target}\n"
        return inst_list, program_string
    if num_ctrl == 2:
        inst_list.append(instruction(Instype.Toffoli, [controls[0], controls[1], target]))
        program_string += f"Toffoli {controls[0]}, {controls[1]}, {target}\n"
        return inst_list, program_string

    # Recursive step / Ladder decomposition for n > 2
    # We need num_ctrl - 2 work qubits.
    if len(work_qubits) < num_ctrl - 2:
        raise ValueError(f"Not enough work qubits for MCX. Need {num_ctrl-2}, got {len(work_qubits)}")

    # 1. Compute Ladder (Forward)
    # Tof(c0, c1, w0)
    inst_list.append(instruction(Instype.Toffoli, [controls[0], controls[1], work_qubits[0]]))
    program_string += f"Toffoli {controls[0]}, {controls[1]}, {work_qubits[0]}\n"
    
    for i in range(1, num_ctrl - 2):
        # Tof(c_{i+1}, w_{i-1}, w_{i})
        c_idx = i + 1
        w_target = work_qubits[i]
        w_ctrl = work_qubits[i-1]
        inst_list.append(instruction(Instype.Toffoli, [controls[c_idx], w_ctrl, w_target]))
        program_string += f"Toffoli {controls[c_idx]}, {w_ctrl}, {w_target}\n"

    # 2. Target Operation
    # Tof(c_{last}, w_{last}, target)
    inst_list.append(instruction(Instype.Toffoli, [controls[-1], work_qubits[num_ctrl-3], target]))
    program_string += f"Toffoli {controls[-1]}, {work_qubits[num_ctrl-3]}, {target}\n"

    # 3. Uncompute Ladder (Backward)
    for i in range(num_ctrl - 3, 0, -1):
        c_idx = i + 1
        w_target = work_qubits[i]
        w_ctrl = work_qubits[i-1]
        inst_list.append(instruction(Instype.Toffoli, [controls[c_idx], w_ctrl, w_target]))
        program_string += f"Toffoli {controls[c_idx]}, {w_ctrl}, {w_target}\n"
        
    inst_list.append(instruction(Instype.Toffoli, [controls[0], controls[1], work_qubits[0]]))
    program_string += f"Toffoli {controls[0]}, {controls[1]}, {work_qubits[0]}\n"

    return inst_list, program_string


def generate_lcu_benchmark(hamiltonian_terms: List[str]) -> Tuple[List[instruction], str]:
    """
    Generates a quantum program implementing the LCU lemma for a list of Pauli terms.
    
    Args:
        hamiltonian_terms: List of Pauli strings, e.g., ["XX", "ZZ", "IY"].
                           Assumes all strings have same length.
                           Assumes equal coefficients (alpha = 1) for simplicity of PREP.
    
    Returns:
        Tuple of (instruction_list, program_string)
    """
    if not hamiltonian_terms:
        return [], ""

    num_data = len(hamiltonian_terms[0])
    num_terms = len(hamiltonian_terms)
    
    # 1. Calculate Helper Qubits
    # Select Register: log2(num_terms)
    num_select = math.ceil(math.log2(num_terms))
    if num_select == 0: num_select = 1 # Edge case for single term
    
    # Work Register: Needed for Multi-Control decomposition
    # Max controls we will apply is num_select. 
    # Work qubits needed = num_select - 2.
    num_work = get_required_helpers_for_mcx(num_select)
    
    total_helpers = num_select + num_work
    
    inst_list = []
    program_string = ""
    
    # --- Header ---
    program_string += f"q = alloc_data({num_data})\n"
    program_string += f"s = alloc_helper({total_helpers})\n"
    
    # Map helper names
    # s0...sk are Select qubits
    # s{k+1}... are Work qubits
    select_qubits = [f"s{i}" for i in range(num_select)]
    work_qubits = [f"s{num_select + i}" for i in range(num_work)]
    for s_q in select_qubits:
        inst_list.append(instruction(Instype.H, [s_q]))
        program_string += f"H {s_q}\n"
    for k, term in enumerate(hamiltonian_terms):
        if len(term) != num_data:
            raise ValueError("All Hamiltonian terms must have the same length.")
        # 2a. Condition on index k
        # We want the Select register to be all |1>s when the index is k.
        # So we apply X gates to the 0-bits of the binary representation of k.
        
        flipped_indices = []
        for bit_idx in range(num_select):
            # Check if bit at bit_idx is 0 in k
            # (using Little Endian for s0..sn)
            if not ((k >> bit_idx) & 1):
                inst_list.append(instruction(Instype.X, [select_qubits[bit_idx]]))
                program_string += f"X {select_qubits[bit_idx]}\n"
                flipped_indices.append(bit_idx)
                
        # 2b. Apply Controlled-Paulis
        # The control is ALWAYS the full `select_qubits` list (which represents |11..1> now)
        for data_idx, pauli_char in enumerate(term):
            target_q = f"q{data_idx}"
            
            if pauli_char == 'I':
                continue
                
            elif pauli_char == 'X':
                # Multi-Controlled X
                emit_mcx(select_qubits, target_q, work_qubits, inst_list, program_string)
                
            elif pauli_char == 'Z':
                # Controlled-Z = H * CX * H
                inst_list.append(instruction(Instype.H, [target_q]))
                program_string += f"H {target_q}\n"
                
                emit_mcx(select_qubits, target_q, work_qubits, inst_list, program_string)
                
                inst_list.append(instruction(Instype.H, [target_q]))
                program_string += f"H {target_q}\n"
                
            elif pauli_char == 'Y':
                # Controlled-Y = Sdg * CX * S
                # Note: Y = iXZ. Ignoring global phase i for LCU structure usually, 
                # but physically Y = S X S^dagger.
                # To implement controlled-Y: Sdg on target -> MCX -> S on target
                
                inst_list.append(instruction(Instype.Sdg, [target_q]))
                program_string += f"Sdg {target_q}\n"
                
                emit_mcx(select_qubits, target_q, work_qubits, inst_list, program_string)
                
                inst_list.append(instruction(Instype.S, [target_q]))
                program_string += f"S {target_q}\n"

        # 2c. Un-condition (Restore Select state)
        for bit_idx in flipped_indices:
            inst_list.append(instruction(Instype.X, [select_qubits[bit_idx]]))
            program_string += f"X {select_qubits[bit_idx]}\n"

    for s_q in select_qubits:
        inst_list.append(instruction(Instype.H, [s_q]))
        program_string += f"H {s_q}\n"
    for i, s_q in enumerate(select_qubits):
        inst_list.append(instruction(Instype.MEASURE, [s_q], classical_address=i))
        program_string += f"c{i} = MEASURE {s_q}\n"
        
        # Immediate release as per OS requirement
        inst_list.append(instruction(Instype.RELEASE, [s_q]))
        program_string += f"deallocate_helper({s_q})\n"
        
    # Clean up work qubits
    for w_q in work_qubits:
        inst_list.append(instruction(Instype.RELEASE, [w_q]))
        program_string += f"deallocate_helper({w_q})\n"

    program_string += "deallocate_data(q)\n"
    
    return inst_list, program_string



def norm_2_distance(matrix1: np.ndarray, matrix2: np.ndarray):
    """Calculates the sum of squared differences."""
    return np.sum(np.abs(matrix1 - matrix2) ** 2)

class LCUTester:
    def __init__(self, terms: list[str]):
        self.terms = terms
        self.num_data = len(terms[0])
        self.num_terms = len(terms)
        # Calculate normalization factor alpha.
        # In the provided generator, PREP uses Hadamards on 'num_select' qubits.
        # V|0> = sum |k> / sqrt(2^num_select).
        # The LCU lemma implements M / (2^num_select).
        import math
        self.num_select = math.ceil(math.log2(self.num_terms))
        if self.num_select == 0: self.num_select = 1
        self.alpha = 2 ** self.num_select

    def instructions_to_qiskit_unitary(self, inst_list, num_data, total_helpers) -> np.ndarray:
        """
        Converts the instruction list to a Qiskit Operator (Unitary).
        Ignores measurements to capture the pre-collapse state (W).
        """
        total_qubits = num_data + total_helpers
        qc = QuantumCircuit(total_qubits)

        # Mapping: q0...qN -> 0...N
        # s0...sM -> N...N+M
        # Helper function to parse 'qX' or 'sX' to integer index
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

            if t == Instype.H:
                qc.h(qs[0])
            elif t == Instype.X:
                qc.x(qs[0])
            elif t == Instype.Y:
                qc.y(qs[0])
            elif t == Instype.Z:
                qc.z(qs[0])
            elif t == Instype.S:
                qc.s(qs[0])
            elif t == Instype.Sdg:
                qc.sdg(qs[0])
            elif t == Instype.CNOT:
                qc.cx(qs[0], qs[1])
            elif t == Instype.Toffoli:
                qc.ccx(qs[0], qs[1], qs[2])
            elif t == Instype.MEASURE or t == Instype.RELEASE:
                # We stop before measurement to verify the Unitary W
                continue 
            else:
                print(f"Warning: instruction {t} not implemented in tester adapter.")

        # Get full unitary W
        return np.array(Operator(qc))

    def get_exact_operator(self) -> np.ndarray:
        """
        Constructs the matrix sum(Pk) manually.
        """
        dim = 2 ** self.num_data
        total_op = np.zeros((dim, dim), dtype=complex)

        for term in self.terms:
            # Note: Qiskit Pauli format is reversed string (Little Endian).
            # Term "XZ" means X on q0, Z on q1.
            # Qiskit "ZX" means Z on q1 (left), X on q0 (right).
            # So we reverse the term string for Qiskit.
            qiskit_str = term[::-1] 
            mat = Pauli(qiskit_str).to_matrix()
            total_op += mat
            
        return total_op

    def extract_block_encoding(self, W: np.ndarray, num_data, total_helpers) -> np.ndarray:
        """
        Extracts the top-left block of W where all helper qubits are |0>.
        
        Qiskit tensor ordering: |helper_last> ... |helper_0> |data_last> ... |data_0>
        (Assuming standard little-endian indexing where q0 is LSB)
        
        Wait, Qiskit Operator logic:
        If we added qubits q0..qN then s0..sM.
        Index 0 corresponds to |0...000>.
        Index 1 corresponds to |0...001> (q0=1).
        
        We want the subspace where s0..sM are ALL 0.
        These are the indices where bits [num_data ... total-1] are 0.
        """
        dim_data = 2 ** num_data
        
        # We just need the top-left dim_data x dim_data submatrix?
        # Only if the helper qubits are the MSBs (Most Significant Bits).
        # In instructions_to_qiskit_unitary, we mapped s_i to num_data + i.
        # So s_i are indeed the MSBs.
        # The state |0>_s |psi>_q corresponds to indices 0 to 2^num_data - 1.
        
        return W[0:dim_data, 0:dim_data]

    def run_test(self):
        print(f"Testing LCU for terms: {self.terms}")
        
        # 1. Generate Circuit Instructions
        inst_list, _ = generate_lcu_benchmark(self.terms)
        
        # Determine resource counts from list (simple scan)
        max_s = -1
        for inst in inst_list:
            for q in inst.get_qubitaddress():
                if q.startswith('s'):
                    idx = int(q[1:])
                    if idx > max_s: max_s = idx
        total_helpers = max_s + 1

        # 2. Get Circuit Unitary (Block Encoding W)
        W = self.instructions_to_qiskit_unitary(inst_list, self.num_data, total_helpers)
        
        # 3. Project onto <0|_helpers
        projected_op = self.extract_block_encoding(W, self.num_data, total_helpers)
        
        # 4. Construct Exact Target H
        exact_H = self.get_exact_operator()
        
        # 5. Compare
        # The LCU circuit implements H / alpha
        scaled_exact = exact_H / self.alpha
        
        dist = norm_2_distance(projected_op, scaled_exact)
        
        print(f"Normalization Factor (alpha): {self.alpha}")
        print(f"2-Norm Distance (Block vs H/alpha): {dist:.6e}")
        
        if dist < 1e-10:
            print("[PASS] LCU Implementation Correct.")
        else:
            print("[FAIL] Discrepancy detected.")
            print("Projected:\n", np.round(projected_op, 2))
            print("Target (Scaled):\n", np.round(scaled_exact, 2))





if __name__ == "__main__":
    # Test Case 1: Simple
    tester = LCUTester(["XX", "ZZ"])
    tester.run_test()
    
    print("-" * 30)
    
    # Test Case 2: Mixed with Identity
    # H = X I + Z Z
    tester2 = LCUTester(["XI", "ZZ"])
    tester2.run_test()