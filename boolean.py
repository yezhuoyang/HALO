import random
from typing import List, Tuple, Dict, Optional
from instruction import construct_qiskit_circuit, instruction, Instype, get_gate_type_name
# --- Assuming these are imported from your instruction.py ---
# from instruction import instruction, Instype, get_gate_type_name
# For the purpose of this script to run standalone, I will include 
# the necessary Enum/Class stubs. If you have the file, you can remove these stubs.
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator  # <- use AerSimulator (Qiskit 2.x)
# <STUBS START>
from enum import Enum


# <STUBS END>


# --- 1. Your Toffoli Decomposition ---

def toffoli_decomposition(qubit1: str, qubit2: str, qubit3: str) -> Tuple[List[instruction], str]:
    """
    We construct the Toffoli gate by decomposing it into a series of CNOT and single-qubit gates.
    """
    program_string = ""
    inst_list = []
    
    inst_list.append(instruction(Instype.H, [qubit3]))
    program_string += f"H {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit2, qubit3]))
    program_string += f"CNOT {qubit2}, {qubit3}\n"
    inst_list.append(instruction(Instype.Tdg, [qubit3]))
    program_string += f"Tdg {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit1, qubit3]))
    program_string += f"CNOT {qubit1}, {qubit3}\n"
    inst_list.append(instruction(Instype.T, [qubit3]))
    program_string += f"T {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit2, qubit3]))
    program_string += f"CNOT {qubit2}, {qubit3}\n"
    inst_list.append(instruction(Instype.Tdg, [qubit3]))
    program_string += f"Tdg {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit1, qubit3]))
    program_string += f"CNOT {qubit1}, {qubit3}\n"
    inst_list.append(instruction(Instype.T, [qubit2]))
    program_string += f"T {qubit2}\n"
    inst_list.append(instruction(Instype.T, [qubit3]))
    program_string += f"T {qubit3}\n"
    inst_list.append(instruction(Instype.H, [qubit3]))
    program_string += f"H {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit1, qubit2]))
    program_string += f"CNOT {qubit1}, {qubit2}\n"
    inst_list.append(instruction(Instype.T, [qubit1]))
    program_string += f"T {qubit1}\n"
    inst_list.append(instruction(Instype.Tdg, [qubit2]))
    program_string += f"Tdg {qubit2}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit1, qubit2]))
    program_string += f"CNOT {qubit1}, {qubit2}\n"
    
    return inst_list, program_string


# --- 2. AST Definitions ---

class BooleanNode: pass

class VarNode(BooleanNode):
    def __init__(self, index: int):
        self.index = index
    def __repr__(self): return f"q{self.index}"

class OpNode(BooleanNode):
    def __init__(self, op: str, left: BooleanNode, right: BooleanNode = None):
        self.op = op
        self.left = left
        self.right = right 
    def __repr__(self):
        if self.op == 'NOT': return f"~({self.left})"
        sym = '&' if self.op == 'AND' else '|'
        return f"({self.left} {sym} {self.right})"

def generate_random_ast(depth: int, num_vars: int) -> BooleanNode:
    if depth == 0 or random.random() < 0.15:
        return VarNode(random.randint(0, num_vars - 1))
    
    op = random.choice(['AND', 'OR', 'NOT'])
    if op == 'NOT':
        return OpNode('NOT', generate_random_ast(depth - 1, num_vars))
    else:
        return OpNode(op, generate_random_ast(depth - 1, num_vars), 
                          generate_random_ast(depth - 1, num_vars))


# --- 3. The Compiler ---

class BenchmarkCompiler:
    def __init__(self):
        self.inst_list: List[instruction] = []
        self.program_str = ""
        
        # Allocator State
        self.next_helper_id = 0
        self.max_helper_usage = 0
        self.current_helper_usage = 0

    def alloc(self) -> str:
        hid = self.next_helper_id
        self.next_helper_id += 1
        self.current_helper_usage += 1
        if self.current_helper_usage > self.max_helper_usage:
            self.max_helper_usage = self.current_helper_usage
        return f"s{hid}"

    def dealloc(self, addr: str):
        if not addr.startswith('s'): return
        self.current_helper_usage -= 1
        self.inst_list.append(instruction(Instype.RELEASE, [addr]))
        self.program_str += f"deallocate_helper({addr})\n"

    def emit_standard(self, inst_type: Instype, qubits: List[str]):
        """Emits a standard gate (not Toffoli)."""
        inst = instruction(inst_type, qubits)
        self.inst_list.append(inst)
        q_str = ", ".join(qubits)
        cmd = get_gate_type_name(inst_type)
        self.program_str += f"{cmd} {q_str}\n"

    def emit_toffoli_decomposed(self, q1: str, q2: str, q3: str):
        """Injects the decomposed Toffoli sequence."""
        new_insts, new_str = toffoli_decomposition(q1, q2, q3)
        self.inst_list.extend(new_insts)
        self.program_str += new_str

    def _apply_logic(self, op: str, a: str, b: str, out: str):
        """Applies logic. REPLACES standard Toffoli with the decomposition."""
        if op == 'NOT':
            self.emit_standard(Instype.CNOT, [a, out])
            self.emit_standard(Instype.X, [out])
        
        elif op == 'AND':
            # Replaced single Toffoli with decomposition
            self.emit_toffoli_decomposed(a, b, out)
        
        elif op == 'OR':
            # De Morgan: ~(~a & ~b)
            # 1. Flip inputs
            self.emit_standard(Instype.X, [a])
            self.emit_standard(Instype.X, [b])
            
            # 2. Decomposed Toffoli
            self.emit_toffoli_decomposed(a, b, out)
            
            # 3. Restore inputs and Flip output
            self.emit_standard(Instype.X, [b]) 
            self.emit_standard(Instype.X, [a]) 
            self.emit_standard(Instype.X, [out])

    def recurse(self, node: BooleanNode) -> str:
        # Base Case: Variable
        if isinstance(node, VarNode):
            return f"q{node.index}"

        # Recursive Step
        if isinstance(node, OpNode):
            # A. Compute Children
            addr_left = self.recurse(node.left)
            addr_right = None
            if node.right:
                addr_right = self.recurse(node.right)

            # B. Allocate Output
            addr_out = self.alloc()

            # C. Compute Logic (Decomposed)
            self._apply_logic(node.op, addr_left, addr_right, addr_out)

            # D. Uncompute Children (Backward) & Release
            if node.right:
                self.unrecurse(node.right, addr_right)
            self.unrecurse(node.left, addr_left)

            return addr_out

    def unrecurse(self, node: BooleanNode, addr_curr: str):
        if isinstance(node, VarNode):
            return

        if isinstance(node, OpNode):
            # A. Recompute inputs to enable uncomputation (The naive churn)
            addr_left = self.recurse(node.left)
            addr_right = None
            if node.right:
                addr_right = self.recurse(node.right)

            # B. Uncompute Logic (Same logic applies for reversible gates)
            self._apply_logic(node.op, addr_left, addr_right, addr_curr)

            # C. Release Self
            self.dealloc(addr_curr)

            # D. Clean up the inputs we just re-computed
            if node.right:
                self.unrecurse(node.right, addr_right)
            self.unrecurse(node.left, addr_left)


# --- 4. Main Generator ---

def generate_decomposed_benchmark(num_vars: int, depth: int) -> Tuple[List[instruction], str]:
    # 1. Generate Expression
    root = generate_random_ast(depth, num_vars)
    
    # 2. Compile
    compiler = BenchmarkCompiler()
    
    # Compile the tree
    output_addr = compiler.recurse(root)
    
    # 3. Final Measure & Cleanup
    compiler.program_str += f"c0 = MEASURE {output_addr}\n"
    compiler.inst_list.append(instruction(Instype.MEASURE, [output_addr], classical_address=0))
    
    compiler.dealloc(output_addr)
    compiler.program_str += "deallocate_data(q)\n"

    # 4. Construct Header
    header = f"# Naive Boolean Benchmark (Decomposed Toffoli)\n"
    header += f"# Expression: {root}\n"
    header += f"q = alloc_data({num_vars})\n"
    header += f"s = alloc_helper({compiler.max_helper_usage})\n"
    header += "set_shot(100)\n\n"
    
    return compiler.inst_list, header + compiler.program_str





# Ensure we have the necessary imports from previous modules
# from instruction import instruction, Instype, construct_qiskit_circuit
# from qiskit_aer import AerSimulator
# from qiskit import transpile

def evaluate_ast(node: 'BooleanNode', input_values: Dict[int, int]) -> int:
    """
    Classically evaluates the boolean AST for a specific set of input values.
    
    Args:
        node: The root of the BooleanNode tree.
        input_values: A dictionary mapping variable indices to 0 or 1.
                      e.g., {0: 1, 1: 0, 2: 1}
    Returns:
        1 if True, 0 if False.
    """
    if isinstance(node, VarNode):
        # Return the value of the variable from inputs (default to 0 if missing)
        return input_values.get(node.index, 0)
    
    if isinstance(node, OpNode):
        if node.op == 'NOT':
            # 1 - val is equivalent to boolean NOT for 0/1 integers
            return 1 - evaluate_ast(node.left, input_values)
        
        val_left = evaluate_ast(node.left, input_values)
        # Short-circuit logic is not strictly necessary for correctness but good for perfromance
        # However, for full evaluation we just compute both.
        
        if node.op == 'AND':
            val_right = evaluate_ast(node.right, input_values)
            return val_left & val_right
            
        if node.op == 'OR':
            val_right = evaluate_ast(node.right, input_values)
            return val_left | val_right
            
    raise ValueError(f"Unknown node type: {type(node)}")


def get_resource_counts(inst_list: List[instruction], num_vars: int) -> Tuple[int, int, int]:
    """
    Scans the instruction list to find the highest index of helper qubits used.
    
    Returns:
        (num_data_qubits, num_syndrome_qubits, num_classical_bits)
    """
    max_s_index = -1
    max_c_index = 0 # Assume at least c0 exists if we measure
    
    for inst in inst_list:
        # Check qubit addresses
        for q_addr in inst.get_qubitaddress():
            if q_addr.startswith('s'):
                idx = int(q_addr[1:])
                if idx > max_s_index:
                    max_s_index = idx
        
        # Check classical addresses
        if inst.get_type() == Instype.MEASURE:
            c_addr = inst.get_classical_address()
            if c_addr > max_c_index:
                max_c_index = c_addr

    # Data qubits are fixed by input, helpers are max_index + 1
    return num_vars, max_s_index + 1, max_c_index + 1

def verify_circuit_correctness(inst_list: List[instruction], root_node: 'BooleanNode', num_vars: int) -> bool:
    """
    Verifies that the compiled quantum circuit matches the boolean logic of the AST
    for ALL possible classical inputs.
    
    Args:
        inst_list: The compiled instructions (without input initialization).
        root_node: The classical AST ground truth.
        num_vars: The number of input variables.
        
    Returns:
        True if the circuit is correct for all inputs, False otherwise.
    """
    
    # 0. Print Logic Information
    print(f"\n{'='*60}")
    print(f"Generated Logical Expression: {root_node}")
    print(f"{'='*60}\n")

    # 1. Determine circuit resources
    data_n, syn_n, meas_n = get_resource_counts(inst_list, num_vars)
    

    print(f"Circuit Resources: Data Qubits = {data_n}, Helper Qubits = {syn_n}, Classical Bits = {meas_n}\n")

    # 2. Setup Simulator
    sim = AerSimulator()
    total_inputs = 2 ** num_vars
    print(f"Starting verification for {total_inputs} input combinations...")
    
    # 3. Iterate over all possible input bitstrings
    for i in range(total_inputs):
        # Generate input map for this iteration
        # e.g., i=5 (101 binary) -> {0: 1, 1: 0, 2: 1}
        input_map = {}
        init_instructions = []
        
        debug_input_str = []
        
        for bit_idx in range(num_vars):
            bit_val = (i >> bit_idx) & 1
            input_map[bit_idx] = bit_val
            
            # If input bit is 1, we must add an X gate to initialize the qubit
            if bit_val == 1:
                # We prepend these instructions
                init_instructions.append(instruction(Instype.X, [f"q{bit_idx}"]))
                debug_input_str.append(f"q{bit_idx}=1")
            else:
                debug_input_str.append(f"q{bit_idx}=0")
        
        # 4. Calculate Expected Result (Classical Oracle)
        expected_bool = evaluate_ast(root_node, input_map)
        
        # 5. Build Verification Circuit (Input Init + Benchmark Logic)
        full_circuit_insts = init_instructions + inst_list
        
        qc = construct_qiskit_circuit(data_n, syn_n, meas_n, full_circuit_insts)
        tqc = transpile(qc, sim)
        
        # 6. Run Simulation
        # We use a small number of shots because the circuit should be deterministic (boolean logic)
        result = sim.run(tqc, shots=100).result()
        counts = result.get_counts(tqc)
        
        # 7. Check Result
        # The result key is a hex or binary string depending on qiskit version/settings, usually binary '0' or '1'
        # We look for the most frequent result.
        measured_state = max(counts, key=counts.get)
        
        # Note: Qiskit keys are little-endian or simple strings. 
        # For a single measurement c0, the key is usually just "0" or "1".
        # If there are multiple classical bits, we might need to parse.
        # Assuming we care about c0 (the Least Significant Bit of the result)
        measured_val = int(measured_state, 2) & 1
        
        # --- MIDDLE OUTPUT: PRINT CURRENT STATUS ---
        formatted_input = f"[{', '.join(debug_input_str)}]"
        print(f"Input: {formatted_input:<25} | Ideal: {expected_bool} | Quantum Counts: {counts}")

        if measured_val != expected_bool:
            print(f"\n[FAIL] Mismatch Detected!")
            print(f"       Input: {formatted_input}")
            print(f"       Expected: {expected_bool}, Got: {measured_val}")
            print(f"       AST: {root_node}\n")
            return False

    print(f"\n[PASS] Circuit verified correctly for all {total_inputs} inputs.")
    return True












if __name__ == "__main__":
    # Test Logic
    # 1. Generate a small random benchmark
    num_vars = 8
    depth = 4
    root = generate_random_ast(depth, num_vars)
    print(f"Generated AST: {root}")
    
    # 2. Compile it (using the Decomposed Toffoli compiler from previous step)
    compiler = BenchmarkCompiler()
    out_addr = compiler.recurse(root)
    
    # Add final measurement
    compiler.inst_list.append(instruction(Instype.MEASURE, [out_addr], classical_address=0))
    compiler.dealloc(out_addr) # Release final result qubit
    
    # 3. Verify it
    # Note: We pass the raw instruction list. The verification function handles X-gate injection.
    is_correct = verify_circuit_correctness(compiler.inst_list, root, num_vars)
    
    if is_correct:
        print("SUCCESS: The quantum circuit implements the classical logic perfectly.")
    else:
        print("FAILURE: Logic mismatch detected.")