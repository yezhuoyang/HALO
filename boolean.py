import random
from typing import List, Tuple, Dict
from instruction import instruction, Instype, get_gate_type_name


# --- AST Nodes ---
class BooleanNode: pass

class VarNode(BooleanNode):
    def __init__(self, index: int):
        self.index = index
    def __repr__(self): return f"q{self.index}"

class OpNode(BooleanNode):
    def __init__(self, op: str, left: BooleanNode, right: BooleanNode = None):
        self.op = op
        self.left = left
        self.right = right # None for NOT
    def __repr__(self):
        if self.op == 'NOT': return f"~({self.left})"
        sym = '&' if self.op == 'AND' else '|'
        return f"({self.left} {sym} {self.right})"

# --- Logic Generator ---
def generate_random_ast(depth: int, num_vars: int) -> BooleanNode:
    if depth == 0 or random.random() < 0.15:
        return VarNode(random.randint(0, num_vars - 1))
    
    op = random.choice(['AND', 'OR', 'NOT'])
    if op == 'NOT':
        return OpNode('NOT', generate_random_ast(depth - 1, num_vars))
    else:
        return OpNode(op, generate_random_ast(depth - 1, num_vars), 
                          generate_random_ast(depth - 1, num_vars))

# --- The Compiler ---

class BenchmarkCompiler:
    def __init__(self, decompose_toffoli: bool = False):
        self.inst_list: List[instruction] = []
        self.program_str = ""
        self.decompose_toffoli = decompose_toffoli
        
        # Allocator State
        self.active_helpers: Dict[int, bool] = {} # map id -> is_active
        self.next_helper_id = 0
        self.max_helper_usage = 0
        self.current_helper_usage = 0

    def alloc(self) -> str:
        """
        Allocates a logical helper ID. 
        In a real OS, this maps to a physical qubit. 
        Here, we just track IDs to ensure unique names in the script.
        """
        # We perform a simple linear search for a free ID or create a new one
        # to simulate the OS's perspective of "requesting a resource"
        # However, for the script generation, we usually just increment to ensure unique variable names
        # and let the OS map them. To make the benchmark readable, let's just increment.
        
        hid = self.next_helper_id
        self.next_helper_id += 1
        
        self.current_helper_usage += 1
        if self.current_helper_usage > self.max_helper_usage:
            self.max_helper_usage = self.current_helper_usage
            
        return f"s{hid}"

    def dealloc(self, addr: str):
        if not addr.startswith('s'): return
        self.current_helper_usage -= 1
        # Add the explicit system call instruction
        self.inst_list.append(instruction(Instype.RELEASE, [addr]))
        self.program_str += f"deallocate_helper({addr})\n"

    def emit(self, inst_type: Instype, qubits: List[str], comment: str = ""):
        # Check for decomposition
        if self.decompose_toffoli and inst_type == Instype.Toffoli:
            # Assuming toffoli_decomposition function exists in your scope or we import it
            # For this snippet, I will just emit the standard line for clarity
            # unless you include the decomposition logic here.
            pass
            
        inst = instruction(inst_type, qubits)
        self.inst_list.append(inst)
        # Use the string representation logic or custom format
        q_str = ", ".join(qubits)
        cmd = get_gate_type_name(inst_type)
        self.program_str += f"{cmd} {q_str}"
        if comment: self.program_str += f"  # {comment}"
        self.program_str += "\n"

    # --- Recursive Compilation Core ---

    def recurse(self, node: BooleanNode) -> str:
        """
        Computes the result of 'node' into a newly allocated helper.
        Returns the address of that helper.
        """
        # 1. Base Case: Variable (Data Qubit)
        if isinstance(node, VarNode):
            return f"q{node.index}"

        # 2. Recursive Step
        if isinstance(node, OpNode):
            # A. Compute Children
            # We must compute them, store their addresses
            addr_left = self.recurse(node.left)
            addr_right = None
            if node.right:
                addr_right = self.recurse(node.right)

            # B. Allocate Output
            addr_out = self.alloc()

            # C. Compute Logic (Forward)
            self._apply_logic(node.op, addr_left, addr_right, addr_out)

            # D. Uncompute Children (Backward) & Release
            # This is the "Naive" part: we immediately clean up inputs
            if node.right:
                self.unrecurse(node.right, addr_right)
            self.unrecurse(node.left, addr_left)

            return addr_out

    def unrecurse(self, node: BooleanNode, addr_curr: str):
        """
        Uncomputes 'node' (which currently sits in 'addr_curr').
        Then releases 'addr_curr'.
        """
        # 1. Base Case: Variable
        # Data qubits are persistent, we do not dealloc or uncompute them.
        if isinstance(node, VarNode):
            return

        # 2. Recursive Uncompute
        if isinstance(node, OpNode):
            # A. To uncompute Self, we need inputs. Recompute them!
            # This causes the "Bennett" exponential blowup / high churn
            addr_left = self.recurse(node.left)
            addr_right = None
            if node.right:
                addr_right = self.recurse(node.right)

            # B. Uncompute Logic (Inverse)
            # Since standard boolean gates (And/Or) are their own inverse logic 
            # (when applied to result + inputs), we apply the same gates.
            self._apply_logic(node.op, addr_left, addr_right, addr_curr)

            # C. Release Self
            # We are now back to |0>, so we release immediately
            self.dealloc(addr_curr)

            # D. Clean up the inputs we just re-computed
            if node.right:
                self.unrecurse(node.right, addr_right)
            self.unrecurse(node.left, addr_left)

    def _apply_logic(self, op: str, a: str, b: str, out: str):
        """Helper to emit gates for operations"""
        if op == 'NOT':
            # Logic: out = ~a. 
            # 1. Copy a to out (CNOT)
            # 2. Flip out (X)
            self.emit(Instype.CNOT, [a, out])
            self.emit(Instype.X, [out])
        
        elif op == 'AND':
            # Logic: out = a & b (Toffoli)
            self.emit(Instype.Toffoli, [a, b, out])
        
        elif op == 'OR':
            # Logic: out = a | b = ~(~a & ~b)
            # 1. Invert inputs (non-destructively by wrapping)
            #    Wait, we can't invert 'a' in place if it's a shared node?
            #    In this strict tree, nodes aren't shared. Safe to invert in place temporarily.
            self.emit(Instype.X, [a])
            self.emit(Instype.X, [b])
            self.emit(Instype.Toffoli, [a, b, out])
            self.emit(Instype.X, [b]) # Restore b
            self.emit(Instype.X, [a]) # Restore a
            self.emit(Instype.X, [out]) # Flip output

def generate_naive_benchmark(num_vars: int, depth: int) -> Tuple[List[instruction], str]:
    # 1. Generate Expression
    root = generate_random_ast(depth, num_vars)
    
    # 2. Compile
    compiler = BenchmarkCompiler(decompose_toffoli=False)
    
    # We delay header generation until after compilation to know 'max_helper_usage'
    output_addr = compiler.recurse(root)
    
    # 3. Final Measure & Cleanup
    compiler.program_str += f"c0 = MEASURE {output_addr}\n"
    compiler.inst_list.append(instruction(Instype.MEASURE, [output_addr], classical_address=0))
    
    # Final dealloc of the root result
    compiler.dealloc(output_addr)
    compiler.program_str += "deallocate_data(q)\n"

    # 4. Construct Header
    header = f"# Naive Boolean Benchmark\n"
    header += f"# Expression: {root}\n"
    header += f"q = alloc_data({num_vars})\n"
    header += f"s = alloc_helper({compiler.max_helper_usage})\n" # Exact max needed
    header += "set_shot(100)\n\n"
    
    return compiler.inst_list, header + compiler.program_str

# --- Usage Example ---
if __name__ == "__main__":
    # Generate a circuit with 4 inputs and depth 3
    # This will generate A LOT of alloc/dealloc due to recomputation
    insts, text = generate_naive_benchmark(num_vars=4, depth=3)
    print(text)