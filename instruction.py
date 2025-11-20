from enum import Enum
from typing import List, Tuple, Optional
import qiskit.qasm2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator  # <- use AerSimulator (Qiskit 2.x)
import pickle


"""
instruction.py

This module defines the atomic structure of a user process: the Instruction class
Each instruction has a type (e.g., H, CNOT, MEASURE), a list of target qubits, and a timestamp.

Classes:
    Instruction -- Represents a quantum gate operation or measurement

Example:
    >>> inst = Instruction(type=Instype.H, qubit="q1")
    >>> print(inst)
    H gate on qubit 0 at time 5
"""

class Instype(Enum):
    H = 1
    X = 2
    Y = 3
    Z = 3
    T = 4
    Tdg = 5
    U = 6
    S = 7
    Sdg = 8
    SX = 9
    RZ = 10
    RX = 11
    RY = 12
    U3 = 13
    Toffoli = 14
    CNOT = 15
    CH = 16
    SWAP = 17
    CSWAP = 18
    CP = 19
    RESET = 20
    MEASURE = 21
    RELEASE = 22



def get_gate_type_name(type: Instype) -> str:
    """
    Get the name of the gate type.
    
    Args:
        type (Instype): The type of the instruction.
    
    Returns:
        str: The name of the instruction type.
    """
    match type:
        case Instype.H:
            return "H"
        case Instype.X:
            return "X"  
        case Instype.Y:
            return "Y"
        case Instype.Z:
            return "Z"
        case Instype.T:
            return "T"
        case Instype.U:
            return "U"
        case Instype.Tdg:
            return "Tdg"
        case Instype.S:
            return "S"
        case Instype.Sdg:
            return "Sdg"
        case Instype.SX:
            return "SX"
        case Instype.RZ:
            return "RZ"
        case Instype.RX:
            return "RX"
        case Instype.RY:
            return "RY"
        case Instype.U3:
            return "U3"
        case Instype.Toffoli:
            return "Toffoli"
        case Instype.CNOT:
            return "CNOT"
        case Instype.SWAP:
            return "SWAP"
        case Instype.CSWAP:
            return "CSWAP"
        case Instype.CP:   
            return "CP"
        case Instype.CH:
            return "CH"
        case Instype.RESET:
            return "RESET"
        case Instype.MEASURE:
            return "MEASURE"
        case Instype.RELEASE:
            return "RELEASE"
        case _:
            raise ValueError("Unknown instruction type")




class instruction:

    """
    Initialize a new instruction.

    qubit address are represented as string.

    For data qubit, the convention for data qubit k is qk,
    for helper qubit, the convention of helper qubit k is sk
    """
    def __init__(self, type:Instype, qubitaddress:List[str],classical_address: int=None, reset_address: int=None, params: List[float]=None) -> None:
        self._type=type
        self._qubitaddress=qubitaddress
        self._scheduled_mapped_address={} # This is the physical address after scheduling
        self._params=params if params is not None else [] # The rotation angles for rotation gates, for example, for RZ(0.2*pi), params=[0.2]
        if self._type==Instype.MEASURE:
            if classical_address is None:
                raise ValueError("Classical address must be provided for MEASURE instruction.")
            self._classical_address=classical_address
        if self._type==Instype.RESET:
            if reset_address is None:
                raise ValueError("Reset address must be provided for RESET instruction.")
            self._reset_address=reset_address
        self._helper_qubit_count=0
        for addr in qubitaddress:
            if addr.startswith('s'):
                self._helper_qubit_count+=1



    def get_reset_address(self) -> int:
        """
        Get the reset address associated with the instruction.
        
        Returns:
            int: The reset address for the instruction.
        """
        if self._type != Instype.RESET:
            raise ValueError("Reset address is only applicable for RESET instructions.")
        return self._reset_address



    def get_helper_qubit_count(self) -> int:
        """
        Get the number of helper qubits involved in the instruction.
        
        Returns:
            int: The number of helper qubits.
        """
        return self._helper_qubit_count



    def get_all_helper_qubit_addresses(self) -> List[str]:
        """
        Get the addresses of all helper qubits involved in the instruction.
        
        Returns:
            List[str]: A list of helper qubit addresses.
        """
        helper_qubit_addresses = []
        for addr in self._qubitaddress:
            if addr.startswith('s'):
                helper_qubit_addresses.append(addr)
        return helper_qubit_addresses



    def get_all_data_qubit_addresses(self) -> List[str]:
        """
        Get the addresses of all data qubits involved in the instruction.
        
        Returns:
            List[str]: A list of data qubit addresses.
        """
        data_qubit_addresses = []
        for addr in self._qubitaddress:
            if addr.startswith('q'):
                data_qubit_addresses.append(addr)
        return data_qubit_addresses



    def is_release_helper_qubit(self) -> bool:
        """
        Check if the instruction is a release operation.
        
        Returns:
            bool: True if the instruction is a release, False otherwise.
        """
        return self._type == Instype.RELEASE



    def is_system_call(self) -> bool:
        """
        Check if the instruction is a system call.
        
        Returns:
            bool: True if the instruction is a system call, False otherwise.
        """
        return self._type in {Instype.RELEASE}



    def is_two_qubit_gate(self) -> bool:
        """
        Check if the instruction is a two-qubit gate.
        
        Returns:
            bool: True if the instruction is a two-qubit gate, False otherwise.
        """
        return self._type in {Instype.CNOT, Instype.CH, Instype.SWAP, Instype.CP}


    def get_params(self) -> List[float]:
        """
        Return the parameters associated with the instruction.
        
        Returns:
            List[float]: The list of parameters for the instruction.
        """
        return self._params
    

    def get_classical_address(self) -> int:
        """
        Get the classical address associated with the instruction.
        
        Returns:
            int: The classical address for the instruction.
        """
        if self._type != Instype.MEASURE:
            raise ValueError("Classical address is only applicable for MEASURE instructions.")
        return self._classical_address


    def reset_mapping(self):
        self._scheduled_mapped_address={} # This is the physical address after scheduling


    def is_reset(self) -> bool:
        """
        Check if the instruction is a reset operation.
        
        Returns:
            bool: True if the instruction is a reset, False otherwise.
        """
        return self._type == Instype.RESET



    def is_measurement(self) -> bool:
        """
        Check if the instruction is a measurement operation.
        
        Returns:
            bool: True if the instruction is a measurement, False otherwise.
        """
        return self._type == Instype.MEASURE



    def set_scheduled_mapped_address(self, virtualaddress: str, physicaladdress: int):
        """
        Set the scheduled mapped address for a given qubit address.
        
        Args:
            qubitaddress (virtualAddress): The virtual address of the qubit.
            physicaladdress (int): The physical address to which the qubit is mapped.
        """
        self._scheduled_mapped_address[virtualaddress] = physicaladdress


    def get_scheduled_mapped_address(self, virtualAddress: str) -> int:
        """
        Get the scheduled mapped addresses for the instruction.
        
        Returns:
            dict: A dictionary mapping virtual qubit addresses to physical addresses.
        """
        return self._scheduled_mapped_address[virtualAddress]


    def get_processID(self) -> int:
        """
        Get the process ID associated with the instruction.
        """
        return self._processID


    def get_type(self) -> Instype:
        """
        Get the type of the instruction.
        """
        return self._type

    def get_qubitaddress(self) -> List[str]:
        """
        Get the list of qubit addresses associated with the instruction.
        """
        return self._qubitaddress


    def __str__(self) -> str:
        outputstr=""
        match self._type:
            case Instype.H:
                outputstr+="H"
            case Instype.X:
                outputstr+="X"
            case Instype.Y:
                outputstr+="Y"
            case Instype.Z:
                outputstr+="Z"
            case Instype.T:
                outputstr+="T"
            case Instype.Tdg:
                outputstr+="Tdg"
            case Instype.S:
                outputstr+="S"
            case Instype.Sdg:
                outputstr+="Sdg"
            case Instype.SX:
                outputstr+="SX"
            case Instype.RZ:
                outputstr+="RZ("+str(self._params[0])+"*pi)"
            case Instype.RX:
                outputstr+="RX("+str(self._params[0])+"*pi)"
            case Instype.RY:
                outputstr+="RY("+str(self._params[0])+"*pi)"
            case Instype.U3:
                outputstr+="U3("+str(self._params[0])+"*pi, "+str(self._params[1])+"*pi, "+str(self._params[2])+"*pi)"
            case Instype.U:
                outputstr+="U("+str(self._params[0])+"*pi, "+str(self._params[1])+"*pi, "+str(self._params[2])+"*pi)"
            case Instype.Toffoli:
                outputstr+="Toffoli"    
            case Instype.CNOT:
                outputstr+="CNOT"
            case Instype.CH:
                outputstr+="CH"
            case Instype.SWAP:
                outputstr+="SWAP"
            case Instype.CSWAP:
                outputstr+="CSWAP"
            case Instype.CP:
                outputstr+="CP("+str(self._params[0])+"*pi)"
            case Instype.RESET:
                outputstr+="RESET"
            case Instype.MEASURE:
                outputstr+="c" + str(self._classical_address) + "=MEASURE"
            case Instype.RELEASE:
                outputstr+="RELEASE"
        outputstr+=" qubit("
        for i, addr in  enumerate(self._qubitaddress):
            outputstr+=addr
            if addr in self._scheduled_mapped_address:
                outputstr+="->"+str(self._scheduled_mapped_address[addr])
            if i!=len(self._qubitaddress)-1:
                outputstr+=", "
        outputstr+=")"
        return outputstr

    def __repr__(self):
        return self.__str__()





def construct_qiskit_circuit(num_data_qubit: int, num_syndrome_qubit: int, num_classical_bits: int, instruction_list: List[instruction]) -> QuantumCircuit:
    """
    Construct a qiskit circuit from the instruction list.
    Also help to visualize the circuit.
    """
    dataqubit = QuantumRegister(num_data_qubit, "q")

    # Second part: 2 qubits named 's'
    syndromequbit = QuantumRegister(num_syndrome_qubit, "s")

    # Classical registers (optional, if you want measurements)
    classicalbits = ClassicalRegister(num_classical_bits, "c")

    # Combine them into one circuit
    qiskit_circuit = QuantumCircuit(dataqubit,  syndromequbit, classicalbits)

    for inst in instruction_list:
        if inst.is_system_call():
            continue
        addresses = inst.get_qubitaddress()
        qiskitaddress=[]
        for addr in addresses:
            if addr.startswith('s'):
                qiskitaddress.append(syndromequbit[int(addr[1:])])
            else:
                qiskitaddress.append(dataqubit[int(addr[1:])])
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
                qiskit_circuit.u3(params[0], params[1], params[2], qiskitaddress[0])
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
                qiskit_circuit.reset(qiskitaddress[0])
            case Instype.MEASURE:
                classical_address=inst.get_classical_address()
                qiskit_circuit.measure(qiskitaddress[0], classical_address)

    return qiskit_circuit





import re


_param_list_re = re.compile(r"\(([^)]*)\)")

def _parse_float_list(s: str) -> List[float]:
    """
    Parse a comma-separated list of floats inside parentheses.
    Accepts whitespace; returns [] if s is empty/only spaces.
    """
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [float(p) for p in parts if p]


def _grab_params(text_after_gate: str) -> Tuple[List[float], str]:
    """
    Given '(<params>) rest', return (params_list, rest_without_paren_prefix).
    If no parentheses, returns ([], original_text).
    """
    m = _param_list_re.match(text_after_gate.strip())
    if not m:
        return [], text_after_gate.strip()
    params = _parse_float_list(m.group(1))
    # remove the '(...)' prefix from the remaining text
    remainder = text_after_gate.strip()[m.end():].strip()
    return params, remainder


def parse_program_from_file(file_path: str) -> Tuple[List[instruction], int, int, int]:
    """
    Parse the custom process program DSL into a `process` object.
    Return (inst_list, data_n, syn_n,measure_n).

    Expects header lines:
        q = alloc_data(N)
        s = alloc_helper(M)
        set_shot(S)

    Supports gates like:
        H q0
        CNOT q0, q2
        RX(1.234) q3
        U(theta, phi, lam) q1
        ...
        c0 = MEASURE q2
        deallocate_data(q)
        deallocate_helper(s)
    """
    with open(file_path, "r") as file:
        qasm_code = file.read()    

    # normalize and split lines
    raw_lines = [ln.strip() for ln in qasm_code.strip().splitlines() if ln.strip()]

    # ---- Pass 1: read header (alloc & shots) ----
    data_n = 0
    syn_n = 0
    measure_n = 0

    alloc_data_re = re.compile(r"^q\s*=\s*alloc_data\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    alloc_syn_re  = re.compile(r"^s\s*=\s*alloc_helper\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    set_shot_re   = re.compile(r"^set_shot\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)

    # We’ll keep the non-header lines to parse as instructions
    instr_lines: List[str] = []

    for ln in raw_lines:
        if m := alloc_data_re.match(ln):
            data_n = int(m.group(1))
            continue
        if m := alloc_syn_re.match(ln):
            syn_n = int(m.group(1))
            continue
        if m := set_shot_re.match(ln):
            shots = int(m.group(1))
            continue
        # not a header line → it’s an instruction or dealloc
        instr_lines.append(ln)


    inst_list=[]


    # ---- Parsers for instruction lines ----
    # Simple 1-qubit no-parameter gates
    oneq_no_param = {
        "h": Instype.H,
        "x": Instype.X,
        "y": Instype.Y,
        "z": Instype.Z,
        "t": Instype.T,
        "tdg": Instype.Tdg,
        "s": Instype.S,
        "sdg": Instype.Sdg,
        "sx": Instype.SX,
        "reset": Instype.RESET,
    }

    # Two-qubit gates without params
    twoq_no_param = {
        "cnot": Instype.CNOT,
        "cx": Instype.CNOT,   # alias
        "ch": Instype.CH,
        "swap": Instype.SWAP,
    }

    # Three-qubit no-param
    threeq_no_param = {
        "cswap": Instype.CSWAP,
        "toffoli": Instype.Toffoli,
        "ccx": Instype.Toffoli,  # alias
    }

    # Helpers
    def add_oneq_gate(kind: Instype, addr_tok: str, params: Optional[List[float]] = None):
        if params:
            inst_list.append(instruction(kind, [addr_tok], params=params))
        else:
            inst_list.append(instruction(kind, [addr_tok]))

    def add_twoq_gate(kind: Instype, tok_a: str, tok_b: str, params: Optional[List[float]] = None):
        if params:
            inst_list.append(instruction(kind, [tok_a, tok_b], params=params))
        else:
            inst_list.append(instruction(kind, [tok_a, tok_b]))

    def add_threeq_gate(kind: Instype, tok_a: str, tok_b: str, tok_c: str):
        inst_list.append(instruction(kind, [tok_a, tok_b, tok_c]))

    # Regexes for instruction shapes
    measure_re = re.compile(r"^c\s*(\d+)\s*=\s*MEASURE\s+([qs]\d+)\s*$", re.IGNORECASE)
    # dealloc lines (we'll synthesize syscalls ourselves, but allow them to appear)
    dealloc_q_re = re.compile(r"^deallocate_data\s*\(\s*q\s*\)\s*$", re.IGNORECASE)
    dealloc_s_re = re.compile(
        r"^\s*deallocate_helper\s*\(\s*(s\d+)\s*\)\s*$",
        re.IGNORECASE
    )

    for ln in instr_lines:
        # Skip optional trailing dealloc directives; we'll add syscalls at the end.
        if dealloc_q_re.match(ln):
            continue

        # Measurement: "c0 = MEASURE q2"
        m = measure_re.match(ln)
        if m:
            measure_n += 1
            cidx = int(m.group(1))
            qtok = m.group(2)
            inst_list.append(instruction(Instype.MEASURE, [qtok], classical_address=cidx))
            continue

        # deallocate_helper(s0)
        mdealloc_s = dealloc_s_re.match(ln)
        if mdealloc_s:
            sreg = mdealloc_s.group(1)
            # synthesize RELEASE instructions for all helper qubits
            inst_list.append(instruction(Instype.RELEASE, [sreg]))
            continue


        # Tokenize: first word is gate mnemonic (maybe with params), the rest are args
        # We'll manually pull params when present (RX/RY/RZ/U/CU1).
        # Examples:
        #   "H q0"
        #   "RX(1.23) q1"
        #   "U(θ,φ,λ) q3"
        #   "CNOT q0, q2"
        #   "CU1(π/2) q0, q1"
        parts = ln.split(None, 1)
        if not parts:
            continue
        gate_full = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
        gate = gate_full.lower()

        # Parameterized single-qubit
        if gate.startswith("rx"):
            params, rest2 = _grab_params(rest if gate == "rx" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RX, target, params=params)
            continue

        if gate.startswith("ry"):
            params, rest2 = _grab_params(rest if gate == "ry" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RY, target, params=params)
            continue

        if gate.startswith("rz"):
            params, rest2 = _grab_params(rest if gate == "rz" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RZ, target, params=params)
            continue

        if gate.startswith("u3"):  # treat like U
            params, rest2 = _grab_params(rest if gate == "u3" else gate_full[2:] + rest)
            target = rest2
            if len(params) != 3:
                raise ValueError(f"U3 expects 3 parameters, got {params}")
            add_oneq_gate(Instype.U3, target, params=params)
            continue

        if gate.startswith("u(") or gate == "u":
            # handle forms like "U( ... ) q0" or weird tokenization
            params, rest2 = _grab_params(ln[1:] if gate_full.lower().startswith("u(") else rest)
            target = rest2
            if len(params) != 3:
                raise ValueError(f"U expects 3 parameters, got {params}")
            add_oneq_gate(Instype.U, target, params=params)
            continue

        # Parameterized two-qubit
        if gate.startswith("cu1") or gate.startswith("cp"):  # accept CP as CU1 alias if desired
            params, rest2 = _grab_params(rest if gate in ("cu1", "cp") else gate_full[3:] + rest)
            if len(params) != 1:
                raise ValueError(f"CU1/CP expects 1 parameter, got {params}")
            # rest2 shape: "q0, q1"
            toks = [t.strip() for t in rest2.split(",")]
            if len(toks) != 2:
                raise ValueError(f"CU1 expects two qubit args, got '{rest2}'")
            add_twoq_gate(Instype.CP, toks[0], toks[1], params=params)
            continue

        # No-parameter one-qubit?
        if gate in oneq_no_param:
            # rest should be like "q0"
            add_oneq_gate(oneq_no_param[gate], rest)
            continue

        # Two-qubit no-param?
        if gate in twoq_no_param:
            toks = [t.strip() for t in rest.split(",")]
            if len(toks) != 2:
                raise ValueError(f"{gate.upper()} expects two qubit args, got '{rest}'")
            add_twoq_gate(twoq_no_param[gate], toks[0], toks[1])
            continue

        # Three-qubit no-param?
        if gate in threeq_no_param:
            toks = [t.strip() for t in rest.split(",")]
            if len(toks) != 3:
                raise ValueError(f"{gate.upper()} expects three qubit args, got '{rest}'")
            add_threeq_gate(threeq_no_param[gate], toks[0], toks[1], toks[2])
            continue

        raise ValueError(f"Unsupported or malformed instruction line: '{ln}'")


    return inst_list, data_n, syn_n, measure_n
    


benchmark_suit={
    0:"cat_state_prep_n4",
    1:"cat_state_verification_n4",
    2:"repetition_code_distance3_n3",
    3:"shor_parity_measurement_n4",
    4:"shor_stabilizer_XZZX_n3",
    5:"shor_stabilizer_ZZZZ_n4",
    6:"syndrome_extraction_surface_n3"
}


if __name__ == "__main__":

   file_path = "C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\cat_state_verification_n4"




   inst_list, data_n, syn_n, measure_n = parse_program_from_file(file_path)


   qiskit_circuit = construct_qiskit_circuit(data_n, syn_n, measure_n, inst_list)


   shots = 2000

   sim = AerSimulator()
   tqc = transpile(qiskit_circuit, sim)

   # Run with 1000 shots
   result = sim.run(tqc, shots=shots).result()
   counts = result.get_counts(tqc)

   with open("C://Users//yezhu//Documents//HALO//benchmark//result2000shots//cat_state_verification_n4_counts.pkl", "wb") as f:
     pickle.dump(counts, f)


   #print(simulate_result)
   #parse_qasm_instruction(0,qasm_code)
