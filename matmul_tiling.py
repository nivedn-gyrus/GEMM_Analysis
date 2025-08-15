import math
import numpy as np


# -----------------------------------
# Instruction Queue
# -----------------------------------
class InstrQueue:
    def __init__(self, instr_type, queue_depth=1):
        self.instr_type = instr_type
        self.queue_depth = queue_depth  # was hardcoded 16 why?
        self.instr_queue = []
        self.instr_empty = True
        self.instr_full = False

    def instr_push(self, instr):
        if len(self.instr_queue) >= self.queue_depth:
            raise OverflowError(f"{self.instr_type} queue is full")
        self.instr_queue.append(instr)
        self._update_flags()

    def instr_pop(self):
        if not self.instr_queue:
            raise IndexError(f"{self.instr_type} queue is empty")
        instr = self.instr_queue.pop(0)  # FIFO
        self._update_flags()
        return instr

    def _update_flags(self):
        self.instr_empty = len(self.instr_queue) == 0
        self.instr_full = len(self.instr_queue) >= self.queue_depth

    def __len__(self):
        return len(self.instr_queue)
    


# -----------------------------------
# Memory Operation Unit
# -----------------------------------
class MemOper:
    def __init__(self, instr_type, latency, data_width, core_ref):
        self.instr_type = instr_type
        self.data_width = data_width
        self.latency = latency
        self.core_ref = core_ref
        # At this moment it is only one depth
        self.instr_queue = InstrQueue(instr_type, 1)
        self.wait_cycles = 0
        self.active_instr = None

    def set_waitcycles(self, num_bytes):
        # ceil for integer cycles
        transfer_cycles = math.ceil(num_bytes / self.data_width)
        self.wait_cycles = self.latency + transfer_cycles

    def MemProcess(self):
        # If an instruction is active, process it
        if self.active_instr:
            if self.wait_cycles > 0:
                self.wait_cycles -= 1
            else:
                # Complete the copy

                tile = self.active_instr['tile']
                size = self.active_instr['size']
                dim1, dim2 = self.active_instr['dims']

                if self.instr_type == "L1_to_L2":
                    self.core_ref.L2_cache[tile] = self.core_ref.L1_cache[tile].copy()
                
                elif self.instr_type == "L2_to_L1":
                    self.core_ref.L1_cache[tile] = self.core_ref.L2_cache[tile].copy()
                
                elif self.instr_type == "L3_to_L2":
                    if tile.startswith("A_tile"):
                        parts = tile.split("_")
                        i_idx = int(parts[2])
                        k_idx = int(parts[3])
                        tile_matrix = self.core_ref.L3_cache["A"][
                            i_idx * dim1 : i_idx * dim1 + dim1,
                            k_idx * dim2 : k_idx * dim2 + dim2
                        ]
                        self.core_ref.L2_cache[tile] = tile_matrix.copy()

                    elif tile.startswith("B_tile"):
                        parts = tile.split("_")
                        k_idx = int(parts[2])
                        j_idx = int(parts[3])
                        tile_matrix = self.core_ref.L3_cache["B"][
                            k_idx * dim1 : k_idx * dim1 + dim1,
                            j_idx * dim2 : j_idx * dim2 + dim2
                        ]

                        self.core_ref.L2_cache[tile] = tile_matrix.copy()


                elif self.instr_type == "L2_to_L3":
                    self.core_ref.L3_cache[tile] = self.core_ref.L2_cache[tile].copy()
                    

                print(f"[{self.instr_type}] Completed: {self.active_instr}")
                self.active_instr = None

        # If no active instruction and queue not empty, start next
        if not self.active_instr and not self.instr_queue.instr_empty:
            self.active_instr = self.instr_queue.instr_pop()
            num_bytes = self.active_instr.get("size", 0)
            self.set_waitcycles(num_bytes)
            print(f"[{self.instr_type}] Started: {self.active_instr}")

        return 0   
    
# -----------------------------------
# Core Processing Unit
# -----------------------------------
class CoreUnit:
    def __init__(self, l1_size, l2_size, mac_cnt):
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.mac_cnt = mac_cnt

        # cache 
        self.L1_cache = {}
        self.L2_cache = {}
        self.L3_cache = {}

        self.L1_to_L2_Process = MemOper("L1_to_L2", 10, 64, self)
        self.L2_to_L1_Process = MemOper("L2_to_L1", 10, 64, self)
        self.L3_to_L2_Process = MemOper("L3_to_L2", 200, 32, self)
        self.L2_to_L3_Process = MemOper("L2_to_L3", 200, 32, self)

        self.MAC_InstrQueue = InstrQueue("MAC", 1)

        self.mac_wait_cycle = 0
        self.active_mac_instr = None

    def mac_operation(self):
        if self.active_mac_instr:
            if self.mac_wait_cycle > 0:
                self.mac_wait_cycle -= 1
            else:
                print(f"[MAC] Completed: {self.active_mac_instr}")
                self.active_mac_instr = None

        if not self.active_mac_instr and not self.MAC_InstrQueue.instr_empty:
            self.active_mac_instr = self.MAC_InstrQueue.instr_pop()
            m, n, k = self.active_mac_instr.get("dims", (1, 1, 1))
            mac_ops = m * n * k
            self.mac_wait_cycle = math.ceil(mac_ops / self.mac_cnt)
            print(f"[MAC] Started: {self.active_mac_instr}")

            a_tile = self.L1_cache[self.active_mac_instr.get("tile_A", None)]
            b_tile = self.L1_cache[self.active_mac_instr.get("tile_B", None)]
            matmul_result = a_tile @ b_tile
            print(f"[MATMUL] Result: {matmul_result}")

    def main_loop(self):
        self.L2_to_L1_Process.MemProcess()
        self.L1_to_L2_Process.MemProcess()
        self.L3_to_L2_Process.MemProcess()
        self.L2_to_L3_Process.MemProcess()
        self.mac_operation()


def create_tile_map(A_matrix, B_matrix, tile_size):
    """

    Maps the tiles from A and B by name to their respective dictionary 
    for better access while doing tile multipllication

    """

    assert A_matrix.shape[1] == B_matrix.shape[0]

    m, n = A_matrix.shape
    n, p = B_matrix.shape

    tile_map_matrix_A = {}
    tile_map_matrix_B = {}

    for i in range(0, m, tile_size):
        for j in range(0, p, tile_size):
            for k in range(0, n, tile_size):
        
                i_end = min(i + tile_size, m)
                j_end = min(j + tile_size, p)
                k_end = min(k + tile_size, n)

                A_tile = A[i:i_end, k:k_end]
                B_tile = B[k:k_end, j:j_end]

                A_tile_name = f"A_tile_{i // tile_size}_{k // tile_size}"
                B_tile_name = f"B_tile_{k // tile_size}_{j // tile_size}"

                tile_map_matrix_A[A_tile_name] = A_tile
                tile_map_matrix_B[B_tile_name] = B_tile

    return tile_map_matrix_A, tile_map_matrix_B

def get_tile_size(A, B):
    """
    This Function is to determine the tile size
    TODO: Its only a base more work might be needed 
    """

    L1_capacity = 16 * 1024 
    L1_mac = 4096

    m, n = A.shape
    n, p = B.shape

    total_memory = m * n + n * p + m * n

    tile_sizes_to_choose_from = [2, 4, 8, 16, 64, 128 , 256]
    best_tile_size = tile_sizes_to_choose_from[0]

    if   total_memory > L1_capacity:

        for t_size in tile_sizes_to_choose_from :

            mem_tile = 3 * (tile_size * tile_size)

            if mem_tile <= L1_capacity:
                best_tile_size = t_size

            else:
                break
    
    return best_tile_size

def get_tile_processing_order(tile_map_matrix_A, tile_map_matrix_B):
    """
    Creates a basic instuction order to do Matrix Multiplication
    
    TODO: Find better Optimizing order 
    """
    A_tiles_info = []
    for name in tile_map_matrix_A.keys():
        if name.startswith("A_tile"):
            parts = name.split("_")
            i_tile = int(parts[2])
            k_tile = int(parts[3])
            A_tiles_info.append((i_tile, k_tile, name))

    B_tiles_info = []
    for name in tile_map_matrix_B.keys():
        if name.startswith("B_tile"):
            parts = name.split("_")
            k_tile = int(parts[2])
            j_tile = int(parts[3])
            B_tiles_info.append((k_tile, j_tile, name))

    processing_order = []
    A_tiles_info.sort(key=lambda x: (x[0], x[1]))
    B_tiles_info.sort(key=lambda x: (x[0], x[1]))


    # Adding all the tile to L2 for simplicity
    for i_tile, k_tile, a_name in A_tiles_info:
        processing_order.append(("LOAD_TO_L2", a_name, "A_tile"))

    for b_k_tile, j_tile, b_name in B_tiles_info:
        processing_order.append(("LOAD_TO_L2", b_name, "B_tile"))

    # MATMUL order
    for i_tile, k_tile, a_name in A_tiles_info:
        processing_order.append(("LOAD_TO_L1", a_name, "A_tile"))
        for b_k_tile, j_tile, b_name in B_tiles_info:
            if b_k_tile == k_tile:
                processing_order.append(("LOAD_TO_L1", b_name, "B_tile"))
                processing_order.append(("MATMUL", a_name, b_name))
                processing_order.append(("EVICT_TO_L2", b_name))
        processing_order.append(("EVICT_TO_L2", a_name))

    return processing_order
    
def add_instr_to_queue_for_processing(instr_order, core_unit):

    for instruction in instr_order:

        tile_name = instruction[1]

        if instruction[0] == "LOAD_TO_L2":
            core_unit.L3_to_L2_Process.instr_queue.instr_push({"op": "LOAD", "tile": tile_name , "size": tile_size * tile_size, "dims":(tile_size , tile_size) })


        elif instruction[0] == "LOAD_TO_L1":
            core_unit.L2_to_L1_Process.instr_queue.instr_push({"op": "LOAD", "tile": tile_name , "size": tile_size * tile_size, "dims":(tile_size , tile_size)})

        elif instruction[0] == "EVICT_TO_L2":
            core_unit.L1_to_L2_Process.instr_queue.instr_push({"op": "EVICT", "tile": tile_name, "size": tile_size * tile_size, "dims":(tile_size , tile_size)})

        elif instruction[0] == "MATMUL":
            a_tile_name = instruction[1]
            b_tile_name = instruction[2]

            core_unit.MAC_InstrQueue.instr_push({"op": "MATMUL", "tile_A": a_tile_name, "tile_B": b_tile_name , "dims": (tile_size, tile_size, tile_size)})

        
if __name__ == "__main__":


    # The input matrix  for GEMM
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    B = np.array([[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]])

    # Shapes of each Matrices
    m, n = A.shape
    n, p = B.shape

    # Calculating the actual Matrix to compare the result
    ref_result =  A @ B

    # -----------------------------------
    # Hardware setup  --> Only one core
    # -----------------------------------
    l1_size = 16 * 1024
    l2_size = 16 * 1024 * 1024
    mac_cnt = 4096

    # creating the Core instance
    core_unit = CoreUnit(l1_size, l2_size, mac_cnt)

    # We have the matrix in L3 --> That is the base assumption
    core_unit.L3_cache['A'] = A
    core_unit.L3_cache['B'] = B
    print(f"Copied matrix A : {A}")
    print(f"Copied matrix B : {B}")


    # Tile Size
    tile_size = 2  # get_tile_size(A, B) 
    tile_map_matrix_A, tile_map_matrix_B = create_tile_map(A, B, tile_size)
    instr_order = get_tile_processing_order(tile_map_matrix_A, tile_map_matrix_B, core_unit)

    # # Example instructions
    # core_unit.L2_to_L1_Process.instr_queue.instr_push({"op": "LOAD", "tile": "A1", "size": 1024})
    # core_unit.L2_to_L1_Process.instr_queue.instr_push({"op": "LOAD", "tile": "B1", "size": 1024})
    # core_unit.MAC_InstrQueue.instr_push({"op": "MATMUL", "dims": (64, 64, 64)})
    # core_unit.L1_to_L2_Process.instr_queue.instr_push({"op": "EVICT", "tile": "C1", "size": 2048})

    add_instr_to_queue_for_processing(instr_order, tile_map_matrix_A, tile_map_matrix_B, core_unit, tile_size)


    # -----------------------------------
    # Main Simulation Loop
    # -----------------------------------
    CycleCnt = 0
    MAX_CYCLES = 200

    while CycleCnt < MAX_CYCLES:
        CycleCnt += 1
        print(f"\n--- Cycle {CycleCnt} ---")
        core_unit.main_loop()

        # Break early if no work left
        all_empty = (
            core_unit.L1_to_L2_Process.instr_queue.instr_empty
            and core_unit.L2_to_L1_Process.instr_queue.instr_empty
            and core_unit.L3_to_L2_Process.instr_queue.instr_empty
            and core_unit.L2_to_L3_Process.instr_queue.instr_empty
            and core_unit.MAC_InstrQueue.instr_empty
            and not core_unit.active_mac_instr
            and not core_unit.L1_to_L2_Process.active_instr
            and not core_unit.L2_to_L1_Process.active_instr
            and not core_unit.L3_to_L2_Process.active_instr
            and not core_unit.L2_to_L3_Process.active_instr
        )
        if all_empty:
            print("\n[Simulation completed]")
            break  