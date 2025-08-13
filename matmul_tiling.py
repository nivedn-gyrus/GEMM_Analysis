import numpy as np

PER_CORE_INSTANCES = {}
L2_SHARED_GLOBAL_INSTANCE = None

# Number of instance = number of cores
class L1_chache:
    def __init__(self, size):
        self.size = size
        self.memory_used = 0
        self.data ={}
        self.mac = 4096

    # Inserting data
    def copy_data_to_L1_from_L2(self, name, data):
        size_bytes = data.size
        if self.memory_used + size_bytes > self.size:
            raise MemoryError("L1 memory is overflowing")
        self.data[name] = data.copy()
        self.memory_used += size_bytes

        print(f" Loaded '{name}' tile of shape {data.shape} in L1 ")
        print(f" memory avialable is {self.size - self.memory_used }")
        


    # Deleting data 
    def delete_data(self, name):
        if name in self.data:
            size_bytes = self.data[name].size
            self.memory_used -= size_bytes
            del self.data[name]

            print(f"[L1] removed '{name}'")
            print(f" memory avialable is {self.size - self.memory_used }")
        


    def check_mac(self, A_name, B_name):
        
        mac_calculated = self.data[A_name].shape[0] * self.data[A_name].shape[1] * self.data[B_name].shape[1]
        assert mac_calculated < self.mac
        print("Mac calculated in L1 is :", mac_calculated)


    def calculate_mat_mul(self, A_name , B_name):
        result = self.data[A_name] @ self.data[B_name]

        return result


# only one instance is created
class L2_shared:
    def __init__(self, size):
        self.size = size
        self.memory_used = 0
        self.data ={}

    def copy_data_to_L2(self, name, data):
        size_bytes = data.size
        if self.memory_used + size_bytes > self.size:
            raise MemoryError("L2 memory is overflowing")
        self.data[name] = data.copy()
        self.memory_used += size_bytes

        
        print(f" Loaded '{name}' data of shape {data.shape} in L2 ")
        print(f" memory avialable is {self.size - self.memory_used }")

        

    def delete_data(self, name):
        if name in self.data:
            size_bytes = self.data[name].size
            self.memory_used -= size_bytes
            del self.data[name]

            print(f"[L2] removed '{name}'")
            print(f" memory avialable is {self.size - self.memory_used }")

        

def create_core_instances(num_core, core_capacity):
    for i in range(num_core):
        PER_CORE_INSTANCES[f"core_{i}"] = L1_chache(core_capacity)

def create_tile_map(A_matrix, B_matrix):

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

                print("#" * 20)    
                print(f"i: {i}")
                print(f"i_end: {i_end}")
                print(f"k : {k}")
                print(f"k_end : {k_end}")
                print(f"j : {j}")
                print(f"j_end : {j_end}")
                print("#" * 20)

                A_tile = A[i:i_end, k:k_end]
                B_tile = B[k:k_end, j:j_end]

                A_tile_name = f"A_tile_{i // tile_size}_{k // tile_size}"
                B_tile_name = f"B_tile_{k // tile_size}_{j // tile_size}"

                tile_map_matrix_A[A_tile_name] = A_tile
                tile_map_matrix_A[B_tile_name] = B_tile

    return tile_map_matrix_A, tile_map_matrix_B

def get_tile_processing_order(tile_map_matrix_A, tile_map_matrix_B):
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

    for i_tile, k_tile, a_name in A_tiles_info:
        processing_order.append(("LOAD", a_name, "A_tile"))
        for b_k_tile, j_tile, b_name in B_tiles_info:
            if b_k_tile == k_tile:
                processing_order.append(("LOAD", b_name, "B_tile"))
                processing_order.append(("MATMUL", a_name, b_name))
                processing_order.append(("EVICT", b_name))
        processing_order.append(("EVICT", a_name))

    return processing_order

def tiled_matrix_multiplication(A_name, B_name, tile_size):

    A = L2_SHARED_GLOBAL_INSTANCE.data[A_name]
    B = L2_SHARED_GLOBAL_INSTANCE.data[B_name]

    assert A.shape[1] == B.shape[0]
    

    # m, n = A.shape
    # n, p = B.shape
    # C = np.zeros((m, p))

    C_tiles = {}

    tile_map_matrix_A, tile_map_matrix_B = create_tile_map(A, B)
    order = get_tile_processing_order(tile_map_matrix_A, tile_map_matrix_A)

    core_id = 0
    core_name = f"core_{core_id}"
    core_instance = PER_CORE_INSTANCES[core_name]

    for instruction in order:

        tile_name = instruction[1]


        if instruction[0] == "LOAD":
            
            if tile_name in tile_map_matrix_A:
                core_instance.copy_data_to_L1_from_L2(tile_name, tile_map_matrix_A[tile_name] )
            elif tile_name in tile_map_matrix_B:
                core_instance.copy_data_to_L1_from_L2(tile_name, tile_map_matrix_B[tile_name] )

        elif instruction[0] == "EVICT":

            core_instance.delete_data(tile_name)

        elif instruction[0] == "MATMUL":       
        
            a_tile_name = instruction[1]
            b_tile_name = instruction[2]

            core_instance.check_mac( a_tile_name , b_tile_name)
            partial_result = core_instance.calculate_mat_mul(a_tile_name , b_tile_name)

            i_tile = int(a_tile_name.split("_")[2])
            j_tile = int(b_tile_name.split("_")[3])
            c_tile_name = f"C_tile_{i_tile}_{j_tile}"

            if c_tile_name not in C_tiles:
                C_tiles[c_tile_name] = np.zeros_like(partial_result)  # Assuming its being created inside L1

            C_tiles[c_tile_name] += partial_result

        # print(ref_tile_res)
        # print("\n")
        # print(result)

        # mac = (i_end - i ) * (k_end - k) * (j_end - j)
        
        # print("MAC:", mac)
        # assert mac < 4096

        # # Assumption for simple calculation that tiles have the same H and W
        # L1_memory_being_used = tile_size * tile_size + tile_size * tile_size + tile_size * tile_size  # tile_A1 * tile_A2 + tile_B1 * tile_B2 + tile_C1 * tile_C2 
        # print("Used L1 memory : ", L1_memory_being_used)
        # assert L1_memory_being_used < 16384 # 16KB = 16 * 1024 Bytes


        # L2
        # L3 
        # L2 to L1 load .
        # Reusing L1 .
        # function to static, _ 

        # C[i:i_end, j:j_end] += A_tile @ B_tile

        # C[i:i_end, j:j_end] += result

        # TODO : Think of reuse part
        # core_instance.delete_data(A_tile_name)
        # core_instance.delete_data(B_tile_name)


    return C_tiles

def check_memory_of_original_matrix_combined_to_copy_to_L2(A, B):
    A_size_byte = A.size
    B_size_byte = B.size

    memory_required = A_size_byte + B_size_byte
    if memory_required < L2_SHARED_GLOBAL_INSTANCE.size :
        return True
    return False

def reconstruct_C_matrix(C_tiles, m, p, tile_size):

    C = np.zeros((m, p))

    for tile_name, tile_data in C_tiles.items():
   
        _, _, i_str, j_str = tile_name.split("_")
        i_tile = int(i_str)
        j_tile = int(j_str)

        i_start = i_tile * tile_size
        j_start = j_tile * tile_size

 
        C[i_start:i_start + tile_data.shape[0],j_start:j_start + tile_data.shape[1]] = tile_data

    return C

if __name__ == "__main__":
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    B = np.array([[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]])


    m, n = A.shape
    n, p = B.shape

    ref_result =  A @ B


    # creating L2 instance
    L2_capacity = 16 * 1024 * 1024
    L2_SHARED_GLOBAL_INSTANCE = L2_shared(L2_capacity)

    # TODO: later check for L1 or L2 tiling is being done
    # L2_memory_check_flag = check_memory_of_original_matrix_combined_to_copy_to_L2(A, B)

    # if not L2_memory_check_flag:
    #     # Add a way to 
    #     raise MemoryError("L2 Memory Overflow")

    L2_SHARED_GLOBAL_INSTANCE.copy_data_to_L2("A", A )
    L2_SHARED_GLOBAL_INSTANCE.copy_data_to_L2("B", B )

    matrix_A_name = "A"
    matrix_B_name = "B"
    

    tile_size = 2

    # creating L1 instance for each core
    num_cores = 24
    core_capacity = 16*1024
    create_core_instances(num_cores, core_capacity)

    C_tiles = tiled_matrix_multiplication(matrix_A_name, matrix_B_name, tile_size)

    C = reconstruct_C_matrix(C_tiles, m, p, tile_size)

    print(ref_result)
    print("\n")
    print(C)