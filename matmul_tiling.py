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
        pass


    # Deleting data 
    def delete_data(self, name):
        if name in self.data:
            size_bytes = self.data[name].size
            self.memory_used -= size_bytes
            del self.data[name]

            print(f"[L1] removed '{name}'")
            print(f" memory avialable is {self.size - self.memory_used }")
        pass


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

        
        print(f" Loaded '{name}' data of shape {data.shape} in L1 ")
        print(f" memory avialable is {self.size - self.memory_used }")

        pass

    def delete_data(self, name):
        if name in self.data:
            size_bytes = self.data[name].size
            self.memory_used -= size_bytes
            del self.data[name]

            print(f"[L2] removed '{name}'")
            print(f" memory avialable is {self.size - self.memory_used }")

        pass

def create_core_instances(num_core, core_capacity):
    for i in range(num_core):
        PER_CORE_INSTANCES[f"core_{i}"] = L1_chache(core_capacity)


def tiled_matrix_multiplication(A_name, B_name, tile_size):

    A = L2_SHARED_GLOBAL_INSTANCE.data[A_name]
    B = L2_SHARED_GLOBAL_INSTANCE.data[B_name]

    assert A.shape[1] == B.shape[0]
    

    m, n = A.shape
    n, p = B.shape
    C = np.zeros((m, p))

    core_id = 0

    for i in range(0, m, tile_size):
        for j in range(0, p, tile_size):
            for k in range(0, n, tile_size):
        
                i_end = min(i + tile_size, m)
                j_end = min(j + tile_size, p)
                k_end = min(k + tile_size, n)

                A_tile = A[i:i_end, k:k_end]
                B_tile = B[k:k_end, j:j_end]

                ref_tile_res = A_tile @ B_tile
                
                core_name = f"core_{core_id}"
                core_instance = PER_CORE_INSTANCES[core_name]
                
                A_tile_name = f"A_tile_{i}_{k}"
                B_tile_name = f"B_tile_{j}_{k}"

                core_instance.copy_data_to_L1_from_L2(A_tile_name, A_tile )
                core_instance.copy_data_to_L1_from_L2(B_tile_name, B_tile )
                
                core_instance.check_mac( A_tile_name , B_tile_name)
                result = core_instance.calculate_mat_mul(A_tile_name , B_tile_name)

                print(ref_tile_res)
                print("\n")
                print(result)

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

                C[i:i_end, j:j_end] += result

                # TODO : Think of reuse part
                core_instance.delete_data(A_tile_name)
                core_instance.delete_data(B_tile_name)


    return C

def check_memory_of_original_matrix_combined_to_copy_to_L2(A, B):
    A_size_byte = A.size
    B_size_byte = B.size

    memory_required = A_size_byte + B_size_byte
    if memory_required < L2_SHARED_GLOBAL_INSTANCE.size :
        return True
    return False

if __name__ == "__main__":
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    B = np.array([[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]])


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

    C_tiled = tiled_matrix_multiplication(matrix_A_name, matrix_B_name, tile_size)

    print(ref_result)
    print("\n")
    print(C_tiled)