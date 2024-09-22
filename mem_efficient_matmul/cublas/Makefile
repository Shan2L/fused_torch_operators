all: cublasSgemmBatched cublasSgemmStridedBatched

cublasSgemmBatched: cublasSgemmBatched.cu
	nvcc cublasSgemmBatched.cu -lcublas -g -O0 -o cublasSgemmBatched

cublasSgemmStridedBatched: cublasSgemmStridedBatched.cu
	nvcc cublasSgemmStridedBatched.cu -lcublas -g -O0 -o cublasSgemmStridedBatched


clean:
	rm -rf cublasSgemmBatched cublasSgemmStridedBatched