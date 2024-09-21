all: cublasSgemmBatched

cublasSgemmBatched: cublasSgemmBatched.cu
	nvcc cublasSgemmBatched.cu -lcublas -o cublasSgemmBatched

.clean:
	rm -rf cublasSgemmBatched