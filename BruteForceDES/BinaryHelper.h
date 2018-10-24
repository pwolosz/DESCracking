#pragma once
#include "stdint.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Constants.h"

__host__ __device__ uint64_t shift_left(uint64_t block, int bits, int block_size) {
	return ((block << bits) | (block >> block_size - bits)) & (UINT64_MAX >> (64-block_size));
}

__host__ __device__ uint64_t shift_right(uint64_t block, int bits, int block_size) {
	return ((block >> bits) | (block << block_size - bits)) & (UINT64_MAX >> (64 - block_size));
}

__host__ __device__ void split_block(uint64_t block, uint64_t *left_block, uint64_t *right_block, int block_size) {
	*left_block = block >> (block_size / 2);
	*right_block = (block & (UINT64_MAX << (block_size / 2))) >> (block_size / 2);
}

__host__ __device__ uint64_t pc1(uint64_t block, int block_size) {
	uint64_t out_block = 0;
	for (int i = 0; i < PC1_SIZE; i++) {
		uint64_t mask = ((1LLU << (block_size - PC1[i])) & block) >> (block_size - PC1[i]);
		out_block = (out_block << 1) | mask;
	}

	return out_block;
}