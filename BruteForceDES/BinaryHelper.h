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
	*right_block = (block & (UINT64_MAX >> (64 - block_size / 2)));
}

__host__ __device__ uint64_t merge_blocks(uint64_t left_block, uint64_t right_block, int block_size) {
	return (left_block << block_size) + right_block;
}

__host__ __device__ uint64_t pc(uint64_t block, int block_size, int pc_table[], int pc_size) {
	uint64_t out_block = 0;
	for (int i = 0; i < pc_size; i++) {
		uint64_t mask = ((1LLU << (block_size - pc_table[i])) & block) >> (block_size - pc_table[i]);
		out_block = (out_block << 1) | mask;
	}

	return out_block;
}