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

__host__ __device__ unsigned short* get_indexes(uint64_t block, int block_number) {
	int bit_number = 6 * (8 - block_number);
	uint64_t mask = ((uint64_t)63) << bit_number;
	uint64_t selected_block = (block & mask) >> bit_number;
	unsigned short *indexes = new unsigned short[2];

	indexes[1] = ((selected_block & (1U << 5)) >> 4) + (selected_block & 1U);
	indexes[0] = (selected_block >> 1) & 15;

	return indexes;
}

__host__ __device__ uint64_t code_with_s(uint64_t block) {
	uint64_t coded_block = 0;
	unsigned short *indexes;
	for (int i = 1; i <= 8; i++) {
		indexes = get_indexes(block, i);
		coded_block = (coded_block << 4) + S[i - 1][indexes[1]][indexes[0]];
	}

	return coded_block;
}