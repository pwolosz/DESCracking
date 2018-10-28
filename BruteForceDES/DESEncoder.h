#pragma once
#include "stdint.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BinaryHelper.h"

__host__ __device__ uint64_t* generate_keys(uint64_t key) {
	uint64_t permuted_key = pc(key, 64, PC1, 56);
	uint64_t left_key, right_key;
	uint64_t *left_keys = new uint64_t[16];
	uint64_t *right_keys = new uint64_t[16];
	uint64_t *keys = new uint64_t[16];

	split_block(permuted_key, &left_key, &right_key, 56);
	left_keys[0] = shift_left(left_key, FIRST_KEY_ITERATION[0], 28);
	right_keys[0] = shift_left(right_key, FIRST_KEY_ITERATION[0], 28);

	for (int i = 1; i < 16; i++) {
		left_keys[i] = shift_left(left_keys[i-1], FIRST_KEY_ITERATION[i], 28);
		right_keys[i] = shift_left(right_keys[i-1], FIRST_KEY_ITERATION[i], 28);
	}

	for (int i = 0; i < 16; i++) {
		keys[i] = pc(merge_blocks(left_keys[i], right_keys[i], 28), 56, PC2, 48);
	}

	return keys;
}

__host__ __device__ uint64_t encode_block(uint64_t block, uint64_t key, int block_size) {
	block = pc(block, block_size, E, 48);
	block = key ^ block;
	uint64_t coded_block = code_with_s(block);
	coded_block = pc(coded_block, 32, P, 32);

	return coded_block;
}

__host__ __device__ uint64_t encode_block(uint64_t block, uint64_t *keys) {
	uint64_t left_block, right_block, next_left_block, next_right_block;

	split_block(block, &left_block, &right_block, 64);

	for (int i = 0; i < 16; i++) {
		next_left_block = right_block;
		next_right_block = left_block ^ encode_block(right_block, keys[i], 32);
		right_block = next_right_block;
		left_block = next_left_block;
	}

	block = pc(merge_blocks(next_right_block, next_left_block, 32), 64, IP_REV, 64);

	return block;
}

__host__ __device__ uint64_t* encode(uint64_t* message_blocks, uint64_t key, int block_count)
{
	uint64_t *blocks = new uint64_t[block_count];
	uint64_t *keys = generate_keys(key);

	for (int i = 0; i < block_count; i++) {
		blocks[i] = encode_block(pc(message_blocks[i], 64, IP, 64), keys);
	}

	return blocks;
}


