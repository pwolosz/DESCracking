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

__host__ __device__ uint64_t* encode(uint64_t* message_blocks, uint64_t key, int blocks)
{
	return generate_keys(key);
}


