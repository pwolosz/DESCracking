#pragma once
#include "stdint.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Constants.h"
#include <cmath>
#include <stdio.h>

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

__host__ __device__ void get_indexes(uint64_t block, int block_number, int &left_index, int &right_index) {
	int bit_number = 6 * (8 - block_number);
	uint64_t mask = ((uint64_t)63) << bit_number;
	uint64_t selected_block = (block & mask) >> bit_number;

	right_index = ((selected_block & (1U << 5)) >> 4) + (selected_block & 1U);
	left_index = (selected_block >> 1) & 15;
}

__host__ __device__ uint64_t code_with_s(uint64_t block) {
	uint64_t coded_block = 0;
	int left_index, right_index;
	for (int i = 1; i <= 8; i++) {
		get_indexes(block, i, left_index, right_index);
		coded_block = (coded_block << 4) + S[i - 1][right_index][left_index];
	}

	return coded_block;
}

__host__ __device__ uint64_t pow2(int exp)
{
	uint64_t result = 1ULL;
	for (int i = 0; i < exp; i++) {
		result = result << 1;
	}
	return result;
}

__device__ __host__ uint64_t string_to_int(char* string, int size) {
	uint64_t message = 0;
	for (int i = 0; i < size; i++) {
		message = (message << BITS_PER_CHAR) + (int)string[i];
	}

	return message;
}

__device__ __host__ char* int_to_string(uint64_t message) {
	char *str = new char[MAX_MESSAGE_LENGTH];
	int bits_value = pow2(BITS_PER_CHAR), index = 0;

	while (message != 0) {
		int val = message % bits_value;
		str[index] = val;
		message = message >> BITS_PER_CHAR;
		index++;
	}
	int i = 0;
	char *str_val = new char[index + 1];

	while (i < index) {
		str_val[index - i - 1] = str[i];
		i++;
	}
	str_val[index] = '\0';
	delete(str);

	return str_val;
}

__device__ __host__ int power(int number, int power) {
#ifdef __CUDA_ARCH__
	return powf(number, power);
#else
	return pow(number, power);
#endif
}

__device__ __host__ uint64_t* get_messages(int length) {
	int *indexes = new int[length];
	int words_count = power(ALPHABET_SIZE, length);

	for (int i = 0; i < length; i++) {
		indexes[i] = 0;
	}

	uint64_t *words = new uint64_t[words_count];
	char* word = new char[length];
	for (int i = 0; i < words_count; i++) {
		int k = 0;
		indexes[k]++;
		while (indexes[k] == ALPHABET_SIZE) {
			indexes[k] = 0;
			k++;
			indexes[k]++;
		}

		for (int j = 0; j < length; j++) {
			word[j] = ALPHABET[indexes[j]];
		}

		words[i] = string_to_int(word, length);
	}
	
	return words;
}

__device__ __host__ uint64_t encode_message(char *str, int size) {
	return string_to_int(str, size);
}
