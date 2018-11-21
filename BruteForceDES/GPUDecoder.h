#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "stdint.h"
#include "BinaryHelper.h"
#include "DESEncoder.h"
#include "iterator"
#include "Helpers.h"
#include <ctime>

__global__ void decrypt(uint64_t coded_message, uint64_t* message, int key_size, int dev_block_size, int blocks_x, int message_blocks, int *is_finised, uint64_t *dev_all_messages) {
	int index = threadIdx.x;
	int block = blockIdx.x;
	int dim_pow = 0;
	int dev_b = dev_block_size;
	int deb_i = blocks_x;
	while (dev_b > 1) {
		dev_b = dev_b / 2;
		dim_pow++;
	}
	while (deb_i > 1) {
		deb_i = deb_i / 2;
		dim_pow++;
	}
	uint64_t encoded;
	int p = key_size - dim_pow;
	uint64_t val = pow2(p);

	for (uint64_t i = index * val + block * val*dev_block_size; i <= (index + 1) * val - 1 + block * val*dev_block_size; i++) {
		for (int j = 0; j < get_messages_count(); j++) {
			if (*is_finised == 1) {
				return;
			}
			encoded = encode(dev_all_messages[j], i);
			if (encoded == coded_message) {
				message[0] = dev_all_messages[j];
				*is_finised = 1;
				return;
			}
		}
	}

	printf("%d - %d finished\n ", block, index);
}

uint64_t *allocate_messages() {
	uint64_t *messages = new uint64_t[get_messages_count()];
	int index = 0;

	for (int i = 0; i < MAX_MESSAGE_LENGTH; i++) {
		int m_count = power(ALPHABET_SIZE, i + 1);
		uint64_t *m = get_messages(i + 1);

		for (int j = 0; j < m_count; j++) {
			messages[index + j] = m[j];
		}
		index += m_count;
	}


	return messages;
}

uint64_t decrypt_gpu(uint64_t message, int key_size) {
	int block_size = 512;
	int used_device_blocks = 4096;

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}

	uint64_t *dev_all_messages;
	int *is_finised;
	uint64_t *dev_decoded_message = 0, *decoded_message = new uint64_t[1];
	cudaStatus = cudaMalloc((void**)&dev_decoded_message, sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMalloc((void**)&is_finised, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	uint64_t *messages = allocate_messages();
	cudaStatus = cudaMalloc((void**)&dev_all_messages, get_messages_count() * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(dev_all_messages, messages, get_messages_count() * sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}
	decrypt<<<used_device_blocks, block_size>>>(message, dev_decoded_message, key_size, block_size, used_device_blocks, 1, is_finised, dev_all_messages);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching decrypt!\n", cudaStatus);
		return 0;
	}

	cudaStatus = cudaMemcpy(decoded_message, dev_decoded_message, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}

	cudaFree(dev_decoded_message);
	cudaFree(is_finised);

	return decoded_message[0];
}