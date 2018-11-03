
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "stdint.h"
#include "BinaryHelper.h"
#include "DESEncoder.h"
#include "iterator"
#include "Helpers.h"

__global__ void decrypt(uint64_t* coded_message, uint64_t* message, int key_size, int dev_block_size, int blocks_x, int message_blocks) {
	int index = threadIdx.x;
	int block = blockIdx.x;
	int dim_pow = 0;
	while (dev_block_size > 1) {
		dev_block_size = dev_block_size / 2;
		dim_pow++;
	}
	while (blocks_x > 1) {
		blocks_x = blocks_x / 2;
		dim_pow++;
	}
	uint64_t *encoded;

	int p = key_size - dim_pow;
	uint64_t val = pow2(p);
	if (index != 0 || block != 0) return;
	printf("%llu-%llu\n", index * val, (index + 1) * val - 1);
	for (int j = 1; j <= MAX_MESSAGE_LENGTH; j++) {
		printf("j=%d\n", j);
		uint64_t *messages = get_messages(j);
		printf("Got messages\n");
		for (uint64_t i = index * val; i <= (index+1) * val - 1; i++) {
			for (int k = 0; k < power(ALPHABET_SIZE, j); k++) {
				printf("i=%llu k=%d\n", i, k);
				encoded = encode(new uint64_t[1]{ messages[k] }, i, 1);
				if (encoded[0] == coded_message[0]) {
					message[0] = messages[k];
					return;
				}
			}
		}
	}
}

int main()
{
	uint64_t *messages = get_messages(1);
	uint64_t *encoded = encode(new uint64_t[1]{ messages[0] }, 222, 1);
	int size = 0;
	char *m = new char[MAX_MESSAGE_LENGTH + 1];
	char *message = new char[MAX_MESSAGE_LENGTH + 1];
	uint64_t key = 1;
	printf("Using 32b key and MAX_MESSAGE_LENGTH=%d\nMessage: ", MAX_MESSAGE_LENGTH);
	scanf("%s", m);

	message = get_message(m, &size);

	printf("Coding message: %s\n", message);
	uint64_t *message_blocks = encode_message(message, size);
	uint64_t *encoded_message = encode(message_blocks, key, 1);
	
	printf("%llu\n", encoded_message[0]);

	int key_size = 32;
	int block_size = 512;
	int blocks = 1;
	int used_device_blocks = 4096;
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}

	uint64_t *dev_encoded_message = 0;
	cudaStatus = cudaMalloc((void**)&dev_encoded_message, blocks * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	uint64_t *dev_decoded_message=0, *decoded_message = new uint64_t[blocks];
	cudaStatus = cudaMalloc((void**)&dev_decoded_message, blocks * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(dev_encoded_message, encoded_message, blocks * sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}

	decrypt<<<1, 1>>>(dev_encoded_message, dev_decoded_message, key_size, block_size, 4096, blocks);
	
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

	cudaStatus = cudaMemcpy(decoded_message, dev_decoded_message, blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}

	printf("------------\n");
	printf("Decoded message numerical: %llu\n", decoded_message[0]);
	printf("Decoded message string: %s\n", int_to_string(decoded_message[0]));
}

