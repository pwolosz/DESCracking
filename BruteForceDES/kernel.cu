#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "stdint.h"
#include "BinaryHelper.h"
#include "DESEncoder.h"
#include "iterator"
#include "Helpers.h"
#include "CPUDecoder.h"
#include "GPUDecoder.h"

int main(char **args)
{
	int size = 0;
	char *m = new char[MAX_MESSAGE_LENGTH + 1];
	char *message = new char[MAX_MESSAGE_LENGTH + 1];
	int device_opt;
	uint64_t key = 820224;
	
	printf("Select decoding option: 1 - CPU / 2 - GPU (default): ");

	scanf("%d", &device_opt);

	printf("Using 32b key and MAX_MESSAGE_LENGTH=%d\nMessage: ", MAX_MESSAGE_LENGTH);
	scanf("%s", m);

	message = get_message(m, &size);

	printf("Coding message: %s\n", message);
	printf("Key: %llu\n", key);
	uint64_t message_block = encode_message(message, size);
	uint64_t encoded_message = encode(message_block, key);
	int key_size = 32;
	uint64_t decoded_message;
	clock_t begin = clock();
	
	if (device_opt != 1) {
		printf("Using GPU \n");
		decoded_message = decrypt_gpu(encoded_message, key_size);
	}
	else {
		printf("Using CPU \n");
		decoded_message = decrypt_cpu(encoded_message, key_size);
	}

	clock_t end = clock();

	printf("------------\n");
	printf("Decoded message numerical: %llu\n", decoded_message);
	printf("Decoded message string: %s\n", int_to_string(decoded_message));
	printf("%llu seconds ellapsed\n", uint64_t(end - begin) / CLOCKS_PER_SEC);

}

