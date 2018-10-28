
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "stdint.h"
#include "BinaryHelper.h"
#include "DESEncoder.h"
#include "iterator"


int main()
{
	uint64_t message[] = { 123 };
	uint64_t key = 1383827165325090801;
	int blocks = 1;
	uint64_t *message_blocks = encode(message, key, blocks);

	for (int i = 0; i < blocks; i++) {
		printf("%llu \n", message_blocks[i]);
	}
}

