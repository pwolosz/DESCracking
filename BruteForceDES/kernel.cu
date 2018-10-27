
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "stdint.h"
#include "BinaryHelper.h"
#include "DESEncoder.h"
#include "iterator"


int main()
{
	uint64_t message[] = { 81985529216486895 };
	uint64_t key = 1383827165325090801;
	int blocks = 1;
	uint64_t *keys = encode(message, key, 1);
}

