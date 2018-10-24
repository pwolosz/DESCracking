#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdint.h"

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT
#endif

CONSTANT  int PC1_SIZE = 56;

CONSTANT  int PC1[] = {	57,		49,		41,		33,		25,		17,		9,
						1,		58,		50,		42,		34,		26,		18,
						10,		2,		59,		51,		43,		35,		27,
						19,		11,		3,		60,		52,		44,		36,
						63,		55,		47,		39,		31,		23,		15,
						7,		62,		54,		46,		38,		30,		22,
						14,		6,		61,		53,		45,		37,		29,
						21,		13,		5,		28,		20,		12,		4};

