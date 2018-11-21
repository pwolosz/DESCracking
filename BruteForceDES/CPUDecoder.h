#pragma once
#include "stdint.h"
#include "Helpers.h"
#include "BinaryHelper.h"
#include "DESEncoder.h"
#include "GPUDecoder.h"

uint64_t decrypt_cpu(uint64_t message, int key_size) {
	uint64_t *messages = allocate_messages();

	for (int k = 0; k < pow2(key_size); k++) {
		for (int i = 0; i < get_messages_count(); i++) {
			int m_count = power(ALPHABET_SIZE, i + 1);

			if (message == encode(messages[i], k)) {
				return messages[i];
			}
		}
	}

	return 0;
}