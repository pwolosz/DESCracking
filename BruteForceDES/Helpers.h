#pragma once

#include "Constants.h"
#include <stdio.h>

bool is_in_alphabet(char c) {
	for (int i = 0; i < ALPHABET_SIZE; i++) {
		if (ALPHABET[i] == c) {
			return true;
		}
	}

	return false;
}

void print_alphabet() {
	printf("Alphabet: ");
	for (int i = 0; i < ALPHABET_SIZE; i++) {
		printf("%c ", ALPHABET[i]);
	}
	printf("\n");
}

char* get_message(char* m, int *size) {
	char *message = new char[MAX_MESSAGE_LENGTH + 1];
	*size = 0;
	while (*size < MAX_MESSAGE_LENGTH && m[*size] != '\0') {
		if (!is_in_alphabet(m[*size])) {
			printf("Character not in the alphabet\n");
			return 0;
		}
		message[*size] = m[*size];
		*size += 1;
	}
	message[*size] = '\0';

	return message;
}