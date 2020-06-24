#include <stdio.h>
#include <stdlib.h>
#include "api_header.h"

int main() {
	while(true) {
		printf("Choose option: s - start project, q - quit, default start: ");
		char op;
		scanf(" %c", &op);
		switch(op) {
			case 's':
				figure_type();
				break;
			case 'q':
				return 0;
			default:
				figure_type();
				break;
		}
	}
	return 0;
}
