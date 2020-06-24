#ifndef H_API_HEADER_LS
#define H_API_HEADER_LS

void figure_type();

void vector_functions();

void vector_file_function(float* (*function)(float*, float*, int, size_t, size_t, int));
void matrix_file_function(float* (*function)(float*, float*, int, size_t, size_t, int));

void vector_addition();
void vector_random_number_addition();
void vector_provided_number_addition();

void vector_subtraction();
void vector_random_number_subtraction();
void vector_provided_number_subtraction();

void vector_dot_product();
void vector_random_number_dot_product();
void vector_provided_number_dot_product();

void vector_scalar_multiplication();
void vector_random_number_scalar_multiplication();
void vector_provided_number_scalar_multiplication();
void vector_file_number_scalar_multiplication();

void matrix_functions();

void matrix_addition();
void matrix_random_number_addition();
void matrix_provided_number_addition();

void matrix_subtraction();
void matrix_random_number_subtraction();
void matrix_provided_number_subtraction();

void matrix_scalar_multiplication();
void matrix_random_number_scalar_multiplication();
void matrix_provided_number_scalar_multiplication();
void matrix_file_number_scalar_multiplication();

void matrix_multiplication();
void matrix_random_number_multiplication();
void matrix_provided_number_multiplication();
void matrix_file_number_multiplication();

void matrix_inversion();
void matrix_random_number_inversion();
void matrix_file_number_inversion();

#endif
