#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <math.h>

// all matrices are zero-based column-major matrices
//   zero-based column-major: A(r,c) = A[r*col_size + c]
//   zero-based row-major:    A(r,c) = A[r + row_size * c]

float* AX_B_inverse(float *A, float *B, int n);

float rand_f32(float lower, float upper)
{
	float frac = (float) rand()/RAND_MAX;  // between 0 and 1
	//printf("%f ",frac);
	float r = lower + (upper - lower) * frac;
	return r;
}

void mat_print(char *msg, float *A, int n){
	int c,r;
	printf("%s: %d x %d\n",msg,n,n);
	for(r = 0; r < n; r++){
		for(c = 0; c < n; c++){
			float v = *(A + r*n + c);
			printf("%g", v);
			if(c != n-1) printf(",");
		}
		printf("\n");
	}
	printf("\n");
}

float* AX_B_inverse(float *A, float *B, int n){
	int i,j,k;

	//mat_print("AX_B_inverse:A:", A, n);
	for(k = 0; k < n; k++){
		// update the k-th column of A
		float temp = sqrt(A[(k*n) + k]);
		for(i = k; i < n; i++){
			A[(i*n) + k] /= temp;
		}

		// update column (k+1) to column (n-1) of A
		for(j = k+1; j < n; j++){// loop over column
			for(i = j; i < n; i++){// loop over row
				A[i*n + j] -= A[i*n + k] * A[j*n + k];
			}
		}
	}

	//mat_print("AX_B_inverse:A2:", A, n);

	//mat_print("AX_B_inverse:B1:", B, n);

	for(i = 0; i < n; i++){
		for(k = 0; k < n; k++){
			for(j = 0; j <= i-1; j++){
				B[i*n+k] -= A[i*n + j] * B[j*n+k];
			}
			B[i*n+k] /= A[i*n + i];
		}

		//for(k = 0; k < n; k++){
		//}
	}

	//mat_print("AX_B_inverse:B2:", B, n);


	for(j = (n-1); j >= 0; j--){
		for(k = 0; k < n; k++){
			for(i = (j+1); i < n; i++){
				B[j*n + k] -= A[i*n + j] * B[i*n+k];
			}
			B[j*n+k] /= A[j*n + j];
		}

		//for(k = 0; k < n; k++){
		//}
	}

	//mat_print("AX_B_inverse:B4:", B, n);

	return B;
}

int main(int argc,char *argv)
{
	int i,j,k;

	int n = 3;
	float *A0 = (float *)malloc(n*n*sizeof(float));
	float *A = (float *)malloc(n*n*sizeof(float));
	float *B = (float *)malloc(n*n*sizeof(float));

	// produce covariance matrix (positive semidefinite) A
	// A = A0' * A0
	srand(0);

	for(i = 0 ; i < n; i++){
		for(j= 0 ; j < n; j++){
			A0[i*n + j] = rand_f32(0, 1);
			B[i*n + j] = (i==j) ? 1.0 : 0.0;
		}
	}

	mat_print("A0:", A0, n);

	for(i = 0 ; i < n; i++){
		for(j = 0 ; j < n; j++){
			float sum=0;
			for(k = 0; k < n; k++){
				sum += A0[k*n + i] * A0[k*n + j];
			}
			A[i*n + j] = sum;//ceil(sum);
		}
	}


	A[0] = 1;
	A[1] = 2;
	A[2] = 2;

	A[3] = 2;
	A[4] = 6;
	A[5] = 6;

	A[6] = 2;
	A[7] = 6;
	A[8] = 7;

	mat_print("A:", A, n);
	mat_print("B:", B, n);


	float *Ainv = AX_B_inverse(A, B, n);


	mat_print("Ainv:", Ainv, n);

	free(A);
	free(B);

	getchar();
	return 0;
}
