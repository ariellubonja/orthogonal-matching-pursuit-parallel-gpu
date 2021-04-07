#include <stdio.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include <assert.h>
#include <cutil.h>
#include <cutil_inline.h>
#include "cublas.h"


#define __LINUX__
#define CONSTANT_MEM	(64*1024)
#define SHARED_MEM_PER_BLOCKS	(48*1024)
#define REGS_PER_BLOCKS	(32*1024)

const int N = 16384;		// length of source
const int M = 2048;			// number of samples
const int K = 256;			// sparsity level of source

/************************************************************************/
/* Preparation                                                          */
/************************************************************************/
#define PI	3.14159265358979323846
template<class T>void generate_sensing_matrix(T *Phi, int M, int N){
	for(int m=0; m<M; m++){// for each row
		for(int n=0; n<N; n++){// for each column
			T sum = T(-6);
			for(int i=0; i<12; i++)	sum += T(rand())/T(RAND_MAX);
			Phi[m*N+n] = sum/sqrt(T(M));
		}
	}
}
template<class T>void generate_spikes(T *Z, int K, int N, T lb=-128, T ub=127){
	memset(Z, 0, N*sizeof(T));
	for(int k=0,temp; k<K; k++){
		do{temp = (rand()%N);}while(Z[temp]);
		Z[temp] = T((rand()+1)*(ub-lb))/RAND_MAX + lb;
	}
}

/************************************************************************/
/* OMP on CPU                                                           */
/************************************************************************/
float *u1 = NULL, *u2 = NULL;
float *R = NULL, *P = NULL, *Z0 = NULL, *H = NULL;
template<class T>void new_omp_cpu(int M, int N, int K){
	R = new T[M];
	P = new T[N];
	Z0 = new T[N];
	H = new T[K*K];
	u1 = new T[K];
	u2 = new T[K];
}
void free_omp_cpu(void){
	delete[] R;
	delete[] P;
	delete[] H;
	delete[] Z0;
	delete[] u1;
	delete[] u2;
}
template<class T>void omp_cpu(T* Zr, int* idx, const T* Y, const T* A_T, const T *AA, int M, int N, int K){
	memset(Zr, 0, N*sizeof(T));
	
	// init Z0
	for(int i=0; i<N; i++){
		Z0[i] = 0;
		for(int j=0; j<M; j++){
			Z0[i] += A_T[(i*M) + j] * Y[j];
		}
	}

	for(int k=0; k<K; k++){

		// step 1: project
		for(int i=0; i<N; i++){
			if(k){
				P[i] = 0;
				if(Zr[i])	continue;
				for(int j=0; j<M; j++)	P[i] += A_T[(i*M) + j] * R[j];
				P[i] = fabs(P[i]) / AA[i];
			}else{
				P[i] = fabs(Z0[i]) / AA[i];
			}
		}

		// step 2: find spike
		idx[k] = 0;
		T spike = P[0];
		for(int i=1; i<N; i++){
			if(P[i]>spike){
				spike = P[i];
				idx[k] = i;
			}
		}

		// step 3: matrix inverse
		if(!k)	H[0] = 1.0/(AA[idx[0]]*AA[idx[0]]);
		else{
			// step 1: compute u_1
			for(int i=0; i<k; i++){
				u1[i] = 0;
				for(int j=0; j<M; j++){
					u1[i] += A_T[idx[i]*M + j] * A_T[idx[k]*M + j];
				}
			}

			// step 2: compute u_2
			for(int i=0; i<k; i++){
				u2[i] = 0.0;
				for(int j=0; j<k; j++){
					u2[i] += H[i*K + j]*u1[j];
				}
			}

			// step 3: compute d
			float d = AA[idx[k]]*AA[idx[k]];
			for(int i=0; i<k; i++){
				d -= u1[i]*u2[i];
			}
			d = 1/d;

			// step 6: compute H
			for(int i=0; i<k; i++){
				for(int j=0; j<k; j++){
					H[i*K + j] += d*u2[i]*u2[j];
				}
			}
			for(int i=0; i<k; i++){
				H[i*K + k] = -d*u2[i];
				H[k*K + i] = -d*u2[i];
			}
			H[k*K + k] = d;
 		}

		// Zr <= H*Z0
		for(int i=0; i<=k; i++){
			Zr[idx[i]] = T(0);
			for(int j=0; j<=k; j++){
				Zr[idx[i]] += H[i*K + j] * Z0[idx[j]];
			}
		}

		memcpy(R, Y, M*sizeof(T));
		for(int i=0; i<M; i++){
			for(int t=0; t<=k; t++){
				R[i] -= Zr[idx[t]]*A_T[idx[t]*M + i];
			}
		}
	}
}


/************************************************************************/
/* Matrix Inverse on GPU                                                */
/************************************************************************/
#define bx blockIdx.x
#define tx threadIdx.x
#define ty threadIdx.y
texture<float4, 2, cudaReadModeElementType>texRefA;
template<class T>__global__ static void d_renew_R(T *R, const T*Y, const T *Zr, const cudaArray *A_T, const int *idx, int M, int k){
	int i = ((bx*blockDim.x)+tx)<<2, t, j;
	float zr;
	float4 a;
	T temp[4];
	temp[0] = Y[i  ];
	temp[1] = Y[i+1];
	temp[2] = Y[i+2];
	temp[3] = Y[i+3];
	for(t=0; t<=k; t++){
		j = idx[t];
		zr = Zr[j];
		a = tex2D(texRefA, (i>>2), j);
		temp[0] -=  zr * a.x;
		temp[1] -=  zr * a.y;
		temp[2] -=  zr * a.z;
		temp[3] -=  zr * a.w;
	}
	R[i  ] = temp[0];
	R[i+1] = temp[1];
	R[i+2] = temp[2];
	R[i+3] = temp[3];
}
template<class T>__global__ static void d_H_x_Z0(T *Zr, const T *H, const T *Z0, const int *idx, int K){
	int n = blockDim.x;
	int i = bx;
	if(n<4){// n = 1, 2, 3
		if(!threadIdx.x){// the 0-th thread
			T temp = T(0);
			for(int j=0; j<n; j++)	temp += H[(i*K) + j] * Z0[idx[j]];
			Zr[idx[i]] = temp;
		}else	return;
	}else{// n>=4
		int j = threadIdx.x;
		__shared__ T temp[1024];
		temp[j] = H[(i*K) + j] * Z0[idx[j]];
		__syncthreads();

		int t = 1<<int(ceil(log(T(n))/log(2.0))-1);
		while(t){
			if((j<t) && (j+t)<n)	temp[j] += temp[j+t];
			t>>=1;
			__syncthreads();

		}
		if(!j)	Zr[idx[i]] = temp[0];
	}
}

template<class T>__global__ static void d_cal_u1(T *u1, const cudaArray *A_T, const int *idx, int M, int k){
	int i = bx, j = tx, idx_i = idx[i], idx_k = idx[k];
	__shared__ T temp[256];
	temp[j] = 0;
	__syncthreads();

	for(int J=0; J<(M>>2); J+=256){
		float4 ai = tex2D(texRefA, J+j, idx_i);
		float4 ak = tex2D(texRefA, J+j, idx_k);
		temp[j] += ai.x * ak.x + ai.y * ak.y + ai.z * ak.z + ai.w * ak.w;
	}
	__syncthreads();

	if(j<128)	temp[j] += temp[j+128];
	if(j<64)	temp[j] += temp[j+64];
	if(j<32)	temp[j] += temp[j+32];
	if(j<16)	temp[j] += temp[j+16];
	if(j<8)		temp[j] += temp[j+8];
	if(j<4)		temp[j] += temp[j+4];
	if(j<2)		temp[j] += temp[j+2];
	if(j<1)		u1[bx] = temp[0] + temp[1];
}
template<class T>__global__ static void d_cal_u2(T *u2, const T *H, const T *u1, int K, int k){
	int i = tx;
	T temp = 0;
	for(int j=0; j<k; j++){
		temp += H[i*K + j]*u1[j];
	}
	u2[i] = temp;
}
template<class T>__global__ static void d_cal_d(T *d, const T *AA, const int *idx, const T *u1, const T *u2, int k){
	T temp = AA[idx[k]];
	(*d) = temp*temp;
	for(int i=0; i<k; i++){
		(*d) -= u1[i]*u2[i];
	}
	(*d) = 1/(*d);
}
template<class T>__global__ static void d_cal_H(T *H, const T *u2, T *d, int K, int k, const T *AA, const int *idx, int N){
	if(k){
		int i = tx, j = bx;
		if((i<k)&&(j<k)){
			H[i*K + j] += (*d)*u2[i]*u2[j];
		}else if((i==k)&&(j==k)){
			H[i*K + j] = (*d);
		}else if(i==k){
			H[i*K + j] = -(*d)*u2[j];
		}else{// j==k
			H[i*K + j] = -(*d)*u2[i];
		}
	}else{
		T temp = AA[idx[0]];
		H[0] = 1.0/(temp*temp);
	}
}

__global__ static void mv_kernel(float* y, cudaArray* A, float* x, int m, int n, float* Z0, float* AA, float* Zr){
	__shared__ float xs[16][16];
	__shared__ float Ps[16][16];
	float4 a;
	float *Psptr = (float*)Ps + (ty<<4) + tx;
	int ay = (bx<<4) + ty;
	float *xptr = x + (ty<<4) + tx;
	float *xsptr = (float*)xs + (tx<<2);

	*Psptr = 0.0f;
	int i;
	for(i = 0; i < (n & ~255); i += 256, xptr += 256){
		xs[ty][tx] = *xptr;
		__syncthreads();
		int ax = tx + (i>>2);
		a = tex2D(texRefA, ax     , ay);
		*Psptr += a.x * *(xsptr      ) + a.y * *(xsptr +   1) + a.z * *(xsptr +   2) + a.w * *(xsptr +   3); 
		a = tex2D(texRefA, ax + 16, ay);
		*Psptr += a.x * *(xsptr +  64) + a.y * *(xsptr +  65) + a.z * *(xsptr +  66) + a.w * *(xsptr +  67); 
		a = tex2D(texRefA, ax + 32, ay);
		*Psptr += a.x * *(xsptr + 128) + a.y * *(xsptr + 129) + a.z * *(xsptr + 130) + a.w * *(xsptr + 131); 
		a = tex2D(texRefA, ax + 48, ay);
		*Psptr += a.x * *(xsptr + 192) + a.y * *(xsptr + 193) + a.z * *(xsptr + 194) + a.w * *(xsptr + 195); 
		__syncthreads();
	}

	if (i + (ty<<4) + tx < n) {
		xs[ty][tx] = *xptr;
	}
	__syncthreads();
	int j;
	for (j = 0; j < ((n - i) >> 6); j++, xsptr += 61) {
		a = tex2D(texRefA, tx + (i >> 2) + (j << 4), ay);
		*Psptr += a.x * *xsptr++ + a.y * *xsptr++ + a.z * *xsptr++ + a.w * *xsptr; 
	}
	__syncthreads();
	int remain = (n - i) & 63;
	if ((tx << 2) < remain) {
		a = tex2D(texRefA, tx + (i >> 2) + (j << 4), ay);
		*Psptr += a.x * *xsptr++;
	}
	if ((tx << 2) + 1 < remain) *Psptr += a.y * *xsptr++;
	if ((tx << 2) + 2 < remain) *Psptr += a.z * *xsptr++;
	if ((tx << 2) + 3 < remain) *Psptr += a.w * *xsptr;
	__syncthreads();

	if (tx < 8) *Psptr += *(Psptr + 8);
	if (tx < 4) *Psptr += *(Psptr + 4);
	if (tx < 2) *Psptr += *(Psptr + 2);
	if (tx < 1) *Psptr += *(Psptr + 1);

	__syncthreads();
	if (ty == 0 && (bx << 4) + tx < m){
		i =  (bx<<4) + tx;
		if(Z0)	Z0[i] = Ps[tx][0];
		y[i] = Zr[i]?0:abs(Ps[tx][0])/AA[i];
	}
}

//template<class T>__global__ void d_find_local_spike(int *idx, T *P, int N){
//	int j = (ty<<4) + tx;
//	__shared__ int loc_idx[256];
//	__shared__ T loc_P[256];
//	loc_idx[j] = j;
//	loc_P[j] = P[j];
//	__syncthreads();
//
//	for(int J=256; J<N; J+=256){
//		T temp = P[J+j];
//		if(temp>loc_P[j]){
//			loc_idx[j] = J+j;
//			loc_P[j] = temp;
//		}
//	}
//
//	__syncthreads();
//
//	int inc = ((M+1)>>1);
//	while(inc){
//		if(j<inc){
//			int k = (j + inc);
//			if(loc_P[j]<loc_P[k]){
//				loc_P[j] = loc_P[k];	
//				loc_idx[j] = loc_idx[k];
//			}
//		}else return;
//		inc>>=1;
//		__syncthreads();
//	}
//
//	if(!j){
//		P[i*M] = loc_P[0];
//		idx[i*M] = loc_idx[0];
//	}else return;
//}


template<class T>__global__ void d_find_local_spike(int *idx, T *P, int N){
	int i = blockIdx.x, j = threadIdx.x, M = blockDim.x;
	__shared__ int loc_idx[1024];
	__shared__ T loc_P[1024];
	loc_idx[j] = (i*M) + j;
	loc_P[j] = P[(i*M) + j];
	__syncthreads();

	int inc = ((M+1)>>1);
	while(inc){
		if(j<inc){
			int k = (j + inc);
			if(loc_P[j]<loc_P[k]){
				loc_P[j] = loc_P[k];	
				loc_idx[j] = loc_idx[k];
			}
		}else return;
		inc>>=1;
		__syncthreads();
	}

	if(!j){
		P[i*M] = loc_P[0];
		idx[i*M] = loc_idx[0];
	}else return;
}
template<class T>__global__ void d_find_global_spike(int *idx, T *P, int M){
	int i = threadIdx.x;
	int inc = ((blockDim.x+1)>>1);

	__shared__ T sm_P[512];
	__shared__ int sm_idx[512];
	sm_P[i] = P[i*M];	
	sm_idx[i] = idx[i*M];
	__syncthreads();

	while(inc){
		if(i<inc){
			int j = i + inc;
			if(sm_P[i]<sm_P[j]){
				sm_P[i] = sm_P[j];	
				sm_idx[i] = sm_idx[j];
			}
		}else return;
		inc>>=1;
		__syncthreads();
	}
	if(!i)	idx[0] = sm_idx[0];
}

float *d_u1 = NULL, *d_u2 = NULL;
float *d_H = NULL, *d_R = NULL, *d_P = NULL, *d_Z0 = NULL;
template<class T>void omp_gpu(T* d_Zr, int *d_idx, T* d_Y, cudaArray* d_A_T, T *d_AA, int M, int N, int K){
	cutilSafeCall(cudaMemset(d_Zr, 0, sizeof(T)*N));
	for(int k=0; k<K; k++){// for each spike
		dim3 grid((N>>4), 1, 1), block(16, 16, 1); 
		mv_kernel<<<grid, block>>>(d_P, d_A_T, (k?d_R:d_Y), N, M, (k?NULL:d_Z0), d_AA, d_Zr);
		d_find_local_spike<<<(N+1023)/1024, min(N,1024)>>>(d_idx+k, d_P, N);				// O(N)
		d_find_global_spike<<<1, (N+1023)/1024>>>(d_idx+k, d_P, min(N,1024));		
		if(k){
			d_cal_u1<<<k, 256>>>(d_u1, d_A_T, d_idx, M, k);			// O(kM)
			d_cal_u2<<<1,k>>>(d_u2, d_H, d_u1, K, k);				// O(k^2)
			d_cal_d<<<1,1>>>(d_u1+k, d_AA, d_idx, d_u1, d_u2, k);	// O(k)	
		}
		d_cal_H<<<k+1,k+1>>>(d_H, d_u2, d_u1+k, K, k, d_AA, d_idx, N);	// O((k+1)*(k+1))
		d_H_x_Z0<<<k+1, k+1>>>(d_Zr, d_H, d_Z0, d_idx, K);				// O((k+1)*(k+1))				// d_Zr <= d_H_I * d_Z0[d_idx]
		d_renew_R<<<((M>>2)+255)/256, min((M>>2),256)>>>(d_R, d_Y, d_Zr, d_A_T, d_idx, M, k);	// O((k+1)*M)	// d_R <= d_Y - d_A_T[d_idx] * d_Zr[d_idx]
	}
}
template<class T>void new_omp_gpu(int M, int N, int K){
	cutilSafeCall(cudaMalloc((void**)&d_H, sizeof(T)*(K*K)));
	cutilSafeCall(cudaMalloc((void**)&d_R, sizeof(T)*M));
	cutilSafeCall(cudaMalloc((void**)&d_P, sizeof(T)*N));
	cutilSafeCall(cudaMalloc((void**)&d_Z0, sizeof(T)*N));
	cutilSafeCall(cudaMalloc((void**)&d_u1, sizeof(T)*K));
	cutilSafeCall(cudaMalloc((void**)&d_u2, sizeof(T)*K));
}
void free_omp_gpu(){
	cutilSafeCall(cudaFree(d_Z0));
	cutilSafeCall(cudaFree(d_H));
	cutilSafeCall(cudaFree(d_R));
	cutilSafeCall(cudaFree(d_P));
	cutilSafeCall(cudaFree(d_u1));
	cutilSafeCall(cudaFree(d_u2));
}


/************************************************************************/
/* Initialize GPU                                                       */
/************************************************************************/
bool InitCUDA(void){
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0){
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++){
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess){
			printf(" --- General Information for device %d ---\n", i);
			printf("Name: %s\n", prop.name );
			printf("Compute capability: %d.%d\n", prop.major, prop.minor);
			printf("Clock rate: %d\n", prop.clockRate);

			printf("Device copy overlap: ");
			if(prop.deviceOverlap)
				printf( "Enabled\n" );
			else
				printf( "Disabled\n" );

			printf( "Kernel execition timeout : " );
			if (prop.kernelExecTimeoutEnabled)
				printf( "Enabled\n" );
			else
				printf( "Disabled\n" );

			printf( " --- Memory Information for device %d ---\n", i );
			printf( "Total global mem: %ld\n", prop.totalGlobalMem );
			printf( "Total constant Mem: %ld\n", prop.totalConstMem );
			printf( "Max mem pitch: %ld\n", prop.memPitch );
			printf( "Texture Alignment: %ld\n", prop.textureAlignment );

			printf( " --- MP Information for device %d ---\n", i );
			printf( "Multiprocessor count: %d\n", prop.multiProcessorCount );
			printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
			printf( "Registers per mp: %d\n", prop.regsPerBlock );
			printf( "Threads in warp: %d\n", prop.warpSize );
			printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
			printf( "Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
			printf( "Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
			printf( "\n" );

			if(prop.major >= 1) {
				break;
			}
		}
	}

	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n\n");
	return true;
}

int main(int argc, char *argv[]){
	if(!InitCUDA())	return 0;

	// get parameters from command lines
	printf("(N, M, K) = (%d, %d, %d)\n", N, M, K);

	// 
	float *A = new float[M*N];		// A = Phi*Psi, of size M*N (M rows, N columns)
	float *A_T = new float[N*M];	// A_T = A^T
	float *AA = new float[N];		// AA = ||A||_2
	float *Y = new float[M];		// measurements/samples
	float *Z = new float[N];		// K-sparse spikes
	float *Zr = new float[N];		// reconstruction of Z
	int *idx = new int[K];

	// allocate all memory for GPU computation
	float *d_AA = NULL;
	cutilSafeCall(cudaMalloc((void**)&d_AA, sizeof(float)*N));

	//float *d_A_T = NULL;
	//cutilSafeCall(cudaMalloc((void**)&d_A_T, sizeof(float)*(N*M)));

	float *d_Y = NULL, *d_Zr = NULL;
	cutilSafeCall(cudaMalloc((void**)&d_Y,   sizeof(float)*M));
	cutilSafeCall(cudaMalloc((void**)&d_Zr,  sizeof(float)*N));

	int *d_idx = NULL;
	cutilSafeCall(cudaMalloc((void**)&d_idx, sizeof(int)*(K+N)));

	time_t t; 
	srand((unsigned)time(&t));
	
	// generate sensing matrix, Phi, of size M*N
	generate_sensing_matrix(A, M, N);
	for(int j=0; j<N; j++){
		AA[j] = 0;
		for(int i=0; i<M; i++){
			A_T[j*M + i] = A[i*N + j];
			AA[j] += A[i*N + j]*A[i*N + j];
		}
		AA[j] = sqrt(AA[j]);
	}
	cutilSafeCall(cudaMemcpy(d_AA, AA, sizeof(float)*N, cudaMemcpyHostToDevice));		// d_AA <= AA
	//cutilSafeCall(cudaMemcpy(d_A_T, A_T, sizeof(float)*N*M, cudaMemcpyHostToDevice));	// d_A_T <= A_T

	cudaArray *d_A_T = NULL;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaMallocArray(&d_A_T, &channelDesc, (M>>2), N);
	cudaMemcpy2DToArray(d_A_T, 0, 0, A_T, M*sizeof(float), M*sizeof(float), N, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texRefA, d_A_T);

	// generate K-sparse Z, uniformly ranging from lb to hb
	generate_spikes(Z, K, N);

	
	/************************************************************************/
	/* Sampling                                                             */
	/************************************************************************/
	for(int i=0; i<M; i++){
		Y[i] = 0;
		for(int j=0; j<N; j++){
			Y[i] += A[(i*N) + j] * Z[j];
		}
	}

#ifdef __LINUX__
	float elapsedTime;
	cudaEvent_t start, stop;
#else
	unsigned timer;
#endif

	/************************************************************************/
	/* OMP on GPU                                                           */
	/************************************************************************/
	new_omp_gpu<float>(M, N, K);

#ifdef __LINUX__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#else
	timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
#endif

	cutilSafeCall(cudaMemcpy(d_Y, Y, sizeof(float)*M, cudaMemcpyHostToDevice));		// d_A <= A
	omp_gpu(d_Zr, d_idx, d_Y, d_A_T, d_AA, M, N, K);
	cutilSafeCall(cudaMemcpy(Zr, d_Zr, sizeof(float)*N, cudaMemcpyDeviceToHost));		// Zr <= d_Zr

#ifdef __LINUX__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Kernel+I/O time is: %f ms\n", elapsedTime);
	//cudaEventDestory(start);
	//cudaEventDestory(stop);
#else
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("Kernel+I/O time is: %f ms\n", cutGetTimerValue(timer));
	CUT_SAFE_CALL(cutDeleteTimer(timer));
#endif

	free_omp_gpu();

	cutilSafeCall(cudaFree(d_Y));
	cutilSafeCall(cudaFree(d_AA));
	cutilSafeCall(cudaFree(d_idx));
	cutilSafeCall(cudaFree(d_Zr));
	//cutilSafeCall(cudaFree(d_A_T));

	// test correctness
	int nerr = 0;
	for(int i=0; i<N; i++)	nerr += (Zr[i] && !Z[i]) || (!Zr[i] && Z[i]);
	printf("Number of Errors: %d\n\n", nerr);



	/************************************************************************/
	/* OMP on CPU                                                           */
	/************************************************************************/
	new_omp_cpu<float>(M, N, K);

#ifdef __LINUX__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#else
	timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
#endif

	omp_cpu(Zr, idx, Y, A_T, AA, M, N, K);

#ifdef __LINUX__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("CPU time is: %f ms\n", elapsedTime);
	//cudaEventDestory(start);
	//cudaEventDestory(stop);
#else
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("Host time is:%f ms\n", cutGetTimerValue(timer));
	CUT_SAFE_CALL(cutDeleteTimer(timer));
#endif

	free_omp_cpu();


	// test correctness
	nerr = 0;
	for(int i=0; i<N; i++)	nerr += (Zr[i] && !Z[i]) || (!Zr[i] && Z[i]);
	printf("Number of Errors: %d\n\n", nerr);


	/************************************************************************/
	/* free memory                                                          */
	/************************************************************************/
	delete[] A;
	delete[] A_T;
	delete[] AA;
	delete[] Y;
	delete[] Z;
	delete[] Zr;
	delete[] idx;

	getchar();
}