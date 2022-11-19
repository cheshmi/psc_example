//
// Created by Kazem on 11/15/22.
// Code are extracted from the PSC code repository
//

#ifndef PSC_EXAMPLE_PSC_H
#define PSC_EXAMPLE_PSC_H


#ifdef MKL
#include "mkl.h"
#endif

#include<immintrin.h>

// For now tested for AVX2
#ifdef __AVX2__
 typedef union
 {
  __m256d v;
  double d[4];
 } v4df_t;

 typedef union
 {
  __m128i v;
  int d[4];
 } v4if_t;

 union vector128
 {
  __m128i     i128;
  __m128d     d128;
  __m128      f128;
 };

#endif




void inline psc_general(double *y, const double *Ax, const int *Ap, const int *Ai,
                        const double *x, const int offset, const int lb, const int ub){
 v4df_t Lx_reg, result, x_reg;
 for (int i = lb, ii=0; i <ub; ++i, ++ii) {
  result.v = _mm256_setzero_pd();
  int j, k;
  for (j = Ap[i], k=i*offset; j < Ap[i+1]-4; j+=4, k+=4) {
   //y[i] += Ax[k] * x[j];
   //_mm256_mask_i32gather_pd()
   const int* aij = Ai+j;
    x_reg.d[0] = x[*aij]; /// TODO replaced with gather
    x_reg.d[1] = x[*(aij+1)];
    x_reg.d[2] = x[*(aij+2)];
    x_reg.d[3] = x[*(aij+3)];
   Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
   result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
  }
  auto h0 = _mm_hadd_pd(_mm256_extractf128_pd(result.v,0), _mm256_extractf128_pd(result.v,1));
  y[i] += (h0[0] + h0[1]);
  for (; j < Ap[i+1]; ++j) {
    y[i] += Ax[j] * x[Ai[j]];
  }
 }
}


void inline psc_general_2D(double *y, const double *Ax, const int *Ap, const int *Ai,
                           const double *x, const int offset, int lb, int ub){
 v4df_t Lx_reg, Lx_reg2, result, result2, x_reg;
 for (int i = lb, ii=0; i <ub; i+=2, ii+=2) {
  result.v = _mm256_setzero_pd();
  result2.v = _mm256_setzero_pd();
  int j, k, k1;
  for (j = Ap[i], k=i*offset, k1 = (i+1)*offset; j < Ap[i+1]-4; j+=4, k+=4,
                                                              k1+=4) {
   const int* aij = Ai+j;
   x_reg.d[0] = x[*aij]; /// TODO replaced with gather
   x_reg.d[1] = x[*(aij+1)];
   x_reg.d[2] = x[*(aij+2)];
   x_reg.d[3] = x[*(aij+3)];
   Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
   Lx_reg2.v = _mm256_loadu_pd((double *) (Ax + k1)); // Skylake	7
   // 	0.5
   result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
   result2.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg.v,result2.v);//Skylake	4	0.5
  }
  auto h0 = _mm256_hadd_pd(result.v, result2.v);
  y[i] += (h0[0] + h0[2]);
  y[i+1] += (h0[1] + h0[3]);
  for (; j < Ap[i+1]; ++j) {
   double x_aij = x[Ai[j]];
   y[i] += Ax[j] * x_aij;
   y[i+1] += Ax[j+offset] * x_aij;
  }
  //y[i] += (h0[0] + h0[1]);
 }
}

void inline psc_general_2D_4(double *y, const double *Ax, const int *Ap, const int *Ai,
                           const double *x, const int offset, int lb, int ub){
 v4df_t Lx_reg, Lx_reg2, Lx_reg3, Lx_reg4, result, result2, result3,
   result4, x_reg, x_reg2;
 int i;
 for (i = lb; i <ub; i+=4) {
  result.v = _mm256_setzero_pd();
  result2.v = _mm256_setzero_pd();
  result3.v = _mm256_setzero_pd();
  result4.v = _mm256_setzero_pd();
  int k=i*offset, k1=(i+1)*offset, k2=(i+2)*offset, k3=(i+3)*offset, j;
  for (j = Ap[i]; j < Ap[i+1]-4; j+=4, k+=4, k1+=4, k2+=4, k3+=4) {
   const int* aij = Ai+j;
   x_reg.d[0] = x[*aij]; /// TODO replaced with gather
   x_reg.d[1] = x[*(aij+1)];
   x_reg.d[2] = x[*(aij+2)];
   x_reg.d[3] = x[*(aij+3)];
   Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
   Lx_reg2.v = _mm256_loadu_pd((double *) (Ax + k1)); // Skylake	7
   Lx_reg3.v = _mm256_loadu_pd((double *) (Ax + k2)); // Skylake	7
   Lx_reg4.v = _mm256_loadu_pd((double *) (Ax + k3)); // Skylake	7
   // 	0.5
   result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
   result2.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg.v,result2.v);//Skylake	4	0.5
   result3.v = _mm256_fmadd_pd(Lx_reg3.v,x_reg.v,result3.v);//Skylake	4	0.5
   result4.v = _mm256_fmadd_pd(Lx_reg4.v,x_reg.v,result4.v);//Skylake	4	0.5
  }
  auto h0 = _mm256_hadd_pd(result.v, result2.v);
  y[i] += (h0[0] + h0[2]);
  y[i+1] += (h0[1] + h0[3]);
  h0 = _mm256_hadd_pd(result3.v, result4.v);
  y[i+2] += (h0[0] + h0[2]);
  y[i+3] += (h0[1] + h0[3]);
  for (; j < Ap[i+1]; ++j) {
   double x_aij = x[Ai[j]];
   y[i] += Ax[j] * x_aij;
   y[i+1] += Ax[j+offset] * x_aij;
   y[i+2] += Ax[j+(2*offset)] * x_aij;
   y[i+3] += Ax[j+(3*offset)] * x_aij;
  }
 }
}




#endif //PSC_EXAMPLE_PSC_H
