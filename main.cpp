//
// Created by Kazem on 11/15/22.
//

#include "aggregation/def.h"
#include "aggregation/sparse_io.h"
#include "aggregation/test_utils.h"
#include "aggregation/utils.h"
#include "aggregation/sparse_utilities.h"

#include "psc.h"

#include <fstream>

#ifdef MKL
#include "mkl.h"
#endif

#define NTIMES 7

int test_PSC(int argc, char *argv[]);

void spmv_baseline(const int m, const int *Ap, const int *Ai, const double *Ax, const double *x, double *y);

void spmv_blas(const double* Ax, const int i0, const int ax_offset, const int num_block, const int *col_loc, const int *col_blk_size,
               const double *x, double *y);

/// A specialized function for a specific pattern.
/// \param A
/// \param blk_sizes
/// \param col_loc
/// \param nnz_offset
/// \param num_blks
void extracting_meta_data(const sym_lib::CSR *A, std::vector<int>& blk_sizes, std::vector<int>& col_loc,
                          int &nnz_offset, int &num_blks){
 nnz_offset = A->p[1] - A->p[0];
 int c_blk_size = 0;
 col_loc.push_back(0);
 for (int j = A->p[0]; j < A->p[1]; ++j) {// checks only one row since all rows are similar.
  if(A->i[j] + 1 == A->i[j+1]){
   c_blk_size++;
  } else {
   blk_sizes.push_back(c_blk_size+1);
   col_loc.push_back(A->i[j+1]);
   c_blk_size=0;
   num_blks++;
  }
 }
}

int main(int argc, char *argv[]){
 test_PSC(argc, argv);
 std::cout<<"\n";
 return 1;
}


int test_PSC(int argc, char *argv[]){
 if(argc < 2)
  return 0;
 std::string f1 = argv[1];
 std::string matrix_name = f1;
 //auto *A_csc = sym_lib::read_mtx(f1);
 sym_lib::CSC *A_csc = NULLPNTR;
 std::ifstream fin(f1);
 sym_lib::read_mtx_csc_real(fin, A_csc);
 //sym_lib::print_csc(1, " A:\n ", A_csc);
 auto *A_csr = sym_lib::csc_to_csr(A_csc);
 int m = A_csr->m, n = A_csr->n;
 double *x = new double[n], *y = new double[m](), *y_blas = new double[m](),
 *y_psc_g = new double[m](), *y_psc_2d = new double[m](), *y_psc_2d4 = new double[m]();
 std::fill_n(x, n, 1.0);
 int num_blks = 0;
 double eps = 1e-8;
 mkl_set_dynamic(0);
 mkl_set_num_threads(1);
 mkl_domain_set_num_threads(1, MKL_DOMAIN_BLAS);

 /// Scalarized code
 std::vector<sym_lib::timing_measurement> t_spmv_array;
 for (int i = 0; i < NTIMES; ++i) {
  std::fill_n(y, m, .0);
  sym_lib::timing_measurement t_spmv; t_spmv.start_timer();
  spmv_baseline(m, A_csr->p, A_csr->i, A_csr->x, x, y);
  t_spmv.measure_elapsed_time();
  t_spmv_array.push_back(t_spmv);
 }
 auto t_baseline_sec = sym_lib::time_median(t_spmv_array).elapsed_time;


 /// BLAS-based
 std::vector<int> blk_size, col_locs;
 int nnz_offset;
 extracting_meta_data(A_csr, blk_size, col_locs, nnz_offset, num_blks);
 std::vector<sym_lib::timing_measurement> t_blas_array;
 for (int i = 0; i < NTIMES; ++i) {
  std::fill_n(y_blas, m, .0);
  sym_lib::timing_measurement t_spmv; t_spmv.start_timer();
  spmv_blas(A_csr->x, m, nnz_offset, num_blks, col_locs.data(), blk_size.data(), x, y_blas);
  t_spmv.measure_elapsed_time();
  t_blas_array.push_back(t_spmv);
 }
 auto t_blas_sec = time_median(t_blas_array).elapsed_time;
 auto check_blas = sym_lib::is_equal<double>(0, m, y, y_blas, eps);

 /// PSC Compile-time
 std::vector<sym_lib::timing_measurement> t_pscg_array;
 for (int i = 0; i < NTIMES; ++i) {
  std::fill_n(y_psc_g, m, .0);
  sym_lib::timing_measurement t_spmv; t_spmv.start_timer();
  psc_general(y_psc_g, A_csr->x, A_csr->p, A_csr->i, x, nnz_offset, 0, m);
  t_spmv.measure_elapsed_time();
  t_pscg_array.push_back(t_spmv);
 }
 auto t_pscg_sec = time_median(t_pscg_array).elapsed_time;
 auto check_psc_g = sym_lib::is_equal<double>(0, m, y, y_psc_g, eps);


 /// PSC 2D unroll-2
 std::vector<sym_lib::timing_measurement> t_psc2d_array;
 for (int i = 0; i < NTIMES; ++i) {
  std::fill_n(y_psc_2d, m, .0);
  sym_lib::timing_measurement t_spmv; t_spmv.start_timer();
  psc_general_2D(y_psc_2d, A_csr->x, A_csr->p, A_csr->i, x, nnz_offset, 0, m);
  t_spmv.measure_elapsed_time();
  t_psc2d_array.push_back(t_spmv);
 }
 auto t_psc2d_sec = time_median(t_psc2d_array).elapsed_time;
 auto check_psc_2d = sym_lib::is_equal<double>(0, m, y, y_psc_2d, eps);

 /// PSC 2D uroll-4
 std::vector<sym_lib::timing_measurement> t_psc2d4_array;
 for (int i = 0; i < NTIMES; ++i) {
  std::fill_n(y_psc_2d4, m, .0);
  sym_lib::timing_measurement t_spmv; t_spmv.start_timer();
  psc_general_2D_4(y_psc_2d4, A_csr->x, A_csr->p, A_csr->i, x, nnz_offset, 0, m);
  t_spmv.measure_elapsed_time();
  t_psc2d4_array.push_back(t_spmv);
 }
 auto t_psc2d4_sec = time_median(t_psc2d4_array).elapsed_time;
 auto check_psc_2d4 = sym_lib::is_equal<double>(0, m, y, y_psc_2d4, eps);

 std::cout<<matrix_name<<","<<m<<","<<n<<","<<A_csr->nnz<<",";
 std::cout<< std::setprecision(25)<<t_baseline_sec<<","<<t_blas_sec<<","<<t_pscg_sec<<","<<t_psc2d_sec<<","<<t_psc2d4_sec<<",";
 std::cout<<check_blas<<","<<check_psc_g<<","<<check_psc_2d<<","<<check_psc_2d4<<",";

 delete A_csr;
 delete A_csc;
 delete[] y; delete[] x; delete[] y_blas, delete[] y_psc_g, delete[] y_psc_2d, delete[] y_psc_2d4;
 return 1;
}


void spmv_baseline(const int m, const int *Ap, const int *Ai, const double *Ax, const double *x, double *y){
 for (int i = 0; i < m; ++i) {
  for (int j = Ap[i]; j < Ap[i+1]; ++j) {
   y[i] += Ax[j] * x[Ai[j]];
  }
 }
}


void spmv_blas(const double* Ax, const int i0, const int ax_lda, const int num_block,
               const int *col_loc, const int *col_blk_size, const double *x, double *y){
 int ax_offset = 0;
 for (int i = 0; i < num_block; ++i) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans,
              i0,  col_blk_size[i],
              1.,              // alpha
              Ax+ax_offset, ax_lda,
              x+col_loc[i], 1,
              1.,              // beta
              y, 1);
  ax_offset+=col_blk_size[i];
 }
}
