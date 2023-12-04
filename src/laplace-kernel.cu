#include <cuda.h>
#include <laplace-cuda.h>

template <int DIM, class Real> void LaplaceBatchedCUDA(std::vector<Real>& U, const std::vector<Real>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                                 const std::vector<Real>& F, const std::vector<Real>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                                 const std::vector<std::pair<long,long>>& trg_src_lst) {
  // TODO
}

template void LaplaceBatchedCUDA<2,double>(std::vector<double>& U, const std::vector<double>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                     const std::vector<double>& F, const std::vector<double>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                     const std::vector<std::pair<long,long>>& trg_src_lst);
template void LaplaceBatchedCUDA<3,double>(std::vector<double>& U, const std::vector<double>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                     const std::vector<double>& F, const std::vector<double>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                     const std::vector<std::pair<long,long>>& trg_src_lst);
