#ifndef _LAPLACE_CUDA_H_
#define _LAPLACE_CUDA_H_

#include <vector>

template <int DIM, class Real> void LaplaceBatchedCUDA(std::vector<Real>& U, const std::vector<Real>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                                 const std::vector<Real>& F, const std::vector<Real>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                                 const std::vector<std::pair<long,long>>& trg_src_lst);

void DeviceSynchronizeCUDA();

#endif
