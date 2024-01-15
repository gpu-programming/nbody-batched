#include <omp.h>
#include <iostream>

#include <cuda.h>
#include <laplace-cuda.h>

template <int DIM, class Real> __global__ void LaplaceKernel(Real* U, const Real* Xt, const long* trg_interac_start, const long Nt,
                                                const Real* F, const Real* Xs, const long* src_cnt, const long* src_dsp,
                                                const long* trg_src_lst, const long Ninterac) {
  const long t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= Nt || trg_interac_start[t] == -1) return;

  Real Xt_[DIM],  U_ = 0;
  for (long k = 0; k < DIM; k++) Xt_[k] = Xt[t*DIM+k];

  const long interac_start = trg_interac_start[t];
  const long trg_node_idx = trg_src_lst[interac_start*2+0];
  for (long interac_idx = interac_start; interac_idx < Ninterac && trg_src_lst[interac_idx*2+0] == trg_node_idx; interac_idx++) {
    const long src_node_idx = trg_src_lst[interac_idx*2+1];
    const long src_dsp_ = src_dsp[src_node_idx];
    const long Ns = src_cnt[src_node_idx];
    const Real* Xs_ = Xs + src_dsp_*DIM;
    const Real* F_ = F + src_dsp_;

    for (long s = 0; s < Ns; s++) {
      Real R2 = 0;
      for (int k = 0; k < DIM; k++) {
        const Real dX = Xt_[k] - Xs_[s*DIM+k];
        R2 += dX * dX;
      }
      if (R2 > 0) {
        if (DIM == 2) U_ += F_[s] * log(R2) * 0.5;
        if (DIM == 3) U_ += F_[s] * rsqrt(R2);
      }
    }
  }
  U[t] = U_;
}


template <int DIM, class Real> void LaplaceBatchedCUDA(std::vector<Real>& U, const std::vector<Real>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                                 const std::vector<Real>& F, const std::vector<Real>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                                 const std::vector<std::pair<long,long>>& trg_src_lst) {
  //cudaFuncSetCacheConfig(LaplaceKernel<DIM,Real>, cudaFuncCachePreferL1);

  std::vector<long> trg_interac_start(Xt.size()/DIM);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < trg_cnt.size(); i++) {
    long interac_offset = std::lower_bound(trg_src_lst.begin(), trg_src_lst.end(), std::make_pair(i,(long)0)) - trg_src_lst.begin();
    for (long j = 0; j < trg_cnt[i]; j++) trg_interac_start[trg_dsp[i]+j] = (trg_src_lst[interac_offset].first == i ? interac_offset : -1);
  }

  Real *Xs_, *Xt_, *F_, *U_;
  cudaMalloc(&Xs_, Xs.size()*sizeof(Real));
  cudaMalloc(&Xt_, Xt.size()*sizeof(Real));
  cudaMalloc(&F_, F.size()*sizeof(Real));
  cudaMalloc(&U_, U.size()*sizeof(Real));

  long *trg_interac_start_, *trg_src_lst_, *src_cnt_, *src_dsp_;
  cudaMalloc(&trg_interac_start_, trg_interac_start.size()*sizeof(long));
  cudaMalloc(&trg_src_lst_, trg_src_lst.size()*2*sizeof(long));
  cudaMalloc(&src_cnt_, src_cnt.size()*sizeof(long));
  cudaMalloc(&src_dsp_, src_dsp.size()*sizeof(long));

  cudaMemcpy(Xs_, &(*Xs.begin()), Xs.size()*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(Xt_, &(*Xt.begin()), Xt.size()*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(F_, &(*F.begin()), F.size()*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemset(U_, 0, U.size()*sizeof(Real));

  cudaMemcpy(trg_interac_start_, &(*trg_interac_start.begin()), trg_interac_start.size()*sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(trg_src_lst_, &(*trg_src_lst.begin()), trg_src_lst.size()*2*sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(src_cnt_, &(*src_cnt.begin()), src_cnt.size()*sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(src_dsp_, &(*src_dsp.begin()), src_dsp.size()*sizeof(long), cudaMemcpyHostToDevice);

  const int BLOCK_SIZE = 64;
  const int BLOCK_COUNT = (Xt.size()/DIM+BLOCK_SIZE-1)/BLOCK_SIZE;
  LaplaceKernel<DIM><<<BLOCK_COUNT, BLOCK_SIZE>>>(U_, Xt_, trg_interac_start_, Xt.size()/DIM,
                                                  F_, Xs_, src_cnt_, src_dsp_,
                                                  trg_src_lst_, trg_src_lst.size());

  cudaMemcpy(&(*U.begin()), U_, U.size()*sizeof(Real), cudaMemcpyDeviceToHost);

  cudaFree(Xs_);
  cudaFree(Xt_);
  cudaFree(F_);
  cudaFree(U_);

  cudaFree(trg_interac_start_);
  cudaFree(trg_src_lst_);
  cudaFree(src_cnt_);
  cudaFree(src_dsp_);

  //cudaDeviceSynchronize(); printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

void DeviceSynchronizeCUDA() {
  cudaDeviceSynchronize();
}

template void LaplaceBatchedCUDA<2,double>(std::vector<double>& U, const std::vector<double>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                     const std::vector<double>& F, const std::vector<double>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                     const std::vector<std::pair<long,long>>& trg_src_lst);

template void LaplaceBatchedCUDA<3,double>(std::vector<double>& U, const std::vector<double>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                     const std::vector<double>& F, const std::vector<double>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                     const std::vector<std::pair<long,long>>& trg_src_lst);
