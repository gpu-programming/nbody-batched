#include <sctl.hpp>
#include <cassert>
#include <numeric>
#include <execution>

#include <laplace-cuda.h>

using namespace sctl;

/**
 * Compute batched kernel evaluations.
 *
 * @param[out] U vector containing the final target potentials.
 *
 * @param[in] Xt vector containing the target particle locations in array-of-struct (AoS) order {x1,y1,z1, ...
 * ,xn,yn,zn}.
 *
 * @param[in] trg_cnt vector containing the number of target particles in each leaf-node of the tree.
 *
 * @param[in] trg_dsp vector containing the offset of target particles in each leaf-node of the tree.
 *
 * @param[in] F vector containing the densities at each source particle.
 *
 * @param[in] Xs vector containing the source particle locations in array-of-struct (AoS) order.
 *
 * @param[in] src_cnt vector containing the number of source particles in each leaf-node of the tree.
 *
 * @param[in] src_dsp vector containing the offset of source particles in each leaf-node of the tree.
 *
 * @param[in] trg_src_lst list of pairs of target and source leaf-node indices in the interaction list. Assume sorted by
 * targets.
 */
template <int DIM, class Real> void LaplaceBatchedReference(std::vector<Real>& U, const std::vector<Real>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                                      const std::vector<Real>& F, const std::vector<Real>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                                      const std::vector<std::pair<long,long>>& trg_src_lst) {
  assert((long)U.size() == trg_dsp[trg_dsp.size()-1]+trg_cnt[trg_cnt.size()-1]);
  for (auto& u : U) u = 0;

  for (const auto& trg_src : trg_src_lst) { // loop over the interaction list
    const long trg_node = trg_src.first;  // index of target node
    const long src_node = trg_src.second; // index of source node
    const long trg_offset = trg_dsp[trg_node]; // offset for target particles
    const long src_offset = src_dsp[src_node]; // offset for source particles
    const long Nt = trg_cnt[trg_node]; // number of target particles
    const long Ns = src_cnt[src_node]; // number of source particles

    for (long t = 0; t < Nt; t++) { // loop over particles in the target node
      for (long s = 0; s < Ns; s++) { // loop over particles in the source node
        Real r2 = 0;
        for (long k = 0; k < DIM; k++) {
          Real dx = Xt[(trg_offset+t)*DIM+k] - Xs[(src_offset+s)*DIM+k];
          r2 += dx * dx;
        }
        if (r2 > 0) {
          if (DIM == 2) {
            U[trg_offset+t] += F[src_offset+s] * log(r2) * 0.5;
          } else if (DIM == 3) {
            U[trg_offset+t] += F[src_offset+s] / sqrt(r2);
          }
        }
      }
    }

    Profile::Add_FLOP(Ns*Nt*(3*DIM+3));

  }
}

template <Integer digits, class Real, Integer N> Vec<Real,N> approx_log(const Vec<Real,N>& x) { // TODO: vectorize
  Vec<Real,N> logx = Vec<Real,N>::Zero();
  for (Integer i = 0; i < N; i++) logx.insert(i, (x[i] > 0 ? sctl::log<Real>(x[i]) : 0));
  return logx;
}

// Optimized CPU implementation
template <int DIM, class Real, int digits=-1, int VecLen=DefaultVecLen<Real>()> void LaplaceBatchedCPU(std::vector<Real>& U, const std::vector<Real>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                                      const std::vector<Real>& F, const std::vector<Real>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                                      const std::vector<std::pair<long,long>>& trg_src_lst) {
  using Vec = Vec<Real,VecLen>;
  assert(trg_dsp.size() && trg_cnt.size());
  assert((long)U.size() == *(trg_dsp.end()-1) + *(trg_cnt.end()-1));
  #pragma omp parallel for schedule(static)
  for (auto& u : U) u = 0;

  const Long Ninterac = trg_src_lst.size();
  std::vector<long> cost(Ninterac), cumulative_cost(Ninterac);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < Ninterac; i++) cost[i] = trg_cnt[trg_src_lst[i].first] * src_cnt[trg_src_lst[i].second];
  std::exclusive_scan(std::execution::par, cost.begin(), cost.end(), cumulative_cost.begin(), (long)0);
  const long total_cost = *(cumulative_cost.end()-1) + *(cost.end()-1);

  const long Nt = trg_cnt.size();
  std::vector<long> trg_cnt_(Nt), trg_dsp_(Nt); // padding for vectorization
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < Nt; i++) trg_cnt_[i] = (trg_cnt[i]+VecLen-1) & ~(VecLen-1);
  std::exclusive_scan(std::execution::par, trg_cnt_.begin(), trg_cnt_.end(), trg_dsp_.begin(), (long)0);
  const long Ntrg = *(trg_cnt_.end()-1) + *(trg_dsp_.end()-1);

  Vector<Real> Xt_(Ntrg * DIM), U_(Ntrg); // padded for vectorization
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < Nt; i++) { // Set Xt_, U_
    for (long j = 0; j < trg_cnt[i]; j++) {
      U_[trg_dsp_[i]+j] = 0;
      for (long k = 0; k < DIM; k++) {
        Xt_[trg_dsp_[i]*DIM + k*trg_cnt_[i]+j] = Xt[(trg_dsp[i]+j)*DIM+k];
      }
    }
  }

  #pragma omp parallel
  {
    const long p = omp_get_thread_num();
    const Long np = omp_get_max_threads();

    // Partition interaction list to minimize load imbalance
    long interac_start = std::lower_bound(cumulative_cost.begin(), cumulative_cost.end(), total_cost*(p+0)/np) - cumulative_cost.begin();
    long interac_end   = std::lower_bound(cumulative_cost.begin(), cumulative_cost.end(), total_cost*(p+1)/np) - cumulative_cost.begin();

    // Shift interac_start and interac_end to target-node boundaries
    if (interac_start > 0) while (interac_start < Ninterac && trg_src_lst[interac_start-1].first == trg_src_lst[interac_start].first) interac_start++;
    if (interac_end   > 0) while (interac_end   < Ninterac && trg_src_lst[interac_end  -1].first == trg_src_lst[interac_end  ].first) interac_end++;

    for (long interac_idx = interac_start; interac_idx < interac_end; interac_idx++) {
      const auto& trg_src = trg_src_lst[interac_idx];
      const long trg_node = trg_src.first;
      const long src_node = trg_src.second;
      const long trg_offset = trg_dsp_[trg_node];
      const long src_offset = src_dsp[src_node];
      const long Nt = trg_cnt_[trg_node];
      const long Ns = src_cnt[src_node];

      for (long t = 0; t < Nt; t+=VecLen) {
        Vec Xt[DIM];
        for (Long k = 0; k < DIM; k++) Xt[k] = Vec::LoadAligned(&*Xt_.begin() + trg_offset*DIM + k*Nt + t);
        Vec U = Vec::LoadAligned(&*U_.begin() + trg_offset + t);
        for (long s = 0; s < Ns; s++) {
          Vec r2 = Vec::Zero();
          for (long k = 0; k < DIM; k++) {
            const Vec dx = Xt[k] - Xs[(src_offset+s)*DIM+k];
            r2 += dx * dx;
          }
          //if (r2 > 0) {
            if (DIM == 2) {
              U += F[src_offset+s] * approx_log<digits>(r2) * 0.5;
            } else if (DIM == 3) {
              U += F[src_offset+s] * approx_rsqrt<digits>(r2, r2 > Vec::Zero());
            }
          //}
        }
        U.StoreAligned(&*U_.begin() + trg_offset + t);
      }

      Profile::Add_FLOP(Ns*Nt*(3*DIM+3));
    }
  }

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < Nt; i++) {
    for (long j = 0; j < trg_cnt[i]; j++) {
      U[trg_dsp[i]+j] = U_[trg_dsp_[i]+j];
    }
  }
}

template <int DIM, class Real> void LaplaceBatchedGPU(std::vector<Real>& U, const std::vector<Real>& Xt, const std::vector<long>& trg_cnt, const std::vector<long>& trg_dsp,
                                                const std::vector<Real>& F, const std::vector<Real>& Xs, const std::vector<long>& src_cnt, const std::vector<long>& src_dsp,
                                                const std::vector<std::pair<long,long>>& trg_src_lst);

/**
 * Compute N-body near interactions.
 *
 * @param[in] N number of particles.
 *
 * @param[in] M maximum particles per leaf box.
 */
template <class Real, int DIM> void nbody_near(const long N_, const long M, const Comm& comm) {
  const long N = N_ / comm.Size();
  srand48(comm.Rank());

  PtTree<Real,DIM> tree(comm);
  { // Init tree
    Vector<Real> X(N*DIM), f(N);
    for (long i = 0; i < N; i++) { // Set coordinates (X), and values (f)
      f[i] = 0;
      for (int k = 0; k < DIM; k++) {
        X[i*DIM+k] = (pow<3>(drand48()*2-1.0)*0.5+0.5)*0.5;
        f[i] = drand48() - 0.5;
      }
    }

    tree.AddParticles("pt", X);
    tree.AddParticleData("density", "pt", f);
    tree.UpdateRefinement(X, M, true, false);
    tree.template Broadcast<Real>("pt");
    tree.template Broadcast<Real>("density");
  }

  std::vector<Real> F, X;
  std::vector<long> pt_cnt, pt_dsp;
  { // Set F, X, pt_cnt, pt_dsp
    Vector<Real> F_, X_;
    Vector<Long> cnt;

    tree.GetData(X_, cnt, "pt");
    for (const auto& a : X_) X.push_back(a);

    tree.GetData(F_, cnt, "density");
    pt_dsp.push_back(0);
    for (long i = 1; i < cnt.Dim(); i++) pt_dsp.push_back(pt_dsp[i-1] + cnt[i-1]);
    for (const auto& a : cnt) pt_cnt.push_back(a);
    for (const auto& a : F_) F.push_back(a);
  }

  std::vector<std::pair<long,long>> trg_src_lst;
  { // determine s2t interaction list
    const auto& node_mid = tree.GetNodeMID();
    const auto& node_attr = tree.GetNodeAttr();
    const auto& node_lst = tree.GetNodeLists();
    const long Nnds = node_mid.Dim();
    SCTL_ASSERT(node_lst.Dim() == Nnds);
    SCTL_ASSERT(node_attr.Dim() == Nnds);

    for (long t = 0; t < Nnds; t++) {
      if (node_attr[t].Leaf && !node_attr[t].Ghost) {
        constexpr int Nnbr = pow<DIM,int>(3);
        constexpr int Nchild = pow<DIM,int>(2);
        const long interac_start = trg_src_lst.size();
        const auto& trg_lst = node_lst[t];

        const auto is_nbr_coarse = [](const int p2n, const int parent_nbr) { // find neighbors at one level coarser
          int max_dist = 0;
          for (int k = 0; k < DIM; k++) {
            const int x0 = ((p2n >>           k ) % 2) * 2 - 1; // coordinates relative to parent
            const int x1 = ((parent_nbr/pow(3,k)) % 3) * 4 - 4; // parent's neighbor's coordinates
            max_dist = std::max<int>(max_dist, abs(x1-x0));
          }
          return (max_dist == 3);
        };
        const auto is_nbr_fine = [](const int nbr_idx, const int nbr_child_idx) { // find neighbors at one level finer
          int max_dist = 0;
          for (int k = 0; k < DIM; k++) {
            const int x0 = ((nbr_idx/pow(3,k)) % 3) * 4 - 4; // neighbor's coordinates
            const int x1 = ((nbr_child_idx>>k) % 2) * 2 - 1; // child's coordinates
            max_dist = std::max<int>(max_dist, abs(x1+x0));
          }
          return (max_dist == 3);
        };

        for (int i = 0; i < Nnbr; i++) { // nbrs at same level
          const long s = trg_lst.nbr[i];
          if (s >= 0 && node_attr[s].Leaf) {
            if (pt_cnt[t] && pt_cnt[s]) trg_src_lst.push_back(std::make_pair(t,s));
          }
        }

        const long p = trg_lst.parent;
        if (p >= 0) { // nbrs at one level coarser
          for (long i = 0; i < Nnbr; i++) {
            const long s = node_lst[p].nbr[i];
            if (s >= 0 && node_attr[s].Leaf && is_nbr_coarse(trg_lst.p2n, i)) {
              if (pt_cnt[t] && pt_cnt[s]) trg_src_lst.push_back(std::make_pair(t,s));
            }
          }
        }

        for (long i = 0; i < Nnbr; i++) { // nbrs at one level finer
          const long nbr = trg_lst.nbr[i];
          if (nbr >= 0 && !node_attr[nbr].Leaf) {
            for (long j = 0; j < Nchild; j++) {
              const long s = node_lst[nbr].child[j];
              if (s >= 0 && node_attr[s].Leaf && is_nbr_fine(i,j)) {
                if (pt_cnt[t] && pt_cnt[s]) trg_src_lst.push_back(std::make_pair(t,s));
              }
            }
          }
        }

        std::sort(trg_src_lst.begin()+interac_start, trg_src_lst.end(), [](const std::pair<long,long>& a, const std::pair<long,long>& b){return a.second < b.second;});
      }
    }
  }

  const auto print_error = [&comm](const std::vector<Real>& U0, const std::vector<Real>& U) {
    StaticArray<Real,2> global_max, local_max{0,0};
    for (const auto& x : Vector<Real>(U0)-Vector<Real>(U)) local_max[0] = std::max<Real>(local_max[0], fabs(x));
    for (const auto& x : U0) local_max[1] = std::max<Real>(local_max[1], fabs(x));
    comm.Allreduce(local_max+0, global_max+0, 2, Comm::CommOp::MAX);
    if (!comm.Rank()) std::cout<<"Relative error = "<<global_max[0]/global_max[1]<<'\n';
  };
  const long Nt = omp_par::reduce(pt_cnt.begin(), pt_cnt.size());
  std::vector<Real> U0(Nt), U(Nt);

  for (auto& a : U0) a = 0;
  Profile::Tic("CPU", &comm, false);
  //LaplaceBatchedReference<DIM>(U0, X, pt_cnt, pt_dsp, F, X, pt_cnt, pt_dsp, trg_src_lst);
  LaplaceBatchedCPU<DIM>(U0, X, pt_cnt, pt_dsp, F, X, pt_cnt, pt_dsp, trg_src_lst);
  Profile::Toc();

  #ifdef HAVE_CUDA
  for (auto& a : U) a = 0;
  DeviceSynchronizeCUDA();
  Profile::Tic("GPU (CUDA)", &comm, false);
  LaplaceBatchedCUDA<DIM>(U, X, pt_cnt, pt_dsp, F, X, pt_cnt, pt_dsp, trg_src_lst);
  Profile::Toc();
  print_error(U0, U);
  #endif

  // Add other GPU implementation below:

  //for (auto& a : U) a = 0;
  //Profile::Tic("GPU", &comm, false);
  //LaplaceBatchedGPU<DIM>(U, X, pt_cnt, pt_dsp, F, X, pt_cnt, pt_dsp, trg_src_lst);
  //Profile::Toc();
  //print_error(U0, U);

  if (0) { // Generate visualization
    tree.AddParticleData("potential", "pt", Vector<Real>());
    tree.AddParticleData("error", "pt", Vector<Real>());
    tree.DeleteData("potential");
    tree.DeleteData("error");
    tree.AddData("potential", Vector<Real>(U0), Vector<Long>(pt_cnt));
    tree.AddData("error", Vector<Real>(U0)-Vector<Real>(U), Vector<Long>(pt_cnt));

    tree.WriteParticleVTK("density", "density", false);
    tree.WriteParticleVTK("potential", "potential", false);
    tree.WriteParticleVTK("error", "error", false);
    tree.WriteTreeVTK("tree", false);
  }
}

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  Profile::Enable(true);

  {
    const Comm& comm = Comm::World();

    Profile::Tic("2D (N=1e7, M=1e3)", &comm, true);
    nbody_near<double,2>(1000000, 1000, comm);
    Profile::Toc();

    Profile::Tic("2D (N=1e7, M=1e4)", &comm, true);
    nbody_near<double,2>(1000000, 10000, comm);
    Profile::Toc();

    Profile::Tic("3D (N=1e7, M=1e3)", &comm, true);
    nbody_near<double,3>(1000000, 1000, comm);
    Profile::Toc();

    Profile::Tic("3D (N=1e7, M=1e4)", &comm, true);
    nbody_near<double,3>(1000000, 10000, comm);
    Profile::Toc();

    Profile::print(&comm);
  }

  Comm::MPI_Finalize();
  return 0;
}

