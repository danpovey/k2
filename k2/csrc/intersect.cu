/**
 * @brief
 * intersect
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <limits>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

namespace intersect_internal {

  struct StateInfo {
    // the state_idx01 in b_fsas_.
    int32_t b_fsas_state_idx01;
    // the state_idx01 in a_fsas_.
    int32_t a_fsas_state_idx01;
  };

  struct ArcInfo {
    int32_t src_ostate;  // source state-index in states_/arcs_
    int32_t dest_ostate;  // dest state-index in states_/arcs_
    int32_t a_arc_idx012;  // The idx012 of the source arc in a_fsas_.
    int32_t b_arc_idx012;  // The idx012 of the source arc in b_fsas_.
    // Note: other fields, e.g. the label and score, can be worked
    // out from the arc-indexes.
  };



/*
static std::ostream &operator<<(std::ostream &os, const StateInfo &s) {
  os << "StateInfo{" << s.a_fsas_state_idx01 << ","
     << OrderedIntToFloat(s.forward_loglike) << "," << s.backward_loglike
     << "}";
  return os;
}

static std::ostream &operator<<(std::ostream &os, const ArcInfo &a) {
  os << "ArcInfo{" << a.a_fsas_arc_idx012 << "," << a.arc_loglike << ","
     << a.u.dest_a_fsas_state_idx01 << "," << a.end_loglike << "}";
  return os;
}
*/

}  // namespace intersect_internal

using namespace intersect_internal;  // NOLINT

// Caution: this is really a .cu file.  It contains mixed host and device code.

/*
   Intersection (a.k.a. composition) that corresponds to decoding for
   speech recognition-type tasks.  This version does only forward-backward
   pruning in the backward pass; the forward pass does no pruning.

   Can use either different decoding graphs (one per acoustic sequence) or a
   shared graph.

   How to use this object:
       Construct it
       Call Intersect()
       Call FormatOutput()
*/
class DeviceIntersector {
 public:
  /**
     This object does intersection on device (the general case, but without treating
     epsilons specially)

       @param [in] a_fsas  An FsaVec (3 axes), must be valid.  Caution: in future,
                           we may require that it be arc-sorted.
       @param [in] b_fsas  An FsaVec (3 axes), must be valid.
       @param [in] b_to_a_map  Map from fsa-index in b_fsas to the index of the FSA
                           in a_fsas that we want to intersect it with.

     Does not fully check its args (see wrapping code).  After constructing this object,
     call Intersect() and then FormatOutput().
   */
  DeviceIntersector(FsaVec &a_fsas, FsaVec &b_fsas,
                    Array1<int32_t> &b_to_a_map):
      c_(a_fsas.Context()),
      a_fsas_(a_fsas),
      b_fsas_(b_fsas),
      b_to_a_map_(b_to_a_map),
      b_state_bits_(HighestBitSet(b_fsas_.TotSize(1))),
      key_bits_(b_state_bits_ + HighestBitSet(a_fsas_.MaxSize(1))) {

    // We may want to tune this hash size eventually.
    // Note: the hash size
    int32_t hash_size = 4 * RoundUpToNearestPowerOfTwo(b_fsas.NumElements()),
        min_hash_size = 1 << 16;
    if (hash_size < min_hash_size)
      hash_size = min_hash_size;
    hash_pair_to_state_ = Hash(c_, hash_size);


    NVTX_RANGE(K2_FUNC);
    c_ = GetContext(a_fsas.shape, b_fsas.shape);

    K2_CHECK_EQ(a_fsas_.Dim0(), b_fsas_.shape.Dim0());
    num_fsas_ = a_fsas_.Dim0();
    K2_CHECK_GT(num_fsas_, 0);
    K2_CHECK_GT(output_beam, 0);

    {
      Array1<int32_t> dest_states = GetDestStates(a_fsas_, true);
      incoming_arcs_ = GetIncomingArcs(a_fsas_, dest_states);
    }

    // Set up carcs_
    InitCompressedArcs();

    RaggedShape combined_shape;
    {
      int32_t axis = 1, num_srcs = 2;
      RaggedShape *vec[2] = {&a_fsas_.shape, &incoming_arcs_.shape};
      combined_shape = Append(axis, num_srcs, vec);
    }

    int32_t num_arcs = a_fsas_.NumElements();
    // arc_scores_ will be used for backward and forward computations
    // simultaneously.
    arc_scores_ =
        Ragged<float>(combined_shape, Array1<float>(c_, num_arcs * 2));

    // set up fsa_info_
    InitFsaInfo();

    int32_t num_seqs = b_fsas.shape.Dim0();

    {  // check that b_fsas are in order of decreasing length.
      Array1<int32_t> r = b_fsas.shape.RowSplits(1).To(GetCpuContext());
      int32_t *r_data = r.Data();
      int32_t prev_t = r_data[1] - r_data[0];
      for (int32_t i = 1; i + 1 < r.Dim(); i++) {
        int32_t this_t = r_data[i + 1] - r_data[i];
        if (this_t > prev_t)
          K2_LOG(FATAL) << "Sequences (DenseFsaVec) must be in sorted "
                           "order from greatest to least length.";
        prev_t = this_t;
      }
      T_ = r_data[1] - r_data[0];  // longest first, so T_ is the length of the
                                   // longest sequence.
    }

    // set up steps_, which contains a bunch of meta-information about the steps
    // of the algorithm.
    InitSteps();

    int32_t num_states = a_fsas_.TotSize(1);
    // this is the largest array size we'll be dealing with.
    size_t product = ((size_t)(T_ + 1) * (size_t)num_states);
    K2_CHECK_EQ((1 + product), (size_t)(int32_t)(1 + product))
        << "Problem size is too large for this algorithm; try reducing "
           "minibatch size.";
  }

  /* Does the main work of intersection/composition, but doesn't produce any
     output; the output is provided when you call FormatOutput(). */
  void Intersect() {
    NVTX_RANGE(K2_FUNC);

    FirstIter();
    Forward();
  }
  /*
    Does pruning and returns a ragged array indexed [fsa][state][arc],
    containing the result of intersection.

         @param [out] arc_map_a_out  If non-NULL, the map from (arc-index of
                                     returned FsaVec) to (arc-index in a_fsas_)
                                     will be written to here.
         @param [out] arc_map_b_out  If non-NULL, the map from (arc-index of
                                     FsaVec) to (offset into
                                     b_fsas_.scores.Data()) will be written to
                                     here.
         @return  Returns a FsaVec that is the composed result.  Note: due to
                  roundoff, it may possibly contain states and/or arcs that are
                  not accessible or not co-accessible.  It will be top-sorted,
                  and deterministic and arc-sorted if the input a_fsas_ had
                  those properties.
   */
  FsaVec FormatOutput(Array1<int32_t> *arc_map_a_out,
                      Array1<int32_t> *arc_map_b_out) {
    NVTX_RANGE(K2_FUNC);
    Array1<float> score_cutoffs = GetScoreCutoffs();
    float *score_cutoffs_data = score_cutoffs.Data();
    int32_t num_states = a_fsas_.TotSize(1);
    int32_t product = ((size_t)(T_ + 1) * (size_t)num_states);

    // We'll do exclusive-sum on the following array, after setting its elements
    // to 1 if the corresponding state was not pruned away.  The order of
    // 'counts' is: (T+1) copies of all the states of fsa index 0, (T+1) copies
    // of all the states of FSA index 1, and so on.  In fact not all FSAs have
    // this many frames, most of them have fewer copies, but using this regular
    // structure avoids having to compute any extra row_ids vectors and the
    // like.  The out-of-range elements will be set to zero.

    Renumbering renumber_states(c_, product);
    char *keep_state_data = renumber_states.Keep().Data();

    int32_t T = T_;
    const int32_t *a_fsas_row_ids1_data = a_fsas_.RowIds(1).Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();
    float **state_scores_data = state_scores_.Data();

    // the following lambda will set elements within `keep_state_data` to 0
    // or 1.
    K2_EVAL(
        c_, product, lambda_set_keep, (int32_t i)->void {
          // i is actually an idx012 of a state.

          // the following works because each FSA has (its num-states * T_+1)
          // states allocated to it.  However (i / (T_+1)) does not directly map
          // to a state index.
          int32_t fsa_idx0 = a_fsas_row_ids1_data[(i / (T + 1))];
          FsaInfo fsa_info = fsa_info_data[fsa_idx0];
          float cutoff = score_cutoffs_data[fsa_idx0];

          int32_t idx_within_fsa = i - (T + 1) * fsa_info.state_offset,
                  t = idx_within_fsa / fsa_info.num_states,
                  state_idx1 = idx_within_fsa % fsa_info.num_states;
          // In the state_scores arrays, there are 2 copies of each FSA's
          // states, for backward and forward.
          int32_t backward_state_idx = (2 * fsa_info.state_offset) + state_idx1,
                  forward_state_idx = backward_state_idx + fsa_info.num_states;

          char keep = 0;
          if (t <= fsa_info.T) {
            // This time is within the bounds for this FSA..
            float forward_score = state_scores_data[t][forward_state_idx],
                  backward_score =
                      state_scores_data[fsa_info.T - t][backward_state_idx];

            if (forward_score + backward_score > cutoff) keep = 1;
          }
          keep_state_data[i] = keep;
        });

    Array1<int32_t> &new2old = renumber_states.New2Old();
    const int32_t *new2old_data = new2old.Data();
    int32_t ans_tot_num_states = new2old.Dim();

    // t_per_fsa will be set below to the number of time-steps that each FSA has
    // states active on; if each FSA i has scores for 0 <= t < T_i, then
    // t_per_fsa[i] will be T_i + 1, because there is also a copy of the state
    // at time T_i.
    Array1<int32_t> t_per_fsa(c_, num_fsas_ + 1);
    int32_t *t_per_fsa_data = t_per_fsa.Data();

    K2_EVAL(
        c_, num_fsas_, lambda_set_t_per_fsa_etc,
        (int32_t i)->void { t_per_fsa_data[i] = fsa_info_data[i].T + 1; });

    ExclusiveSum(t_per_fsa, &t_per_fsa);

    // now t_per_fsa is the row_splits1 of the shape indexed we'll be returning.
    // It allocates fsa_info_data[i].T + 1 time-indexes to the i'th fsa.
    Array1<int32_t> &ans_row_splits1 = t_per_fsa;
    const int32_t *ans_row_splits1_data = ans_row_splits1.Data();
    Array1<int32_t> ans_row_ids1(c_, t_per_fsa.Back());
    RowSplitsToRowIds(ans_row_splits1, &ans_row_ids1);

    // ans_row_ids2 maps to an ans_idx01 that combines FSA-index and time-index.
    Array1<int32_t> ans_row_ids2(c_, ans_tot_num_states);
    int32_t *ans_row_ids2_data = ans_row_ids2.Data();
    // ans_num_arcs is the number of arcs potentially active for a state; we'll
    // prune out the invalid ones later on.
    Array1<int32_t> ans_num_arcs(c_, ans_tot_num_states + 1);
    int32_t *ans_num_arcs_data = ans_num_arcs.Data();

    // ans_state_idx01 contains the state_idx01 in a_fsas_ for each state in
    // the answer.
    Array1<int32_t> ans_state_idx01(c_, ans_tot_num_states);
    int32_t *ans_state_idx01_data = ans_state_idx01.Data();
    const int32_t *a_fsas_row_splits2_data = a_fsas_.RowSplits(2).Data();

    // set ans_row_ids2_data, which contains an ans_idx01 that combines
    // FSA-index and time-index.
    K2_EVAL(
        c_, ans_tot_num_states, lambda_set_row_ids2,
        (int32_t ans_idx012)->void {
          // old_i is the same as the index `i` into lambda_set_keep.  It is
          // also an idx012. The logic is the same as for lambda_set_keep, we
          // keep the code but not the comments.
          int32_t old_i = new2old_data[ans_idx012];
          int32_t fsa_idx0 = a_fsas_row_ids1_data[(old_i / (T + 1))];
          FsaInfo fsa_info = fsa_info_data[fsa_idx0];
          int32_t idx_within_fsa = old_i - (T + 1) * fsa_info.state_offset,
                  t = idx_within_fsa / fsa_info.num_states,
                  a_fsas_state_idx1 = idx_within_fsa % fsa_info.num_states;
          int32_t a_fsas_state_idx01 =
              fsa_info.state_offset + a_fsas_state_idx1;
          int32_t ans_fsa_idx0x = ans_row_splits1_data[fsa_idx0],
                  ans_idx01 = ans_fsa_idx0x + t;
          ans_row_ids2_data[ans_idx012] = ans_idx01;
          ans_state_idx01_data[ans_idx012] = a_fsas_state_idx01;
          // note: fsa_info.state_offset ==
          // a_fsas_row_splits2_data[a_fsas_state_idx01];
          int32_t num_arcs = a_fsas_row_splits2_data[a_fsas_state_idx01 + 1] -
                             a_fsas_row_splits2_data[a_fsas_state_idx01];
          if (t == fsa_info.T)  // No arcs leave copies of states on the last
                                // frame for each FSA.
            num_arcs = 0;
          ans_num_arcs_data[ans_idx012] = num_arcs;
        });

    Array1<int32_t> &ans_row_splits3(ans_num_arcs);
    ExclusiveSum(ans_num_arcs, &ans_row_splits3);
    int32_t tot_arcs = ans_row_splits3.Back();
    Array1<int32_t> ans_row_ids3(c_, tot_arcs);
    RowSplitsToRowIds(ans_row_splits3, &ans_row_ids3);

    // Actually we'll do one more pass of pruning on 'ans' before we return it.
    Ragged<Arc> ans(RaggedShape4(&ans_row_splits1, &ans_row_ids1, -1, nullptr,
                                 &ans_row_ids2, ans_tot_num_states,
                                 &ans_row_splits3, &ans_row_ids3, -1),
                    Array1<Arc>(c_, tot_arcs));
    Arc *ans_arcs_data = ans.values.Data();

    Array1<int32_t> arc_map_a(c_, tot_arcs), arc_map_b(c_, tot_arcs);
    int32_t *arc_map_a_data = arc_map_a.Data(),
            *arc_map_b_data = arc_map_b.Data();

    Renumbering renumber_arcs(c_, tot_arcs);
    char *keep_arc_data = renumber_arcs.Keep().Data();

    const int32_t *ans_row_ids1_data = ans_row_ids1.Data(),
                  *ans_row_ids3_data = ans_row_ids3.Data(),
                  *ans_row_splits2_data = ans.shape.RowSplits(2).Data(),
                  *ans_row_splits3_data = ans_row_splits3.Data(),
                  *states_old2new_data = renumber_states.Old2New().Data();
    CompressedArc *carcs_data = carcs_.Data();
    int32_t scores_stride = b_fsas_.scores.ElemStride0();
    const float *scores_data = b_fsas_.scores.Data();

    K2_EVAL(
        c_, tot_arcs, lambda_set_arcs_and_keep, (int32_t arc_idx0123)->void {
          int32_t ans_state_idx012 = ans_row_ids3_data[arc_idx0123],
                  ans_idx012x = ans_row_splits3_data[ans_state_idx012],
                  ans_idx01 = ans_row_ids2_data[ans_state_idx012],
                  fsa_idx0 = ans_row_ids1_data[ans_idx01],
                  ans_idx0x = ans_row_splits1_data[fsa_idx0],
                  ans_idx0xx = ans_row_splits2_data[ans_idx0x],
                  t_idx1 = ans_idx01 - ans_idx0x,
                  arc_idx3 = arc_idx0123 - ans_idx012x;
          int32_t a_fsas_state_idx01 = ans_state_idx01_data[ans_state_idx012];
          FsaInfo fsa_info = fsa_info_data[fsa_idx0];
          float cutoff = score_cutoffs_data[fsa_idx0];
          int32_t a_fsas_state_idx0x = fsa_info.state_offset,
                  a_fsas_state_idx1 = a_fsas_state_idx01 - a_fsas_state_idx0x;
          int32_t a_fsas_arc_idx012 =
              a_fsas_row_splits2_data[a_fsas_state_idx01] +
              arc_idx3;  //  arc_idx3 is an idx2 w.r.t. a_fsas.
          CompressedArc carc = carcs_data[a_fsas_arc_idx012];
          K2_CHECK_EQ(a_fsas_state_idx1, (int32_t)carc.src_state);
          int32_t a_fsas_dest_state_idx1 = carc.dest_state;
          arc_map_a_data[arc_idx0123] = a_fsas_arc_idx012;
          int32_t scores_index = fsa_info.scores_offset +
                                 (scores_stride * t_idx1) + carc.label_plus_one;
          arc_map_b_data[arc_idx0123] = scores_index;

          float arc_score = carc.score + scores_data[scores_index];

          // unpruned_src_state_idx and unpruned_dest_state_idx are into
          // `renumber_states.Keep()` or `renumber_states.Old2New()`
          int32_t unpruned_src_state_idx = fsa_info.state_offset * (T + 1) +
                                           (t_idx1 * fsa_info.num_states) +
                                           a_fsas_state_idx1,
                  unpruned_dest_state_idx =
                      fsa_info.state_offset * (T + 1) +
                      ((t_idx1 + 1) * fsa_info.num_states) +
                      a_fsas_dest_state_idx1;
          K2_CHECK_EQ(states_old2new_data[unpruned_src_state_idx],
                      ans_state_idx012);
          K2_CHECK_LT(t_idx1, (int32_t)fsa_info.T);

          int32_t ans_dest_state_idx012 =
                      states_old2new_data[unpruned_dest_state_idx],
                  ans_dest_state_idx012_next =
                      states_old2new_data[unpruned_dest_state_idx + 1];
          char keep_this_arc = 0;

          const float *forward_state_scores = state_scores_data[t_idx1];
          // 'next_backward_state_scores' is the state_scores vector for the
          // next frame (t_idx1 + 1); the backward scores are in the opposite
          // order so we index as ((int32_t)fsa_info.T) - (t_idx1 + 1).
          const float *next_backward_state_scores =
              state_scores_data[((int32_t)fsa_info.T) - (t_idx1 + 1)];

          if (ans_dest_state_idx012 < ans_dest_state_idx012_next) {
            // The dest-state of this arc has a number (was not pruned away).
            // below, backward_dest_state_idx and forward_src_state_idx are into
            // the state_scores arrays.
            int32_t backward_dest_state_idx =
                        (2 * fsa_info.state_offset) + a_fsas_dest_state_idx1,
                    forward_src_state_idx = (2 * fsa_info.state_offset) +
                                            fsa_info.num_states +
                                            a_fsas_state_idx1;
            float arc_forward_backward_score =
                forward_state_scores[forward_src_state_idx] + arc_score +
                next_backward_state_scores[backward_dest_state_idx];
            if (arc_forward_backward_score > cutoff) {
              keep_this_arc = 1;
              Arc arc;
              arc.label = static_cast<int32_t>(carc.label_plus_one) - 1;
              // the idx12 into `ans`, which includes the 't' and 'state'
              // indexes, corresponds to the state-index in the FSA we will
              // return (the 't' index will be removed).
              int32_t src_state_idx12 = ans_state_idx012 - ans_idx0xx,
                      dest_state_idx12 = ans_dest_state_idx012 - ans_idx0xx;
              arc.src_state = src_state_idx12;
              arc.dest_state = dest_state_idx12;
              arc.score = arc_score;
              ans_arcs_data[arc_idx0123] = arc;
            }
          }
          keep_arc_data[arc_idx0123] = keep_this_arc;
        });

    if (arc_map_a_out) *arc_map_a_out = arc_map_a[renumber_arcs.New2Old()];
    if (arc_map_b_out) *arc_map_b_out = arc_map_b[renumber_arcs.New2Old()];
    // subsample the output shape, removing arcs that weren't kept
    RaggedShape ans_shape_subsampled =
        SubsampleRaggedShape(ans.shape, renumber_arcs);
    // .. and remove the 't' axis
    return Ragged<Arc>(RemoveAxis(ans_shape_subsampled, 1),
                       ans.values[renumber_arcs.New2Old()]);
  }

  // We can't actually make the rest private for reasons relating to use of
  // __host__ __device__ lambdas, but logically the rest would be private.

  // private:

  void InitCompressedArcs() {
    NVTX_RANGE(K2_FUNC);
    int32_t tot_arcs = a_fsas_.NumElements();
    carcs_ = Array1<CompressedArc>(c_, tot_arcs);
    CompressedArc *carcs_data = carcs_.Data();
    const Arc *arcs_data = a_fsas_.values.Data();
    const int32_t *a_fsas_row_ids1 = a_fsas_.RowIds(1).Data(),
                  *a_fsas_row_ids2 = a_fsas_.RowIds(2).Data(),
                  *a_fsas_row_splits1 = a_fsas_.RowSplits(1).Data(),
                  *a_fsas_row_splits2 = a_fsas_.RowSplits(2).Data();

    // incoming_indexes maps from position in normally-arranged arcs (i.e.
    // arc_idx012 in a_fsas_) to the position that that arc has in
    // incoming_arcs_.
    Array1<int32_t> incoming_indexes = InvertPermutation(incoming_arcs_.values);
    const int32_t *incoming_indexes_data = incoming_indexes.Data();

    K2_EVAL(
        c_, tot_arcs, set_carcs_lambda, (int32_t i)->void {
          Arc arc = arcs_data[i];
          CompressedArc carc;
          carc.src_state = uint16_t(arc.src_state);
          carc.dest_state = uint16_t(arc.dest_state);
          carc.label_plus_one = uint16_t(arc.label + 1);
          carc.fsa_idx = a_fsas_row_ids1[a_fsas_row_ids2[i]];
          carc.score = arc.score;
          {  // set carc.incoming_arc_idx012
            int32_t
                this_fsa_first_arc =
                    a_fsas_row_splits2[a_fsas_row_splits1[carc.fsa_idx]],
                next_fsa_first_arc =
                    a_fsas_row_splits2[a_fsas_row_splits1[carc.fsa_idx + 1]],
                this_fsa_num_arcs = next_fsa_first_arc - this_fsa_first_arc;
            int32_t incoming_arc_idx012 = incoming_indexes_data[i],
                    incoming_arc_idx12 =
                        incoming_arc_idx012 - this_fsa_first_arc;
            // The arrangement of arcs in arc_scores_ (and Step::arc_scores)
            // comes from appending a_fsas_.shape and incoming_arcs_.shape along
            // axis 1. So (2 * this_fsa_first_arc) is the start of this FSA's
            // data, and adding this_fsa_num_arcs means skipping over the arcs
            // arranged as in a_fsas_.shape.
            carc.incoming_arc_idx012 =
                2 * this_fsa_first_arc + this_fsa_num_arcs + incoming_arc_idx12;
          }
          carcs_data[i] = carc;
        });
  }

  void InitFsaInfo() {
    NVTX_RANGE(K2_FUNC);
    int32_t *b_fsas_row_splits1_data = b_fsas_.shape.RowSplits(1).Data(),
            *a_fsas_row_splits1_data = a_fsas_.shape.RowSplits(1).Data(),
            *a_fsas_row_splits2_data = a_fsas_.shape.RowSplits(2).Data();
    int32_t scores_stride = b_fsas_.scores.ElemStride0();

    fsa_info_ = Array1<FsaInfo>(c_, num_fsas_ + 1);
    FsaInfo *fsa_info_data = fsa_info_.Data();
    int32_t num_fsas = num_fsas_;
    K2_EVAL(
        c_, num_fsas_ + 1, lambda_set_fsa_info, (int32_t i)->void {
          FsaInfo info;
          if (i < num_fsas) {
            info.T = uint16_t(b_fsas_row_splits1_data[i + 1] -
                              b_fsas_row_splits1_data[i]);
            info.num_states = uint16_t(a_fsas_row_splits1_data[i + 1] -
                                       a_fsas_row_splits1_data[i]);
          } else {
            info.T = 0;
            info.num_states = 0;
          }
          info.scores_offset = b_fsas_row_splits1_data[i] * scores_stride;
          info.state_offset = a_fsas_row_splits1_data[i];
          info.arc_offset = a_fsas_row_splits2_data[info.state_offset];
          fsa_info_data[i] = info;
        });
  }

  /*
    InitSteps() sets up steps_; it works out metadata and allocates memory, but
    does not do any of the actual computation.
   */
  void InitSteps() {
    NVTX_RANGE(K2_FUNC);
    // This vector, of length num_fsas_, tells us how many copies of (the states
    // of the i'th decoding graph) we have.  It equals (the length of the
    // sequence of log-likes in b_fsas_) + 1.  It is monotonically decreasing
    // (thanks to how we require the FSAs to be sorted).
    Array1<int32_t> num_copies_per_fsa(c_, num_fsas_);
    const int32_t *b_row_splits_data = b_fsas_.shape.RowSplits(1).Data();
    int32_t *num_copies_per_fsa_data = num_copies_per_fsa.Data();

    K2_EVAL(
        c_, num_fsas_, lambda_set_num_copies, (int32_t i)->void {
          num_copies_per_fsa_data[i] =
              1 + b_row_splits_data[i + 1] - b_row_splits_data[i];
        });

    std::vector<int32_t> range(num_fsas_);
    // fill with 1, 2, .. num_fsas_.
    std::iota(range.begin(), range.end(), 1);
    std::vector<RaggedShape> arc_scores_prefixes =
        GetPrefixes(arc_scores_.shape, range);

    ContextPtr c_cpu = GetCpuContext();

    // This vector, of length T_ + 1, tells us, for each frame 0 <= t <= T, how
    // many FSAs have a copy of their decoding-graph states alive on this
    // time-index.  It equals InvertMonotonicDecreasing(num_copies_per_fsa_)
    // and it is also the case that InvertMonotonicDecreasing(num_fsas_per_t_)
    // == num_copies_per_fsa_.
    Array1<int32_t> num_fsas_per_t =
                        InvertMonotonicDecreasing(num_copies_per_fsa),
                    num_fsas_per_t_cpu = num_fsas_per_t.To(c_cpu);

    Array1<int32_t> a_fsas_row_splits1_cpu = a_fsas_.RowSplits(1).To(c_cpu),
                    a_fsas_row_splits12_cpu =
                        a_fsas_.RowSplits(2)[a_fsas_.RowSplits(1)].To(c_cpu);
    int32_t tot_arcs = a_fsas_.NumElements(), tot_states = a_fsas_.TotSize(1);

    steps_.resize(T_ + 1);

    for (int32_t t = 0; t <= T_; t++) {
      Step &step = steps_[t];
      step.t = t;
      step.num_fsas = num_fsas_per_t_cpu[t];

      // the - 1 is because arc_scores_prefixes contains prefixes
      // of length 1, 2, .. (starts from 1 not 0).
      RaggedShape &shape = arc_scores_prefixes[step.num_fsas - 1];
      step.arc_scores = Ragged<float>(
          shape, arc_scores_.values.Arange(0, shape.NumElements()));

      int32_t num_states = a_fsas_row_splits1_cpu[step.num_fsas];
      // * 2 because have both forward and backward.
      step.state_scores = Array1<float>(c_, 2 * num_states);
    }
  }

  void DoStep0() {
    NVTX_RANGE(K2_FUNC);
    // Run step zero of the computation: this initializes the forward
    // probabilities on frame 0, and the backward probabilities on the last
    // frame for each sequence.

    float *scores = steps_[0].state_scores.Data();
    int32_t *a_fsas_row_ids1 = a_fsas_.RowIds(1).Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();
    const float minus_inf = -std::numeric_limits<float>::infinity();
    int32_t tot_states = a_fsas_.TotSize(1);
    K2_EVAL(
        c_, tot_states, lambda_init_state_scores, (int32_t state_idx01)->void {
          int32_t fsa_idx0 = a_fsas_row_ids1[state_idx01];
          FsaInfo this_info = fsa_info_data[fsa_idx0];
          // we could also write:
          // backward_state_idx = (2 * this_info.state_offset) + state_idx1,
          // where state_idx1 = this_info.state_offset - state_idx01.
          // Each FSA has (its states for backward, its states for forward).
          int32_t backward_state_idx = this_info.state_offset + state_idx01,
                  forward_state_idx = backward_state_idx + this_info.num_states,
                  state_idx1 = state_idx01 - this_info.state_offset;

          float start_loglike = (state_idx1 == 0 ? 0 : minus_inf),
                end_loglike =
                    (state_idx1 + 1 == this_info.num_states ? 0 : minus_inf);
          scores[forward_state_idx] = start_loglike;
          scores[backward_state_idx] = end_loglike;
        });
  }


  void Forward() {
    NVTX_RANGE(__K2_FUNC__);
    for (int32_t t = 0; ; t++) {
      NVTX_RANGE("LoopT");

      K2_CHECK_EQ(t + 2, int32_t(iter_to_state_row_splits_cpu_.size()))

      int32_t state_begin = iter_to_state_row_splits_cpu_[t],
          state_end = iter_to_state_row_splits_cpu_[t + 1],
          num_states = state_end - state_begin;

      if (num_states == 0)
        break;  // Nothing left to process.

      // We need to process output-states numbered state_begin..state_end-1.

      // Row 0 of num_arcs will contain the num_arcs leaving each state
      // in b in this batch; row 1 will contain (num_arcs in a * num_arcs in b).
      // If the total of row 1 is small enough and we're using the device,
      // we'll process all pairs of arcs; otherwise we'll do a logarithmic
      // search.
      Array2<int32_t> num_arcs(c_, 2, num_states + 1);

      auto num_arcs_acc = num_arcs.Accessor();
      StateInfo *states_data = states_.Data();
      const int32_t *a_fsas_row_splits2_data = a_fsas_.RowSplits(2).Data(),
          *b_fsas_row_splits2_data = b_fsas_.RowSplits(2).Data();

      auto lambda_find_num_arcs = [=] __host__ __device__ (int32_t i) -> void {
        int32_t state_idx = state_begin + i;
        StateInfo info = states_data[state_idx];
        int32_t b_fsas_state_idx01 = info.b_fsas_state_idx01,
            b_start_arc = b_fsas_row_splits2_data[b_fsas_state_idx01],
            b_end_arc =  b_fsas_row_splits2_data[b_fsas_state_idx01 + 1],
            b_num_arcs = end_arc - start_arc;
        num_arcs_acc(0, i) = b_num_arcs;
        int32_t a_fsas_state_idx01 = info.b_fsas_state_idx01,
            a_start_arc = a_fsas_row_splits2_data[a_fsas_state_idx01],
            a_end_arc =  a_fsas_row_splits2_data[a_fsas_state_idx01 + 1],
            a_num_arcs = end_arc - start_arc;
        num_arcs_acc(1, i) = b_num_arcs * a_num_arcs;
      };
      K2_EVAL(c_, num_states, lambda_find_num_arcs);

      // tot_ab is total of (num-arcs from state a * num-arcs from state b).
      int32_t tot_ab = num_arcs.Row(0)[num_states],
          cutoff = 1 << 30;  // Eventually I'll make cutoff smaller, like 16384,
                           // and implement the other branch.


      const Arc *a_arcs_data = a_fsas_.values.Data(),
          *b_arcs_data = b_fsas_.values.Data();

      int32_t key_bits = key_bits_, b_state_bits = b_state_bits_,
          value_bits = 64 - key_bits;
      // `value_big` is the largest power of 2 that can fit in the 'value' in
      // our hash table.  We use this to distinguish arc-indexes from state-indexes.
      uint64_t value_big = 1 << (value_bits - 1);
      auto state_pair_to_state_acc =
          state_pair_to_state.GeneralAccessor(key_bits);

      K2_CHECK_GT(value_big, (uint64_t)tot_ab) << "Problem size too large "
          "for hash table... redesign or reduce problem size.";


      if (tot_ab < cutoff) {
        Array1<int32_t> row_splits_ab = num_arcs.Row(1),
            num_arcs_b = num_arcs.Row(0);
        ExclusiveSum(row_splits_ab, &row_splits_ab);
        Array<int32_t> row_ids_ab(c_, tot_ab);
        RowSplitsToRowIds(row_splits_ab, &row_ids_ab);

        const int32_t *row_ids_ab_data = row_ids_ab.Data(),
            *row_splits_ab_data = row_splits_ab.Data(),
            *row_splits_b_data = row_splits_b.Data();


        const int32_t *b_fsas_row_ids1_data = b_fsas_.RowIds(1).Data();

        // arcs_newstates_renumbering serves two purposes:
        //  - we'll keep some subset of the `tot_ab` arcs.
        //  - some subset of the dest-states of those arcs will be "new" dest-states
        //    that need to be assigned a state-id.
        // To avoid sequential kernels for computing Old2New() and computing New2Old(),
        // we combine those two renumberings into one.
        Renumbering arcs_newstates_renumbering(tot_ab * 2);
        char *keep_arc_data = arcs_renumbering.Keep().Data(),
            *new_dest_state_data = keep_arc_data + tot_ab;

        auto lambda_set_keep_arc_newstate = [=] __host__ __device__ (int32_t i) -> void {
          // state_i is the index into the block of ostates that we're
          // processing, the actual state index is state_i + state_begin.
          int32_t state_i = row_ids_ab_data[i],
              // arc_pair_idx encodes a_arc_idx2 and b_arc_idx2
              arc_pair_idx = i - row_splits_ab_data[state_i],
              state_idx = state_i + state_begin;
          StateInfo sinfo = states_data[state_idx];
          int32_t num_arcs_b = num_arcs_b[i],
              a_arc_idx2 = arc_pair_idx / num_arcs_b,
              b_arc_idx2 = arc_pair_idx % num_arcs_b;
          // the idx2's above are w.r.t. a_fsas_ and b_fsas_.
          int32_t a_arc_idx01x = a_fsas_row_ids2[sinfo.a_fsas_state_idx01],
              b_arc_idx01x = b_fsas_row_ids2[sinfo.b_fsas_state_idx01],
              a_arc_idx012 = a_arc_idx01x + a_arc_idx2,
              b_arc_idx012 = b_arc_idx01x + b_arc_idx2;
          // Not treating epsilons specially here, see documentation for
          // IntersectDevice() in [currently] fsa_algo.h.
          int keep_arc = (a_arcs_data[a_arc_idx012].label ==
                          b_arcs_data[b_arc_idx012].label);
          keep_arc_data[i] = (char)keep_arc;
          int new_dest_state = 0;
          if (keep_arc && a_arcs_data[a_arc_idx012].label != -1) {
            // investigate whether the dest-state is new (not currently allocated
            // a state-id).  We don't allocate ids for the final-state, so skip this
            // if label is -1.

            int32_t b_dest_state_idx1 = b_arcs_data[b_arc_idx012].dest_state,
                b_dest_state_idx01 = b_dest_state_idx1 + sinfo.b_fsas_state_idx01 -
                                     b_arcs_data[b_arc_idx012].src_state,
                b_fsa_idx0 = b_fsas_row_ids1_data[b_dest_state_idx01],
                a_dest_state_idx1 = a_arcs_data[a_arc_idx012].dest_state;
            uint64_t hash_key = (((uint64_t)a_dest_state_idx1) << b_state_bits_) +
                b_dest_state_idx01, hash_value = value_big | i;
            // If it was successfully inserted, then this arc is assigned
            // responsibility for creating the state-id for its destination
            // state.
            if (state_pair_to_state_acc.Insert(hash_key, hash_value)) {
              new_dest_state = 1;
            }
          }
          new_dest_state_data[i] = (char)new_dest_state;
        };
        K2_EVAL(c_, tot_ab, lambda_set_keep_arc_newstate);

        // When reading the code below, remember this code is a little unusual
        // because we have combined the renumberings for arcs and new-states
        // into one.
        int32_t num_kept_arcs = arcs_renumbering.Old2New()[tot_ab],
            num_kept_tot = arcs_renumbering.New2Old().Dim(),
            num_kept_states = num_kept_tot -  num_kept_arcs;

        int32_t next_state_end = state_end + num_kept_states;
        iter_to_state_row_splits_cpu_.push_back(next_state_end);
        states_.Resize(next_state_end);  // Note: this Resize() won't actually reallocate each time.
        states_data = states_.Data();  // In case it changed (unlikely)

        Array1<int32_t> states_new2old =
            arcs_renumbering.New2Old().Arange(num_kept_arcs, num_kept_tot);
        const int32_t *states_new2old_data = states_new2old.Data(),
            *b_to_a_map_data = b_to_a_map_.Data(),
            *a_fsas_row_splits1_data = a_fsas_.RowSplits(1).Data();

        // set new elements of `states_data`, setting up the StateInfo on the next
        // frame and setting the state indexes in the hash (to be looked up when
        // creating the arcs.
        auto lambda_set_states_data = [=] __host__ __device__ (int32_t i) -> void {
          // the reason for the "- tot_ab" is that this was in the second half of
          // the array of 'kept' of size tot_ab * 2.
          int32_t arc_i = states_new2old_data[i] - tot_ab;

          // The code below repeats what we did when processing arcs in the
          // previous lambda (now just for a small subset of arcs).

          // src_state_i is the index into the block of ostates that we're
          // processing, the actual state index is state_i + state_begin.
          int32_t src_state_i = row_ids_ab_data[arc_i],
              // arc_pair_idx encodes a_arc_idx2 and b_arc_idx2
              arc_pair_idx = arc_i - row_splits_ab_data[src_state_i],
              src_state_idx = state_i + state_begin;  // state_idx is the source
                                                  // state of this arc, not the
                                                  // state we're creating.
          StateInfo src_sinfo = states_data[src_state_idx];
          int32_t num_arcs_b = num_arcs_b[i],
              a_arc_idx2 = arc_pair_idx / num_arcs_b,
              b_arc_idx2 = arc_pair_idx % num_arcs_b;
          // the idx2's above are w.r.t. a_fsas_ and b_fsas_.
          int32_t a_arc_idx01x = a_fsas_row_ids2[src_sinfo.a_fsas_state_idx01],
              b_arc_idx01x = b_fsas_row_ids2[src_sinfo.b_fsas_state_idx01],
              a_arc_idx012 = a_arc_idx01x + a_arc_idx2,
              b_arc_idx012 = b_arc_idx01x + b_arc_idx2;
          ArcInfo b_arc = b_arcs_data[b_arc_idx012],
              a_arc = a_arcs_data[a_arc_idx012];
          K2_DCHECK_EQ(a_arc.label, b_arc.label);

          int32_t b_dest_state_idx1 = b_arcs_data[b_arc_idx012].dest_state,
                b_dest_state_idx01 = b_dest_state_idx1 + sinfo.b_fsas_state_idx01 -
                                     b_arcs_data[b_arc_idx012].src_state,
                b_fsa_idx0 = b_fsas_row_ids1_data[b_dest_state_idx01],
              a_dest_state_idx1 = a_arcs_data[a_arc_idx012].dest_state,
              a_dest_state_idx01 = a_fsas_row_splits1_data[b_to_a_map_data[b_fsa_idx0]] +
                    a_dest_staste_idx1;
          uint64_t hash_key = (((uint64_t)a_dest_state_idx1) << b_state_bits_) +
              b_dest_state_idx01;
          uint64_t value, *key_value_location;
          bool ans = state_pair_to_state_acc.Find(hash_key, &value,
                                                  &key_value_location);
          K2_CHECK(ans);
          K2_CHECK_EQ(value, value_big | arc_i);
          int32_t dest_state_idx = state_end + i;
          state_pair_to_state_acc.SetValue(key_value_location, hash_key,
                                             (uint64_t)dest_state_idx);

          StateInfo dest_sinfo;
          dest_sinfo.a_fsas_state_idx01 = a_dest_state_idx01;
          dest_sinfo.b_fsas_state_idx01 = b_dest_state_idx01;
          states_data[dest_state_idx] = dest_sinfo;
        };
        K2_EVAL(c_, num_kept_states, lambda_set_states_data);


        int32_t old_num_arcs = arcs_.Dim(),
            new_num_arcs = old_num_arcs + num_kept_arcs;
        arcs_.Resize(new_num_arcs);
        arcs_row_ids2_.Resize(new_num_arcs);
        arcs_data = arcs.Data();
        int32_t *arcs_row_ids2_data = arcs_row_ids2_.Data();

        const int32_t *arcs_new2old_data = arcs_renumbering.New2Old().Data();

        auto lambda_set_arc_info = [=] __host__ __device__ (int32_t new_arc_i) -> void {
          // 0 <= arc_i < num_kept_arcs..
          int32_t arc_idx = old_num_arcs + new_arc_i;
          int32_t old_arc_i = arcs_new2old_data[new_arc_i];

          // The code below repeats what we did when processing arcs in the
          // previous lambdas (we do this for all arcs that were kept).

          // src_state_i is the index into the block of ostates that we're
          // processing, the actual state index is state_i + state_begin.
          int32_t src_state_i = row_ids_ab_data[old_arc_i];
          // arc_pair_idx encodes a_arc_idx2 and b_arc_idx2
          int32_t arc_pair_idx = old_arc_i - row_splits_ab_data[src_state_i],
              src_state_idx = state_i + state_begin;  // state_idx is the source
                                                  // state of this arc, not the
                                                  // state we're creating.
          StateInfo src_sinfo = states_data[src_state_idx];
          int32_t num_arcs_b = num_arcs_b[i],
              a_arc_idx2 = arc_pair_idx / num_arcs_b,
              b_arc_idx2 = arc_pair_idx % num_arcs_b;
          // the idx2's above are w.r.t. a_fsas_ and b_fsas_.
          int32_t a_arc_idx01x = a_fsas_row_ids2[src_sinfo.a_fsas_state_idx01],
              b_arc_idx01x = b_fsas_row_ids2[src_sinfo.b_fsas_state_idx01],
              a_arc_idx012 = a_arc_idx01x + a_arc_idx2,
              b_arc_idx012 = b_arc_idx01x + b_arc_idx2;
          ArcInfo b_arc = b_arcs_data[b_arc_idx012],
              a_arc = a_arcs_data[a_arc_idx012];
          K2_DCHECK_EQ(a_arc.label, b_arc.label);

          int32_t dest_state_idx = -1;
          if (a_arc.label != -1) {
            int32_t b_dest_state_idx1 = b_arcs_data[b_arc_idx012].dest_state,
                b_dest_state_idx01 = b_dest_state_idx1 + sinfo.b_fsas_state_idx01 -
                b_arcs_data[b_arc_idx012].src_state,
                b_fsa_idx0 = b_fsas_row_ids1_data[b_dest_state_idx01],
                a_dest_state_idx1 = a_arcs_data[a_arc_idx012].dest_state,
                a_dest_state_idx01 = a_fsas_row_splits1_data[b_to_a_map_data[b_fsa_idx0]] +
                a_dest_staste_idx1;
            uint64_t hash_key = (((uint64_t)a_dest_state_idx1) << b_state_bits_) +
                b_dest_state_idx01;

            uint64_t value;
            bool ans = state_pair_to_state_acc.Find(hash_key, &value);
            K2_CHECK_EQ(value & value_big, 0);  // make sure it's a state-id not
                                                // an arc-id..
            dest_state_idx = static_cast<uint32_t>(value);
          }  // else leave it at -1, it's a final-state and we allocate their
             // state-ids at the end.

          ArcInfo info;
          info.src_ostate = src_state_idx;
          info.dest_ostate = dest_state_idx;
          info.a_arc_idx012 = a_arc_idx012;
          info.b_arc_idx012 = b_arc_idx012;
          arcs_data[old_num_arcs + new_arc_i] = info;
          arcs_row_ids2_data[old_num_arcs + new_arc_i] = src_state_i;
        };
        K2_EVAL(c_, num_kept_arcs, lambda_set_arc_info);

      } else {
        ExclusiveSum(num_arcs, &num_arcs, 1);  // sum
        // Plan to implement binary search here at some point, to get arc ranges...
        K2_LOG(FATAL) << "Not implemented yet, see code..";
      }
    }
  }


  ContextPtr c_;
  FsaVec a_fsas_;  // a_fsas_: decoding graphs, with same Dim0() as
                    // b_fsas_. Note: a_fsas_ has 3 axes.

  FsaVec b_fsas_;

  // map from fsa-index in b_fsas_ to the fsa-index in a_fsas_ that we want to
  // intersect it with.
  Array1<int32_t> b_to_a_map_;

  // iter_to_state_row_splits_cpu_, which is resized on each iteration of the
  // algorithm, is a row-splits array that maps from iteration index to
  // state_idx (index into states_).
  std::vector<int32_t> iter_to_state_row_splits_cpu_;

  // states_ is a resizable array of StateInfo that conceptually is the elements
  // of a ragged array indexed [iter][state], with row_splits1 ==
  // iter_to_state_row_splits_cpu_.
  Array1<StateInfo> states_;

  // arcs_ is a resizable array of ArcInfo that conceptually is the elements
  // of a ragged array indexed [iter][state][arc], with row_splits1 == iter_to_state_row_splits_cpu_
  // and row_splits2 == state_to_arc_row_splits_.
  Array1<ArcInfo> arcs_;

  // arcs_row_ids2_, which always maintained as having the same size as `arcs_`, maps
  // from the output arc to the corresponding ostate index (index into states_).
  Array1<int32_t> arcs_row_ids2_;



  // The hash maps from state-pair, as:
  //   state_pair = (a_fsas_state_idx1 << b_state_bits_) + b_fsas_state_idx01
  //
  // The number of bits in the key (max bits set in `state_pair`) is
  // key_bits_ == b_state_bits_ + HighestBitSet(a_fsas_.MaxSize(1)).
  // The number of bits in the value is 64 minus this; we'll crash if
  // the number of states ends up being too large to store in this
  // value.
  int32_t b_state_bits_;  // == HighestBitSet(b_fsas_.TotSize(1)).
  int32_t key_bits_;  // b_state_bits_ + HighestBitSet(a_fsas_.MaxSize(1)).


  Hash state_pair_to_state_;



  // This is just a copy of a_fsas_.arcs, with a couple extra pieces of
  // information.
  struct CompressedArc {
    // src_state of Arc, as uint16 (an idx1)
    uint16_t src_state;
    // dest_state of Arc, as uint16 (an idx1)
    uint16_t dest_state;
    // label of Arc, plus one, as uint16
    uint16_t label_plus_one;
    // FSA index, as uint16.
    uint16_t fsa_idx;
    // The idx012 of the position of this arc in the arc_scores_ array in the
    // Step.  This position is obtained by (1) renumbering the arc-indexes in
    // a_fsas_ to produce incoming_arcs_, (2) appending a_fsas_.shape and
    // incoming_arcs_.shape along their axis 1 so they are ordered as: (arcs for
    // fsa 0) (incoming arcs for fsa 0) (arcs for fsa 1) (incoming arcs for fsa
    // 1) and so on.  This is where we'll write the end-loglike of this arc in
    // the forward propagation, to make the reduction easier.
    int32_t incoming_arc_idx012;
    float score;
  };
  // The arcs in a_fsas_.arcs, converted to int16_t's and with a little more
  // information.
  Array1<CompressedArc> carcs_;

  // incoming_arcs_.shape is a modified version of the shape a_fsas_.arcs.shape,
  // so arcs are arranged by dest_state rather than src_state, as returned by
  // GetIncomingArcs().  It's used to do reductions for the forward-pass
  // computation.
  Ragged<int32_t> incoming_arcs_;

  // The shape of arc_scores_ the result of appending a_fsas_.shape and
  // incoming_arcs_.shape along axis 1 (i.e. states).  arc_scores_ is used to
  // take the max of the forward and backward probabilities in parallel on each
  // frame.  Each FSA has twice the number of of states as in a_fsas_, and twice
  // the number of arcs; and the order is (the arcs/states as in a_fsas_, for
  // backward propagation), then (the arcs/states as in incoming_arcs_, for
  // forward propagation).
  Ragged<float> arc_scores_;

  struct FsaInfo {
    // T is the number of frames in b_fsas_.scores that we have for this FSA,
    // i.e. `b_fsas_.shape.RowSplits(1)[i+1] -  b_fsas_.shape.RowSplits(1)[i].`
    // The number of copies of the states of a_fsas_ that we have in the total
    // state space equals T+1, i.e. we have copies of those states for times
    // 0 <= t <= T.
    uint16_t T;
    // num_states is the number of states this FSA has.
    uint16_t num_states;
    // scores_offset is the offset of first location in b_fsas_.scores.Data()
    // that is for this FSA, i.e. b_fsas_.scores.Data()[scores_offset] is the
    // score for t=0, symbol=-1 of this FSA.
    // scores_offset == b_fsas_.shape.RowSplits(1)[fsa_idx] *
    // b_fsas_.scores.ElemStride0().
    int32_t scores_offset;
    // state_offset is the idx0x corresponding to this FSA in a_fsas_.
    int32_t state_offset;
    // arc_offset is the idx0xx corresponding to this FSA in a_fsas_.
    int32_t arc_offset;
  };
  // fsa_info_ is of dimension num_fsas_ + 1 (the last one is not correct in all
  // respects, only certain fields make sense).
  Array1<FsaInfo> fsa_info_;

  struct Step {
    // 0 <= t <= T_ is the time whose states we are writing to in the
    // forward pass on this step of the algorithm (we read from those on time
    // t - 1).  For the backward pass on an FSA with `FsaInfo info`, we write to
    // states with time `t = info.T - t`. (Note: t==0 is a special case where we
    // just initialize the times, there is no propagation.  However we do create
    // the Step struct.)  See DoStep0() for that.
    int32_t t;

    // num_fsas is the number of FSAs that have states active on time t.
    int32_t num_fsas;

    // Ragged array where we will write the scores of arcs before reduction;
    // this is a sub-array of arc_scores_, consisting of the first
    // `num_fsas` FSAs.
    Ragged<float> arc_scores;

    // `state_scores` is where we reduce `arc_scores` to; its Dim() equals
    // arc_scores.TotSize(1).  [arc_scores has 3 axes]. This storage is ACTALLY
    // ALLOCATED HERE, unlike other arrays declared here.
    // The order is:  [backward scores for FSA 0][forward scores for FSA 0]
    // [backward scores for FSA 1][forward scores for FSA 1] and so on.
    Array1<float> state_scores;
  };

  // steps_.size() ==  T_ + 1.
  // steps_[0] is "special", on that step we do initialization.
  std::vector<Step> steps_;

  // Pointer to the Data() pointers of the state_scores elements of steps_[i]
  // for 0 <= i <= T_.
  Array1<float *> state_scores_;

  float output_beam_;

  int32_t T_;  // == b_fsas_.MaxSize(1)
};



bool IntersectDevice(FsaVec &a_fsas, int32_t properties_a,
                     FsaVec &b_fsas, int32_t properties_b,
                     const Array1<int32_t> &b_to_a_map,
                     FsaVec *out,
                     Array1<int32_t> *arc_map_a,
                     Array1<int32_t> *arc_map_b) {
  NVTX_RANGE("IntersectDevice");
  K2_CHECK_NE(properties_a & kFsaPropertiesValid, 0);
  K2_CHECK_NE(properties_b & kFsaPropertiesValid, 0);
  K2_CHECK_EQ(a_fsas.NumAxes(), 3);
  K2_CHECK_EQ(b_fsas.NumAxes(), 3);
  K2_CHECK_EQ(b_to_a_map.Dim(), b_fsas.Dim0());
  K2_CHECK_LT(static_cast<uint32_t>(MaxValue(b_to_a_map)),
              static_cast<uint32_t>(a_fsas.Dim0()));

  DeviceIntersector intersector(a_vec, b_fsas, output_beam);

  intersector.Intersect();
  FsaVec ret = intersector.FormatOutput(arc_map_a, arc_map_b);
  *out = ret;
}
}  // namespace k2
