#include <vector>
#include <cfloat>

#include "caffe/layers/kmeans_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void KmeansLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      ClusterLossLayer<Dtype>::LayerSetUp(bottom, top);
        
      // clear weight diff
      FillerParameter filler_param;
      shared_ptr<Filler<Dtype> > filler;
      filler_param.set_type("constant");
      filler_param.set_value(0.f);
      filler.reset(GetFiller<Dtype>(filler_param));
      filler->Fill(this->blobs_[0]->mutable_cpu_diff());      
      

      update_interval_ = this->layer_param_.kmeans_param().update_interval();
      update_iters_ = this->layer_param_.kmeans_param().update_iters();
      current_iter_ = 0;
      current_kmeans_batch_ = 0;
  
      CHECK_GE(update_interval_, update_iters_);
      CHECK_GT(update_iters_, 0);
        
      prepare_centers_.ReshapeLike(*this->blobs_[0]);
      center_count_.resize(this->num_centers_);
      std::fill(center_count_.begin(), center_count_.end(), 1);
      
}



template <typename Dtype>
void KmeansLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    ClusterLossLayer<Dtype>::Reshape(bottom, top);
    prepare_distance_matrix_.ReshapeLike(this->distance_matrix_);
    prepare_assign_matrix_.ReshapeLike(this->assign_matrix_);
    
}

template <typename Dtype>
void KmeansLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}


template <typename Dtype>
void KmeansLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(KmeansLayer);
#endif

INSTANTIATE_CLASS(KmeansLayer);
REGISTER_LAYER_CLASS(Kmeans);

}  // namespace caffe
