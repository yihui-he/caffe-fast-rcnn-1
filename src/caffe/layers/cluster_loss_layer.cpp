#include <vector>
#include <cfloat>

#include "caffe/layers/cluster_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void ClusterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ClusterParameter cluster_param = this->layer_param_.cluster_param();
  reset_centers_ = false;
  coeff_ = (Dtype)1.f;
  num_centers_ = cluster_param.num_centers();
  num_dims_ = bottom[0]->shape(1);
  num_top_ = top.size();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(4);
    weight_shape[0] = num_centers_;
    weight_shape[1] = num_dims_;
    weight_shape[2] = 1;
    weight_shape[3] = 1;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    if (cluster_param.weight_filler().type() != "reset") {
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          cluster_param.weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    }
    else {
      reset_centers_ = true;
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);


  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  top[1]->Reshape(loss_shape);
  assign_matrix_back_.Reshape(num_centers_, 
    1, 
    1, 
    1);
  if (num_top_ == 3) {
    vector<int> top_shape(4);
    top_shape[0] = bottom[0]->shape(0);
    top_shape[1] = 1;
    top_shape[2] = bottom[0]->shape(2);
    top_shape[3] = bottom[0]->shape(3);    
    top[2]->Reshape(top_shape);
  }
}



template <typename Dtype>
void ClusterLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // init internal structures
  distance_matrix_.Reshape(bottom[0]->num(), 
  num_centers_, 
  bottom[0]->height(),
  bottom[0]->width());
assign_matrix_.Reshape(bottom[0]->num(), 
  1, 
  bottom[0]->height(),
  bottom[0]->width());
loss_matrix_.Reshape(bottom[0]->num(), 
  1, 
  bottom[0]->height(),
  bottom[0]->width());
  if (num_top_ == 3) {
    top[2]->ReshapeLike(assign_matrix_);
  }  
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}


template <typename Dtype>
void ClusterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ClusterLossLayer);
#endif

INSTANTIATE_CLASS(ClusterLossLayer);
REGISTER_LAYER_CLASS(ClusterLoss);

}  // namespace caffe
