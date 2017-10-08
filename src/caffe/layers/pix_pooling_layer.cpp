// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void PIXPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PIXPoolingParameter pix_pool_param = this->layer_param_.pix_pooling_param();
  CHECK_GT(pix_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(pix_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = pix_pool_param.pooled_h();
  pooled_width_ = pix_pool_param.pooled_w();
  n_vc_ = pix_pool_param.n_vc();
  spatial_scale_ = pix_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;

  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = n_vc_ * pooled_height_ * pooled_width_;
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization

  vector<int> temp_shape(N_);
  temp_.Reshape(temp_shape);

  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void PIXPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels(); // is 1
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[1]->shape(0);
  
  vector<int> top_shape(2);
  top_shape[0] = M_;
  top_shape[1] = N_;
  top[0]->Reshape(top_shape);
  max_idx_.Reshape(K_);
  caffe_set(M_, Dtype(0), bias_multiplier_.mutable_cpu_data());
  
  // // Figure out the dimensions
  // const int axis = bottom[0]->CanonicalAxisIndex(
  //   this->layer_param_.inner_product_param().axis());
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void PIXPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void PIXPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(PIXPoolingLayer);
#endif

INSTANTIATE_CLASS(PIXPoolingLayer);
REGISTER_LAYER_CLASS(PIXPooling);

}  // namespace caffe
