#include <vector>

#include "stdint.h"

#include "caffe/layers/random_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
RandomDataLayer<Dtype>::~RandomDataLayer<Dtype>() { }

template <typename Dtype>
void RandomDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int n = this->layer_param_.random_data_param().n();
  const int c = this->layer_param_.random_data_param().c();
  const int h = this->layer_param_.random_data_param().h();
  const int w = this->layer_param_.random_data_param().w();
  const int top_size = this->layer_param_.top_size();
  vector<int> top_shape(4);

  top_shape[0] = n;
  top_shape[1] = c;
  top_shape[2] = h;
  top_shape[3] = w;
  top[0]->Reshape(top_shape);
  if (top_size > 1) {
    vector<int> label_shape(1, n);
    top[1]->Reshape(label_shape);
    if (top_size == 3) {
      top[2]->Reshape(label_shape);
    }
  }
}

template <typename Dtype>
void RandomDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(RandomDataLayer, Forward);
#endif

INSTANTIATE_CLASS(RandomDataLayer);
REGISTER_LAYER_CLASS(RandomData);

}  // namespace caffe
