#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cdist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void cdistLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void cdistLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(4);
  top_shape[0] = bottom[0]->shape(0);
  top_shape[1] = bottom[1]->shape(0);
  top_shape[2] = bottom[0]->shape(2);
  top_shape[3] = bottom[0]->shape(3);
  top[0]->Reshape(top_shape);
  cos_ = this->layer_param_.cdist_param().cos();
  
}

template <typename Dtype>
void cdistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;            
}

template <typename Dtype>
void cdistLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;      
}

#ifdef CPU_ONLY
STUB_GPU(cdistLayer);
#endif

INSTANTIATE_CLASS(cdistLayer);
REGISTER_LAYER_CLASS(cdist);

}  // namespace caffe
