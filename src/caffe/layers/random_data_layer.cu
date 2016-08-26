#include <stdint.h>
#include <vector>

#include "caffe/layers/random_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RandomDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int count = top[0]->count();
  const int batch_size = top[1]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();        
  caffe_gpu_rng_uniform(count, Dtype(-1), Dtype(1), top_data);

  // label set to 0, since they are fake 
  if (this->layer_param_.top_size() > 1) {
      Dtype* label_data = top[1]->mutable_gpu_data();
      caffe_gpu_set(batch_size, Dtype(0), label_data);
      if (this->layer_param_.top_size() == 3){
        Dtype* true_label = top[2]->mutable_gpu_data();
        caffe_gpu_set(batch_size, Dtype(1), true_label);
      }  
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RandomDataLayer);

}  // namespace caffe
