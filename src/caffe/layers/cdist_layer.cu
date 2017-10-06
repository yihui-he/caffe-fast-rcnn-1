#include <vector>

#include "caffe/layers/cdist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void cdist_fwd_kernel(const int count, const int channels, const int height, const int width,
              const int out_channels,
              const Dtype *input_data, const Dtype *weight_data, Dtype *output_data) { //, const Dtype *bias_data
    CUDA_KERNEL_LOOP(index, count) {
        output_data += index;
        int ow = index % width;
        index /= width;
        int oh = index % height;
        index /= height;
        int oc = index % out_channels;
        int n = index / out_channels;
  
        int iw = ow;
        int ih = oh;
  
        input_data += ((n * channels ) * height + ih) * width + iw;
        weight_data += oc * channels;
  
        Dtype v = 0;
        for (int i = 0; i < channels; i++){
          v += (input_data - weight_data[i]) * (input_data - weight_data[i])
          input_data += i * height * width;
        }
  
        *output_data = v;
    }
  }


template <typename Dtype>
void cdistLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int dim = top[0]->count();
  spatial_conv_fwd_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, bottom[0]-shape(1), bottom[0]-shape(2), bottom[0]-shape(3), top[0]-shape(0), bottom_data, weight, top_data);
}

template <typename Dtype>
void cdistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(cdistLayer);

}  // namespace caffe
