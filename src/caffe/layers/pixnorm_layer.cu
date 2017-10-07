#include <vector>

#include "caffe/layers/pixnorm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void cdist_fwd_kernel(const int count, const int channels, const int height, const int width,
            const Dtype *input_data, const Dtype *weight_data, Dtype *output_mean, Dtype *output_std) { //, const Dtype *bias_data
  CUDA_KERNEL_LOOP(index, count) {
      output_mean+=index;
      const int w = index % width;
      index /= width;
      const int h = index % height;
      const int n = index / height;

      input_data += ((n * channels ) * height + h) * width + w;

      Dtype v = 0;
      caffe_gpu_strided_dot(channels, weight_data, 1, input_data, i*height*width, v);
      // for (int i = 0; i < channels; i++){
      //   v += input_data[];
      // }

      *output_mean = v;
  }
}

template <typename Dtype>
void pixnormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num;
  int c = bottom[0]->channels();
  num = bottom[0]->count() / c;
  int stride = bottom[0]->count(2);
  const Dtype* mult_data = sum_multiplier_.gpu_data();
  // subtract mean
  for (int i = 0; i < num; ++i){
    caffe_gpu_strided_dot(c, mult_data, 1, bottom_data, stride, mean_);
    bottom_data += num % stride_;
    ++mean_;
  }

  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());  // EX
  caffe_gpu_gemv<Dtype>(CblasTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());  // EX


  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());
  caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(),
      top_data);  // X-EX

  if (this->layer_param_.pixnorm_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.gpu_data(),
        sum_multiplier_.gpu_data(), 0.,
        variance_.mutable_gpu_data());  // E((X-EX)^2)

    // normalize variance
    caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
          variance_.mutable_gpu_data());

    caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());

    caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
  }
}

template <typename Dtype>
void pixnormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


INSTANTIATE_LAYER_GPU_FUNCS(pixnormLayer);


}  // namespace caffe
