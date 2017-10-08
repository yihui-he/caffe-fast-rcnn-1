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
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void PIXPoolForward(const int nthreads, const Dtype* bottom_data,  const Dtype* weight, const int num_output,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, Dtype* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed PIXs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    // Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        int weight_index = ((bottom_data[bottom_index] * pooled_height + ph) * pooled_width +  pw) * num_output;
        caffe_gpu_atomic_add(Dtype(1), argmax_data+weight_index);
        // TODO: it's wrong to not max pooling, though sparse , gradient might sometimes wrong!!
        if (is_empty) {
          ;
        } else {
          caffe_gpu_add(num_output, weight+weight_index, top_data+n*num_output, top_data+n*num_output);
        }
        
        // if (bottom_data[bottom_index] > maxval) {
        //   maxval = bottom_data[bottom_index];
        //   maxidx = bottom_index;
        // }
      }
    }
  }
}

template <typename Dtype>
void PIXPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  caffe_set(count, Dtype(0), top_data);
  
  
  Dtype* argmax_data = max_idx_.mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  PIXPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, weight, N_,
      spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PIXPoolBackward(const int nthreads, const Dtype* temp_data,
    const Dtype* argmax_data, const int num_output, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (argmax_data[index]){
      caffe_gpu_scale(num_output, (Dtype)(argmax_data[index]), temp_data, weight_diff +num_output*index);
    }
  }
}

template <typename Dtype>
void PIXPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // if (!propagate_down[0]) {
  //   return;
  // }
  // const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  
  caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
  bias_multiplier_.gpu_data(), (Dtype)0., temp_.mutable_gpu_data());
  const Dtype* temp_data = temp_.mutable_gpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = K_;
  // caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const Dtype* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  PIXPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, temp_data, argmax_data, N_, this->blobs_[0]->mutable_gpu_diff());
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PIXPoolingLayer);

}  // namespace caffe
