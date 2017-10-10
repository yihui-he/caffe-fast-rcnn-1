#include <cfloat>
#include <vector>

#include "caffe/layers/kmeans_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void KmeansLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  cluster_loss_layer::Forward_gpu(bottom, top);
}

template <typename Dtype>
void KmeansLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
  // do not bp weights      
  this->param_propagate_down_[0] = 0;
  cluster_loss_layer::Backward_gpu(top, propagate_down, bottom);      
  if (current_kmeans_batch_ == 0) {
    if (current_iter_ % update_interval_ == 0) { // start online kmeans
      init_centers();

      current_kmeans_batch_ = 1;
    }
  }
  else {
    current_kmeans_batch_++;
  }

  current_iter_++;

  if (current_kmeans_batch_ > 0) {
    // calc assign
    find_nearest();

    // do kmeans
    minibatch_kmeans();

    if (current_kmeans_batch_ >= update_iters_) {
      // finish update; apply centers

      caffe_gpu_memcpy(this.blob_[0]->count(), (Blob<Dtype>*)prepare_centers_, this.blob_[0]);
      LOG(INFO) << "KMeans centers updated";

      current_kmeans_batch_ = 0;
    }
  }   
}

template <typename Dtype>
void KmeansLayer<Dtype>::init_centers() {
  caffe_gpu_memcpy(this.blob_[0]->count(), this.blob_[0], prepare_centers_);
  Dtype *center_data = prepare_centers_.mutable_cpu_data();

  for (int c = 0; c < num_centers_; c++) {
    if (center_count_[c] == 0) {
      int nearest_code = (int)(assign_matrix_back_.cpu_data()[c] + 0.5f);
      int spatial_size = bottom[0]->height() * bottom[0]->width();
      int nearest_n = nearest_code / spatial_size, nearest_hw = nearest_code % spatial_size;
      const Dtype *src_data = bottom[0]->cpu_data() + nearest_n * bottom[0]->channels() * spatial_size + nearest_hw;
      Dtype *dest_data = center_data + c * num_dims_;
      for (int i = 0; i < num_dims_; i++) {
        dest_data[i] = src_data[i * spatial_size];
      }
    }
  }
  std::fill(center_count_.begin(), center_count_.end(), 0);
}

template <typename Dtype>
__global__ static void calc_distance_matrix(const int count, const int spatial_size,
  const int num_centers, const int num_dims, const Dtype *inputs, const Dtype *clusters,
  Dtype *distance_matrix) {
  CUDA_KERNEL_LOOP(index, count) {
    const int hw = index % spatial_size;
    const int c = index / spatial_size % num_centers;
    const int n = index / spatial_size / num_centers;

    inputs += n * num_dims * spatial_size + hw;
    clusters += c * num_dims;

    Dtype v = 0.f;
    for (int i = 0; i < num_dims; i++) {
      Dtype vi = inputs[i * spatial_size];
      Dtype vc = clusters[i];
      //v += fabsf(vi - vc); // l1
      v += (vi - vc) * (vi - vc); // l2
    }

    distance_matrix[index] = v;
  }
}

template <typename Dtype>
__global__ static void calc_assign_matrix(const int count, const int spatial_size,
  const int num_centers, const Dtype *distance_matrix,
  Dtype *assign_matrix) {
  CUDA_KERNEL_LOOP(index, count) {
    const int hw = index % spatial_size;
    const int n = index / spatial_size;

    distance_matrix += n * num_centers * spatial_size + hw;

    Dtype min_dis = FLT_MAX;
    int ass_center = -1;
    for (int i = 0; i < num_centers; i++) {
      Dtype v = distance_matrix[i * spatial_size];
      if (v < min_dis) {
        min_dis = v;
        ass_center = i;
      }
    }

    assign_matrix[index] = (Dtype)ass_center;
  }
}

template <typename Dtype>
void KmeansLayer<Dtype>::find_nearest(const vector<Blob<Dtype>*>& bottom) {
  int count_dis = prepare_distance_matrix_.count();
  KERNEL_CALL(calc_distance_matrix, count_dis)(
    count_dis,
    bottom[0]->height() * bottom[0]->width(),
    num_centers_,
    num_dims_,
    bottom[0]->gpu_data(),
    prepare_centers_.gpu_data(),
    prepare_distance_matrix_.mutable_gpu_data()
    );

  int count_ass = prepare_assign_matrix_.count();
  KERNEL_CALL(calc_assign_matrix, count_ass)(
    count_ass,
    bottom[0]->height() * bottom[0]->width(),
    num_centers_,
    prepare_distance_matrix_.gpu_data(),
    prepare_assign_matrix_.mutable_gpu_data()
    );
}

template <typename Dtype>
void KmeansLayer<Dtype>::minibatch_kmeans(const vector<Blob<Dtype>*>& bottom) {
  // always update centers on the first device
  Dtype *centers = prepare_centers_.mutable_cpu_data();

  const Dtype *assigns = prepare_assign_matrix_.cpu_data();

  int spatial_size = bottom[0]->height() * bottom[0]->width();
  int num = bottom[0]->num();
  
  const Dtype *input_data = bottom[0]->cpu_data();

  for (int n = 0; n < num; n++) {
    for (int s = 0; s < spatial_size; s++) {
      int center_index = (int)(assigns[n * spatial_size + s] + 0.5f);
      center_count_[center_index]++;

      Dtype ratio = 1.f / center_count_[center_index];
      const Dtype *src_data = input_data + n * num_dims_ * spatial_size + s;
      Dtype *dest_data = centers + center_index * num_dims_;
      for (int i = 0; i < num_dims_; i++) {
        dest_data[i] = dest_data[i] * (1 - ratio) + src_data[i * spatial_size] * ratio;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KmeansLayer);

}  // namespace caffe
