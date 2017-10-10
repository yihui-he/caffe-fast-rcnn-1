#include <vector>

#include "caffe/layers/cluster_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClusterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ClusterParameter cluster_param = this->layer_param_.cluster_param();
  reset_centers_ = false;
  coeff_ = (Dtype)1.f;
  num_centers_ = cluster_param.num_centers();
  num_dims_ = bottom[0]->shape(1);

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
  // init internal structures
  distance_matrix_.Reshape(bottom[0]->num(), 
                           num_centers_, 
                           bottom[0]->height(),
                           bottom[0]->width());
  assign_matrix_.Reshape(bottom[0]->num(), 
                           1, 
                           bottom[0]->height(),
                           bottom[0]->width());
  assign_matrix_back_.Reshape(num_centers_, 
                           1, 
                           1, 
                           1);
  loss_matrix_.Reshape(bottom[0]->num(), 
                           1, 
                           bottom[0]->height(),
                           bottom[0]->width());

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
  Dtype *assign_matrix, Dtype *loss_matrix) {
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
    loss_matrix[index] = min_dis;
  }
}

template <typename Dtype>
__global__ static void calc_assign_matrix_back(const int count, const int spatial_size, 
  const int input_num, const int num_centers,
  const Dtype *distance_matrix, Dtype *assign_matrix_back) {
  CUDA_KERNEL_LOOP(index, count) {
    Dtype min_dis = FLT_MAX;
    int ass_back_spatial = -1, ass_back_inputnum = -1;

    for (int n = 0; n < input_num; n++) {
      for (int s = 0; s < spatial_size; s++) {
        Dtype v = distance_matrix[(n * num_centers + index) * spatial_size + s];

        if (v < min_dis) {
          min_dis = v;
          ass_back_spatial = s;
          ass_back_inputnum = n;
        }
      }
    }

    assign_matrix_back[index] = ass_back_inputnum * spatial_size + ass_back_spatial;
  }
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if (reset_centers_) {
			int num = bottom[0]->num();
      int spatial_size = bottom[0]->height() * bottom[0]->width();
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.f);
      filler.reset(GetFiller<Dtype>(filler_param));
      filler->Fill(this->blobs_[0].get());
			for (int i = 0; i < num_centers_; i++) {
				int sel_num = caffe_rng_rand() % num;
				int sel_spatial = caffe_rng_rand() % spatial_size;

        const Dtype *src = bottom[0]->gpu_data() + bottom[0]->offset(sel_num) + sel_spatial;
        caffe_gpu_axpystep(num_dims_, 
            (Dtype)1.f, 
            src, 
            spatial_size, 
            this->blobs_[0]->mutable_gpu_data() + this->blobs_[0]->offset(i),
            1);
			}
			reset_centers_ = false;
		}
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int count_dis = distance_matrix_.count();
    int spatial_size = bottom[0]->height() * bottom[0]->width();
    calc_distance_matrix<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(count_dis), CAFFE_CUDA_NUM_THREADS>>>(
      count_dis, 
      spatial_size,
      num_centers_,
      num_dims_,
      bottom[0]->gpu_data(),
      this->blobs_[0]->gpu_data(),
      distance_matrix_.mutable_gpu_data()
      );

    int count_ass = assign_matrix_.count();
    calc_assign_matrix<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(count_ass), CAFFE_CUDA_NUM_THREADS>>>(
      count_ass,
      spatial_size,
      num_centers_,
      distance_matrix_.gpu_data(),
      assign_matrix_.mutable_gpu_data(),
      loss_matrix_.mutable_gpu_data()
      );

      calc_assign_matrix_back<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(num_centers_), CAFFE_CUDA_NUM_THREADS>>>(        
      num_centers_,
      spatial_size,
      bottom[0]->num(),
      num_centers_,
      distance_matrix_.gpu_data(),
      assign_matrix_back_.mutable_gpu_data()
      );
      Dtype loss;
      caffe_gpu_asum(count_ass, loss_matrix_, loss);
      top[0]->mutable_cpu_data()[0] = (Dtype)loss / spatial_size / bottom[0]->num()  / num_dims_;

      // merge diversity
      std::set<int> unique_assign;
      const Dtype *dev_assign = assign_matrix_.cpu_data();
      int count = assign_matrix_.count();
      for (int i = 0; i < count; i++) {
        int v = (int)(dev_assign[i] + 0.5f);
        unique_assign.insert(v);
      }

      top[0]->mutable_cpu_data()[1] = (Dtype)unique_assign.size();

}


template <typename Dtype>
__global__ static void bp_acts_kernel(const int count, const int spatial_size, 
  const int num_dims, const Dtype *inputs, const Dtype *clusters, const Dtype *assign_matrix,
  Dtype *act_diff, const Dtype coeff, const Dtype scale_targets) {
  CUDA_KERNEL_LOOP(index, count) {
    const int hw = index % spatial_size;
    const int c = index / spatial_size % num_dims;
    const int n = index / spatial_size / num_dims;

    int idx_center = (int)(assign_matrix[n * spatial_size + hw] + 0.5f);
    Dtype vc = clusters[idx_center * num_dims + c];
    Dtype vi = inputs[index];
    //Dtype diff = coeff * (vi >= vc ? 1 : -1); // l1
    Dtype diff = coeff * (vi - vc) * 2; // l2
    if (scale_targets == 0) {
      act_diff[index] = diff;
    }
    else {
      act_diff[index] = act_diff[index] * scale_targets + diff;
    }
  }
}


__global__ static void bp_weights_kernel(const int count, const int spatial_size, const int num,
  const int num_dims, const Dtype *inputs, const Dtype *clusters, const Dtype *assign_matrix, const Dtype *assign_matrix_back,
  Dtype *cluster_diff, const Dtype coeff, const Dtype scale_targets) {
  CUDA_KERNEL_LOOP(index, count) {
    const int c = index % num_dims;
    const int idx_center = index / num_dims;

    Dtype diff = 0.f;
    int k = 0;
    bool assigned = false;
    for (int n = 0; n < num; n++) {
      for (int s = 0; s < spatial_size; s++, k++) {
        int ass_index = (int)(assign_matrix[k] + 0.5f);

        if (ass_index == idx_center) {
          Dtype vi = inputs[(n * num_dims + c) * spatial_size + s];
          Dtype vc = clusters[index];
          //diff += vc > vi ? 1 : -1; // l1
          diff += (vc - vi) * 2; // l2
          assigned = true;
        }
      }
    }

    if (!assigned) {
      int nearest_input_pos = assign_matrix_back[idx_center];
      int nearest_input_index = (nearest_input_pos / spatial_size * num_dims + c) * spatial_size + nearest_input_pos % spatial_size;

      Dtype vi = inputs[nearest_input_index];
      Dtype vc = clusters[index];

      diff = (vc - vi) * 2; // l2
    }

    diff *= coeff;

    if (scale_targets == 0) {
      cluster_diff[index] = diff;
    }
    else {
      cluster_diff[index] = cluster_diff[index] * scale_targets + diff;
    }
  }
}



template <typename Dtype>
void ClusterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
      if (propagate_down[0]) {
        const Dtype beta_acts = 0;
  
        int count = bottom[0]->count();
        bp_acts_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count,
          bottom[0]->height() * bottom[0]->width(),
          num_dims_,
          bottom[0]->gpu_data(),
          this->blobs_[0]->gpu_data(),
          assign_matrix_.gpu_data(),
          bottom[0]->mutable_gpu_diff(),
          coeff_ / (bottom[0]->count()),
          beta_acts
          );
      }
  
      if (this->param_propagate_down_[0]) {
        const Dtype beta_weights = 0;
  
        int count = this->blobs_[0]->count();
        bp_weights_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count,
          bottom[0]->height() * bottom[0]->width(),
          bottom[0]->num(),
          num_dims_,
          bottom[0]->gpu_data(),
          this->blobs_[0]->gpu_data(),
          assign_matrix_.gpu_data(),
          assign_matrix_back->gpu_data(),
          this->blobs_[0]->mutable_gpu_diff(),
          coeff_ / (bottom[0]->count()),
          beta_weights
          );
      }
}

#ifdef CPU_ONLY
STUB_GPU(ClusterLossLayer);
#endif

INSTANTIATE_CLASS(ClusterLossLayer);
REGISTER_LAYER_CLASS(ClusterLoss);

}  // namespace caffe
