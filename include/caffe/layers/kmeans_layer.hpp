#ifndef KMEANS_LAYER_HPP_
#define KMEANS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/cluster_loss_layer.hpp"
namespace caffe {

/**
 * 
 */
template <typename Dtype>
class KmeansLayer : public cluster_loss_layer<Dtype> {
 public:
  explicit KmeansLayer(const LayerParameter& param)
    : cluster_loss_layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);    
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Kmeans"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  


 protected:
  /// @copydoc KmeansLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * 
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    void init_centers();
    void find_nearest();
    void minibatch_kmeans();

    int update_interval_, update_iters_;
    int current_iter_, current_kmeans_batch_;

    Blob<Dtype> prepare_centers_, prepare_distance_matrix_, prepare_assign_matrix_;
    std::vector<int> center_count_;
};

}  // namespace caffe

#endif  // KMEANS_LAYER_HPP_
