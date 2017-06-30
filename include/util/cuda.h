#ifndef CUDA_H
#define CUDA_H

#ifdef WITH_CUDA
  #include "caffe2/core/context_gpu.h"
#endif

namespace caffe2 {

bool setupCUDA() {
  DeviceOption option;
  option.set_device_type(CUDA);
#ifdef WITH_CUDA
  new CUDAContext(option);
  return true;
#else
  return false;
#endif
}

void set_device_cuda_model(NetDef &model) {
#ifdef WITH_CUDA
  model.mutable_device_option()->set_device_type(CUDA);
#endif
}

TensorCPU get_tensor_blob(const Blob &blob) {
#ifdef WITH_CUDA
  if (blob.IsType<TensorCUDA>()) {
    return TensorCPU(blob.Get<TensorCUDA>());
  }
#endif
  return blob.Get<TensorCPU>();
}

}  // namespace caffe2

#endif  // CUDA_H
