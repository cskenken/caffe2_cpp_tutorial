#ifndef BUILD_H
#define BUILD_H

#include "caffe2/core/net.h"

namespace caffe2 {

static const std::set<std::string> backprop_ops({
  "FC",
  "Relu",
  "Softmax",
  "Conv",
  "MaxPool",
  "LabelCrossEntropy",
  "AveragedLoss"
});

static const std::string gradient_suffix("_grad");

// Operators

OperatorDef *add_create_db_op(NetDef &model, const std::string &reader, const std::string &db_type, const std::string &db_path) {
  auto op = model.add_op();
  op->set_type("CreateDB");
  auto arg1 = op->add_arg();
  arg1->set_name("db_type");
  arg1->set_s(db_type);
  auto arg2 = op->add_arg();
  arg2->set_name("db");
  arg2->set_s(db_path);
  op->add_output(reader);
  return op;
}

OperatorDef *add_tensor_protos_db_input_op(NetDef &model, const std::string &reader, const std::string &data, const std::string &label, int batch_size) {
  auto op = model.add_op();
  op->set_type("TensorProtosDBInput");
  auto arg = op->add_arg();
  arg->set_name("batch_size");
  arg->set_i(batch_size);
  op->add_input(reader);
  op->add_output(data);
  op->add_output(label);
  return op;
}

OperatorDef *add_cout_op(NetDef &model, const std::string &param) {
  auto op = model.add_op();
  op->set_type("Cout");
  auto arg = op->add_arg();
  op->add_input(param);
  return op;
}

OperatorDef *add_accuracy_op(NetDef &model, const std::string &prob, const std::string &label, const std::string &accuracy) {
  auto op = model.add_op();
  op->set_type("Accuracy");
  op->add_input(prob);
  op->add_input(label);
  op->add_output(accuracy);
  return op;
}

OperatorDef *add_label_cross_entropy_op(NetDef &model, const std::string &prob, const std::string &label, const std::string &xent) {
  auto op = model.add_op();
  op->set_type("LabelCrossEntropy");
  op->add_input(prob);
  op->add_input(label);
  op->add_output(xent);
  return op;
}

OperatorDef *add_averaged_loss(NetDef &model, const std::string &xent, const std::string &loss) {
  auto op = model.add_op();
  op->set_type("AveragedLoss");
  op->add_input(xent);
  op->add_output(loss);
  return op;
}

OperatorDef *add_weighted_sum_op(NetDef &model, const std::vector<std::string> &inputs, const std::string &sum) {
  auto op = model.add_op();
  op->set_type("WeightedSum");
  for (const auto &input: inputs) {
    op->add_input(input);
  }
  op->add_output(sum);
  return op;
}

OperatorDef *add_constant_fill_op(NetDef &model, const std::vector<int> &shape, const std::string &param) {
  auto op = model.add_op();
  op->set_type("ConstantFill");
  auto arg1 = op->add_arg();
  arg1->set_name("shape");
  for (auto dim: shape) {
    arg1->add_ints(dim);
  }
  op->add_output(param);
  return op;
}

OperatorDef *add_constant_fill_float_op(NetDef &model, float value, const std::string &param) {
  auto op = add_constant_fill_op(model, { 1 }, param);
  auto arg = op->add_arg();
  arg->set_name("value");
  arg->set_f(value);
  return op;
}

OperatorDef *add_constant_fill_int64_op(NetDef &model, int64_t value, const std::string &param) {
  auto op = add_constant_fill_op(model, { 1 }, param);
  auto arg1 = op->add_arg();
  arg1->set_name("value");
  arg1->set_i(value);
  auto arg2 = op->add_arg();
  arg2->set_name("dtype");
  arg2->set_i(TensorProto_DataType_INT64);
  return op;
}

OperatorDef *add_constant_fill_with_op(NetDef &model, float value, const std::string &input, const std::string &output) {
  auto op = model.add_op();
  op->set_type("ConstantFill");
  auto arg = op->add_arg();
  arg->set_name("value");
  arg->set_f(value);
  op->add_input(input);
  op->add_output(output);
  op->set_is_gradient_op(true);
  return op;
}

OperatorDef *add_iter_op(NetDef &model, const std::string &iter) {
  auto op = model.add_op();
  op->set_type("Iter");
  op->add_input(iter);
  op->add_output(iter);
  return op;
}

OperatorDef *add_learning_rate_op(NetDef &model, const std::string &iter, const std::string &rate, float base, float gamma) {
  auto op = model.add_op();
  op->set_type("LearningRate");
  auto arg1 = op->add_arg();
  arg1->set_name("policy");
  arg1->set_s("step");
  auto arg2 = op->add_arg();
  arg2->set_name("stepsize");
  arg2->set_i(1);
  auto arg3 = op->add_arg();
  arg3->set_name("base_lr");
  arg3->set_f(-base);
  auto arg4 = op->add_arg();
  arg4->set_name("gamma");
  arg4->set_f(gamma);
  op->add_input(iter);
  op->add_output(rate);
  return op;
}

OperatorDef *add_copy_gpu_to_cpu_op(NetDef &model, const std::string &input, const std::string &output) {
  auto op = model.add_op();
#ifdef WITH_CUDA
  op->set_type("CopyGPUToCPU");
#else
  op->set_type("Copy");
#endif
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_copy_cpu_to_gpu_op(NetDef &model, const std::string &input, const std::string &output) {
  auto op = model.add_op();
#ifdef WITH_CUDA
  op->set_type("CopyCPUToGPU");
#else
  op->set_type("Copy");
#endif
  op->add_input(input);
  op->add_output(output);
  return op;
}

// Helpers

void cudnn_op(OperatorDef &op, bool use_cudnn) {
  if (use_cudnn) {
    op.set_engine("CUDNN");
#ifdef WITH_CUDA
    op.mutable_device_option()->set_device_type(CUDA);
#endif
  }
}

void copy_op(const OperatorDef &from, OperatorDef &to, bool use_cudnn) {
  to.CopyFrom(from);
  cudnn_op(to, use_cudnn);
}

void add_to_cpu_op(NetDef &model, const std::vector<std::string> &params, bool use_cudnn) {
  if (use_cudnn) {
    for (auto &param: params) {
      add_copy_gpu_to_cpu_op(model, param, param + "_host");
    }
  }
}

void add_from_gpu_op(NetDef &model, const std::vector<std::string> &params, bool use_cudnn) {
  if (use_cudnn) {
    for (auto &param: params) {
      add_copy_gpu_to_cpu_op(model, param + "_gpu", param);
    }
  }
}

void add_to_gpu_op(NetDef &model, const std::vector<std::string> &params, bool use_cudnn) {
  if (use_cudnn) {
    for (auto &param: params) {
      add_copy_cpu_to_gpu_op(model, param + "_host", param);
    }
  }
}

void add_from_cpu_op(NetDef &model, const std::vector<std::string> &params, bool use_cudnn) {
  if (use_cudnn) {
    for (auto &param: params) {
      add_copy_cpu_to_gpu_op(model, param, param + "_gpu");
    }
  }
}

OperatorDef *add_gradient_op(NetDef &model, const OperatorDef *op) {
  vector<GradientWrapper> output(op->output_size());
  for (auto i = 0; i < output.size(); i++) {
    output[i].dense_ = op->output(i) + gradient_suffix;
  }
  GradientOpsMeta meta = GetGradientForOp(*op, output);
  auto grad = model.add_op();
  grad->CopyFrom(meta.ops_[0]);
  grad->set_is_gradient_op(true);
  return grad;
}

void add_database_ops(NetDef &init_model, NetDef &predict_model, const std::string &name, const std::string &data, const std::string &db, const std::string &db_type, int batch_size, bool use_cudnn) {
  auto suffix = use_cudnn ? "_host" : "";
  auto reader = name + "_dbreader";
  add_create_db_op(init_model, reader, db_type, db);
  predict_model.add_external_input(reader);
  add_tensor_protos_db_input_op(predict_model, reader, data + suffix, "label", batch_size);
  add_to_gpu_op(predict_model, { data }, use_cudnn);
  add_from_cpu_op(predict_model, { "label" }, use_cudnn);
  // add_cout_op(predict_model, data);
}

void add_test_ops(NetDef &predict_model) {
  add_accuracy_op(predict_model, "prob", "label", "accuracy");
}

void add_train_ops(NetDef &init_model, NetDef &predict_model, float learning_rate, float learning_gamma, bool use_cudnn) {
  auto host = use_cudnn ? "_host" : "";
  auto gpu = use_cudnn ? "_gpu" : "";
  cudnn_op(*add_label_cross_entropy_op(predict_model, "prob", std::string("label") + gpu, "xent"), use_cudnn);
  cudnn_op(*add_averaged_loss(predict_model, "xent", "loss"), use_cudnn);
  add_to_cpu_op(predict_model, { "prob" }, use_cudnn);
  add_accuracy_op(predict_model, std::string("prob") + host, "label", "accuracy");
  add_constant_fill_int64_op(init_model, 0, "ITER");
  predict_model.add_external_input("ITER");
  add_iter_op(predict_model, "ITER");
  cudnn_op(*add_constant_fill_with_op(predict_model, 1.0, "loss", "loss" + gradient_suffix), use_cudnn);

  // collect gradient inputs and gradient operators
  std::vector<std::string> gradient_inputs;
  std::vector<const OperatorDef *> gradient_ops;
  std::set<std::string> external_inputs(predict_model.external_input().begin(), predict_model.external_input().end());
  for (const auto &op: predict_model.op()) {
    if (backprop_ops.find(op.type()) != backprop_ops.end()) {
      for (const auto &input: op.input()) {
        if (external_inputs.find(input) != external_inputs.end()) {
          gradient_inputs.push_back(input);
          // std::cout << "input :" << input << std::endl;
        }
      }
      gradient_ops.push_back(&op);
      // std::cout << "type: " << op.type() << std::endl;
    }
  }

  for (auto i = gradient_ops.rbegin(); i != gradient_ops.rend(); ++i) {
    cudnn_op(*add_gradient_op(predict_model, *i), use_cudnn);
  }
  add_learning_rate_op(predict_model, "ITER", "LR", learning_rate, learning_gamma);
  add_constant_fill_float_op(init_model, 1.0, "ONE");
  predict_model.add_external_input("ONE");
  add_from_cpu_op(predict_model, { "ONE", "LR" }, use_cudnn);
  for (auto param: gradient_inputs) {
    cudnn_op(*add_weighted_sum_op(predict_model, { param, std::string("ONE") + gpu, param + gradient_suffix, std::string("LR") + gpu }, param), use_cudnn);
  }
}

}  // namespace caffe2

#endif  // BUILD_H
