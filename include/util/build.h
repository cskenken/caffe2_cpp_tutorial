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

OperatorDef *add_create_db_op(NetDef &model, const std::string &output, const std::string &db_type, const std::string &db) {
  auto op = model.add_op();
  op->set_type("CreateDB");
  auto arg1 = op->add_arg();
  arg1->set_name("db_type");
  arg1->set_s(db_type);
  auto arg2 = op->add_arg();
  arg2->set_name("db");
  arg2->set_s(db);
  op->add_output(output);
  return op;
}

OperatorDef *add_create_db_op(NetDef &model, const std::string &input, const std::string &output, int batch_size) {
  auto op = model.add_op();
  op->set_type("TensorProtosDBInput");
  auto arg = op->add_arg();
  arg->set_name("batch_size");
  arg->set_i(batch_size);
  op->add_input(input);
  op->add_output(output);
  op->add_output("label");
  return op;
}

OperatorDef *add_cout_op(NetDef &model, const std::string &input) {
  auto op = model.add_op();
  op->set_type("Cout");
  auto arg = op->add_arg();
  op->add_input(input);
  return op;
}

OperatorDef *add_accuracy_op(NetDef &model) {
  auto op = model.add_op();
  op->set_type("Accuracy");
  op->add_input("prob");
  op->add_input("label");
  op->add_output("accuracy");
  return op;
}

OperatorDef *add_label_cross_entropy_op(NetDef &model) {
  auto op = model.add_op();
  op->set_type("LabelCrossEntropy");
  op->add_input("prob");
  op->add_input("label");
  op->add_output("xent");
  return op;
}

OperatorDef *add_averaged_loss(NetDef &model) {
  auto op = model.add_op();
  op->set_type("AveragedLoss");
  op->add_input("xent");
  op->add_output("loss");
  return op;
}

OperatorDef *add_weighted_sum_op(NetDef &model, const std::vector<std::string> &inputs, const std::string &output) {
  auto op = model.add_op();
  op->set_type("WeightedSum");
  for (const auto &input: inputs) {
    op->add_input(input);
  }
  op->add_output(output);
  return op;
}

OperatorDef *add_constant_fill_op(NetDef &model, const std::vector<int> &shape, const std::string &output) {
  auto op = model.add_op();
  op->set_type("ConstantFill");
  auto arg1 = op->add_arg();
  arg1->set_name("shape");
  for (auto dim: shape) {
    arg1->add_ints(dim);
  }
  op->add_output(output);
  return op;
}

OperatorDef *add_constant_fill_float_op(NetDef &model, float value, const std::string &output) {
  auto op = add_constant_fill_op(model, { 1 }, output);
  auto arg = op->add_arg();
  arg->set_name("value");
  arg->set_f(value);
  return op;
}

OperatorDef *add_constant_fill_int64_op(NetDef &model, int64_t value, const std::string &output) {
  auto op = add_constant_fill_op(model, { 1 }, output);
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

OperatorDef *add_iter_op(NetDef &model, const std::string &output) {
  auto op = model.add_op();
  op->set_type("Iter");
  op->add_input(output);
  op->add_output(output);
  return op;
}

OperatorDef *add_learning_rate_op(NetDef &model, const std::string &input, const std::string &output, float learning_rate, float learning_gamma) {
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
  arg3->set_f(-learning_rate);
  auto arg4 = op->add_arg();
  arg4->set_name("gamma");
  arg4->set_f(learning_gamma);
  op->add_input(input);
  op->add_output(output);
  return op;
}

// Helpers

void add_gradient_op(NetDef &model, const OperatorDef *op) {
  vector<GradientWrapper> output(op->output_size());
  for (auto i = 0; i < output.size(); i++) {
    output[i].dense_ = op->output(i) + gradient_suffix;
  }
  GradientOpsMeta meta = GetGradientForOp(*op, output);
  auto grad = model.add_op();
  grad->CopyFrom(meta.ops_[0]);
  grad->set_is_gradient_op(true);
}

void add_database_ops(NetDef &init_model, NetDef &predict_model, const std::string &name, const std::string &output, const std::string &db, const std::string &db_type, int batch_size) {
  auto reader = name + "_dbreader";
  add_create_db_op(init_model, reader, db_type, db);
  predict_model.add_external_input(reader);
  add_create_db_op(predict_model, reader, output, batch_size);
  // add_cout_op(predict_model, output);
}

void add_test_ops(NetDef &predict_model) {
  add_accuracy_op(predict_model);
}

void add_train_ops(NetDef &init_model, NetDef &predict_model, float learning_rate, float learning_gamma) {
  add_label_cross_entropy_op(predict_model);
  add_averaged_loss(predict_model);
  add_accuracy_op(predict_model);
  add_constant_fill_int64_op(init_model, 0, "ITER");
  predict_model.add_external_input("ITER");
  add_iter_op(predict_model, "ITER");
  add_constant_fill_with_op(predict_model, 1.0, "loss", "loss" + gradient_suffix);

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
    add_gradient_op(predict_model, *i);
  }
  add_learning_rate_op(predict_model, "ITER", "LR", learning_rate, learning_gamma);
  add_constant_fill_float_op(init_model, 1.0, "ONE");
  predict_model.add_external_input("ONE");
  for (auto param: gradient_inputs) {
    add_weighted_sum_op(predict_model, { param, "ONE", param + gradient_suffix, "LR" }, param);
  }
}

}  // namespace caffe2

#endif  // BUILD_H
