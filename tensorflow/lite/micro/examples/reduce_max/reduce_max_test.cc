#include "tensorflow/lite/micro/examples/reduce_max/models/reduce_max_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace {
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReduceMax());
  return kTfLiteOk;
}
}  // namespace

int main(int argc, char* argv[]) {
  const tflite::Model* model = ::tflite::GetModel(g_reduce_max_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  constexpr int tensor_arena_size = 1448;
  uint8_t tensor_arena[tensor_arena_size];
  // std::memset(tensor_arena, 0, tensor_arena_size);

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       tensor_arena_size);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input(i);
    TFLITE_CHECK_NE(input, nullptr);

    for (int j = 0; j < tflite::GetTensorShape(input).FlatSize(); ++j) {
      tflite::GetTensorData<int8_t>(input)[j] = (rand() % 256) - 128;
    }
  }

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  return kTfLiteOk;
}
