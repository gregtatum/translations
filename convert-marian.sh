# Marian ONNX converter.
/builds/worker/tools/marian-dev/build/marian-conv \
  --from       data/onnx-model/model.npz.best-chrf.npz \
  --to         data/onnx-model/model.onnx \
  --vocabs     data/onnx-model/vocab.enit.spm \
  --gemm-type  float32 \
  --export-as  onnx-encode
