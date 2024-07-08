# Marian ONNX converter.
marian-conv \
  --from       data/onnx-model/model.npz.best-chrf.npz \
  --to         data/onnx-model/model.onnx \
  --vocabs     data/onnx-model/vocab.enit.spm.gz \
  --gemm-type  float32 \
  --export-as  onnx-encode
