import intel_extension_for_pytorch as ipex
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("intfloat/multilingual-e5-base")
model = ipex.optimize(model, dtype=torch.bfloat16)

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
d = torch.randint(vocab_size, size=[batch_size, seq_length])
model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
model = torch.jit.freeze(model)


@torch.inference_model()
def encode_text(inputs):
    return model(inputs)


with torch.cpu.amp.autocast(dtype=torch.bfloat16):
    encode_text(inputs)
