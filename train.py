import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # input: [x, y] -> output: [score]

    def forward(self, x):
        return self.fc(x)

model = TinyNet()
model.eval()

dummy_input = torch.randn(1, 2)

torch.onnx.export(
    model,
    dummy_input,
    "tiny_beer_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17
)
