import torch
import torch.nn as nn
import torch.nn.functional as F

class W8A16LinearLayer(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.register_buffer("int8_weights", torch.zeros((output_features, input_features), dtype=torch.int8))
        self.register_buffer("scales", torch.zeros((output_features), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros((1, output_features), dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, inputs):
        converted_weights = self.int8_weights.to(inputs.dtype).to(inputs.device)  # Convert weights to match input dtype
        output = F.linear(inputs, converted_weights) * self.scales.to(inputs.device)
        if self.bias is not None:
            output += self.bias
        return output

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)
        scales = w_fp32.abs().max(dim=-1).values / 127
        int8_weights = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)
        
        self.int8_weights.copy_(int8_weights)
        self.scales.copy_(scales)
