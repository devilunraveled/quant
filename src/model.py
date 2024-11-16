import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from .custom_layer import W8A16LinearLayer

class Model(torch.nn.Module):
    def __init__(self, modelName, quantization):
        super().__init__()
        self.modelName = modelName
        self.model = AutoModelForCausalLM.from_pretrained(modelName).to('cuda')
        self.model.eval()
        self.quantization = quantization
        self.quantized = set()

    def quantize(self):
        if self.quantization == 1 :
            self.quantizeLayersManually(all = True)
        elif self.quantization == 2 :
            self.quantizeLayersManually(all = False)
        elif self.quantization == 3 :
            self.model = AutoModelForCausalLM.from_pretrained(self.modelName, quantization_config = BitsAndBytesConfig(load_in_8bit=True))
        elif self.quantization == 4 :
            self.model = AutoModelForCausalLM.from_pretrained(self.modelName, quantization_config = BitsAndBytesConfig(load_in_4bit=True))
        elif self.quantization == 5 : 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Enable 4-bit quantization
                bnb_4bit_quant_type="nf4",  # Use NF4 quantization instead of linear
                bnb_4bit_use_double_quant=True,  # Double quantization can help with accuracy
                bnb_4bit_compute_dtype=torch.float16  # Use float16 for computation
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.modelName, quantization_config = bnb_config)

        self.model.eval()

    def _quantize_tensor(self, tensor):
        # Scale to 8-bit integers
        scale = torch.max(torch.abs(tensor)) / 127  # Max value for int8
        quantized_tensor = torch.round(tensor / scale).clamp(-128, 127).to(torch.float32)
        return quantized_tensor

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def quantizeLayersManually(self, all = False):
        self.quantized.clear()
        specialLayers = set(['attn', 'proj'])
        restrictedLayers = set(['wte' 'wpe', 'ln_',])
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.quantized.add(name)
                if ( all or any(layer in name for layer in specialLayers) ) and not any(layer in name for layer in restrictedLayers):
                    param.requires_grad = False
                    param.data = self._quantize_tensor(param.data)

    def mutateLinearLayers(self, module, target_layer):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                self.quantized.add(name)
                # Get old weights and bias
                old_weights = child.weight.data.detach().clone()
                old_bias = child.bias.data.detach().clone() if child.bias is not None else None
                
                # Create new W8A16 layer
                new_module = target_layer(child.in_features, child.out_features, bias=(old_bias is not None))
                
                # Quantize old weights and assign to new module
                new_module.quantize(old_weights)
                
                # Replace the old linear layer with the new one
                setattr(module, name, new_module)

                if old_bias is not None:
                    new_module.bias.copy_(old_bias)  # Assign bias if it exists
            
            else:
                self.mutateLinearLayers(child, target_layer)
