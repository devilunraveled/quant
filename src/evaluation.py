import torch
import numpy as np
import torch.nn as nn
from alive_progress import alive_bar

class Evaluation:
    def __init__(self, model, dataset, tokenizer):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def compute_metrics(self):
        memory_before = self._memory_footprint()
        latency_before = self._inference_latency()

        # # Perform inference to compute perplexity
        # perplexity_before = self._compute_perplexity()

        # Quantize the model
        self.model.quantize()

        memory_after = self._memory_footprint()
        latency_after = self._inference_latency()
        
        perplexity_after = self._compute_perplexity()

        return {
            'Memory Before': f"{memory_before:.2f} MB",
            'Memory After': f"{memory_after:.2f} MB",
            'Latency Before': f"{latency_before:.2f} ms",
            'Latency After': f"{latency_after:.2f} ms",
            # 'Perplexity Before': f"{perplexity_before:.2f}",
            'Perplexity After': f"{perplexity_after:.2f}",
        }

    def _memory_footprint(self):
        param_memory_fp32 = sum(p.nelement() * p.element_size() for name, p in self.model.model.named_parameters() if name not in self.model.quantized)
        param_memory_int8 = sum(p.nelement() for name, p in self.model.model.named_parameters() if name in self.model.quantized)
        
        totalMem = param_memory_fp32 + param_memory_int8
        # Convert to MB
        totalMem = totalMem / 1024 / 1024
        return totalMem

    def _inference_latency(self):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # Extract a few samples from the dataset
        num_samples = 5
        sample_inputs, _ = self._get_sample_data(num_samples=num_samples)

        # Move samples to the same device as the model
        sample_inputs = sample_inputs.to(next(self.model.parameters()).device)

        # Record start time
        start_time.record()
        
        with torch.no_grad():
            _ = self.model(sample_inputs)
        
        # Record end time
        end_time.record()
        
        # Wait for all operations to finish
        torch.cuda.synchronize()
        
        elapsed_time = start_time.elapsed_time(end_time)  # Time in milliseconds

        return elapsed_time/num_samples

    def _get_sample_data(self, num_samples=5):
        """Extract a few samples from the dataset and prepare input."""
        indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        
        samples = [self.dataset[int(i)] for i in indices]
        
        # Extract input_ids from samples
        inputs = [torch.tensor(sample['input_ids']) for sample in samples]  # Convert to tensor

        # Stack inputs into a single tensor
        inputs_tensor = torch.stack(inputs)

        return inputs_tensor, None  # No targets needed here

    def _compute_perplexity(self):
        total_loss = 0.0
        total_datapoints = 0
        perplexity = 0.0

        with alive_bar(len(self.dataset)) as bar:
            with torch.no_grad():
                for data in self.dataset:
                    inputs = torch.tensor(data['input_ids']).to(next(self.model.parameters()).device)  # Move to device
                    targets = torch.cat((inputs[1:], torch.tensor([self.tokenizer.eos_token_id], device=inputs.device)))

                    # Get model outputs
                    outputs = self.model(inputs)

                    logits = outputs.logits

                    loss = self.loss_function(logits.view(-1, logits.size(-1)), targets.view(-1))

                    total_loss = loss.item()
                    
                    sentence_perplexity = np.exp(total_loss)
                    perplexity += sentence_perplexity  # Perplexity calculation
                    total_datapoints += 1

                    bar.text(f"Perplexity: {perplexity/total_datapoints:.2f}")
                    bar()

        return perplexity/len(self.dataset)
