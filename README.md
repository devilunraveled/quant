# Quantization
ANLP assignment based on quantization

## How to Run 

The assignment had 5 different type of quantizations, each can be run through the command `python task.py {i}` where i can be 1, 2, 3, 4, 5.

### Results

| Metric     | Manual Quantization |           | BitsAndBytes Quantization |             |             |
| ---------- | ------------------- | --------- |---------------------------|-------------|-------------|
|            | Whole (Task 1)      | Selective (Task 2) | 8-bit HF (Task 3)       | 4-bit HF (Task 4) | 4-bit NF4 (Task 5) |
|------------|---------------------|-----------|---------------------------|-------------|-------------|
| **Latency**| Before              | 45.58 ms  | 42.56 ms                  | 43.41 ms    | 42.74 ms    | 42.57 ms    |
|            | After               | 29.09 ms  | 30.41 ms                  | 31.03 ms    | 31.89 ms    | 16.91 ms    |
|------------|---------------------|-----------|---------------------------|-------------|-------------|
| **Memory** | Before              | 474.70 MB | 474.70 MB                 | 474.70 MB   | 474.70 MB   | 474.70 MB   |
|            | After               | 118.97 MB | 158.97 MB                 | 156.35 MB   | 115.85 MB   | 115.85 MB   |
|------------|---------------------|-----------|---------------------------|-------------|-------------|
| **Perplexity** | Before          | 30.07     | 30.07                     | 30.07       | 30.07       | 30.07       |
|            | After               | $\infty$ (Overflow) | 46,417,899.07   | 30.19       | 33.42       | 32.05       |
|------------|---------------------|-----------|---------------------------|-------------|-------------|
