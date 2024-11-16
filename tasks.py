from src.model import Model
from src.evaluation import Evaluation

from transformers import AutoTokenizer
from datasets import load_dataset

import pprint
# Example usage
if __name__ == "__main__":
    import sys

    part = sys.argv[1]
    
    model_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_tokenizer.pad_token = model_tokenizer.eos_token

    dataset = load_dataset("wikimedia/wikipedia", '20231101.en', split='train')
    limited_dataset = dataset.select(range(3000))  # Limit to first 3000 samples

    # Tokenize the entire dataset in advance
    tokenized_dataset = limited_dataset.map(lambda x: model_tokenizer(x['text'], padding=True, truncation=True, return_tensors='pt', max_length = 512), batched=True)

    data = [ x['text'] for x in tokenized_dataset ]

    # Create model and evaluation instances
    # Load your pre-trained model and tokenizer
        
    model_instance = Model("gpt2", int(part))
    evaluation_instance = Evaluation(model_instance, tokenized_dataset, model_tokenizer)

    print(f"Dataset Samples: {len(tokenized_dataset)}")

    metrics_results = evaluation_instance.compute_metrics()
    
    pprint.pprint(metrics_results)
