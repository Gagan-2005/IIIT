from transformers import HubertModel

# Check HuBERT model configuration
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True)
config = model.config

print("HuBERT Configuration:")
print(f"  Model: facebook/hubert-base-ls960")
print(f"  Number of hidden layers: {config.num_hidden_layers}")
print(f"  Hidden size: {config.hidden_size}")
print(f"\nWith output_hidden_states=True:")
print(f"  - Returns {config.num_hidden_layers + 1} hidden states")
print(f"  - Index 0: Input embeddings")
print(f"  - Indices 1-{config.num_hidden_layers}: Layer outputs")
print(f"\nTotal layers in hidden_states tuple: {config.num_hidden_layers + 1}")
