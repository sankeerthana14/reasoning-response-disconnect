from transformer_lens import HookedTransformer

def load_model(model_config, device):
    model = HookedTransformer.from_pretrained(model_config['path'], device=device)
    print(f"INFO: Successfully loaded {model_config['name']}!")
    print(f"INFO: Number of Layers: {model.cfg.n_layers}")
    print(f"INFO: Dimension of the Residual Stream: {model.cfg.d_model}") # represents the size of each token at each layer
    print(f"INFO: Device: {device}")
    return model