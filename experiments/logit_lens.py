"""
Logit_lens.py
==================

What this code does:

1. Format the question into a prompt
2. Tokenize and run through the model with run_with_cache
3. Generate the model's actual output answer
4. Get the ground truth answer's first token ID
5. At each layer, apply layer norm → unembedding → softmax
6. Look up the probability of the ground truth token at each layer
7. Save the activation file as a .pt file
8. Return the results

"""