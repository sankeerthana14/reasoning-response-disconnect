import os
import torch
import pandas as pd
import torch.nn.functional as F

def format_prompt(question, model):
    """
    This function formats a single prompt, such that, it is in the correct format to be sent to the model.
    NOTE: Eahc model has different chat template, but transformer lens accounts for it.
    """
    message = [{"role": "user", "content": question}]

    prompt = model.tokenizer.apply_chat_template(
        message,
        tokenize=False, # we want a string not a list of numbers cuz later tokenization is handled separately
        add_generation_prompt=True # adds a token entity s.t. the model knows when to respond
    )

    return prompt

def run_logit_lens(dataset, model, output_dir, model_name, dataset_name):
    """
    1. Format prompt
    2. Convert to Tokens
    3. Run through the model and save in the dict cache
    4. Grab the Residual Stream Activations
    5. Normalize the residual stream activations
    6. Multiply with unembedding matrix
    7. Convert to probabilities using softmax
    8. Collect the GT tokens for each prompt per layer
    9. Compare the probabilites and plto curve
    """
    
    # Initialising the output_dir
    save_dir = os.path.join(output_dir, model_name, dataset_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # creates parent folders

    results = []

    for idx, row in dataset.iterrows():

        # 1. Extracting the relevant values from the dataset
        qid = row['id']
        question = row['question']
        gt_answer = row['gt_answer']

        # 2. Format the Prompt
        prompt = format_prompt(question, model)

        # 3. Tokenization
        tokens = model.to_tokens(prompt)

        # 4. Sent through the model
        logits, cache = model.run_with_cache(tokens)

        # 5. Get Model's Predictions - logits are of size [batch, sequence_length, vocab_size]
        final_token_id = logits[0, -1, :].argmax().item()  # max logits in the final token, take the highest score and converts to proper number
        model_answer = model.to_string(final_token_id)

        # 6. GT Answers
        gt_tokens = model.to_tokens(gt_answer, prepend_bos=False)
        gt_token_id = gt_tokens[0,0].item()

        # 7. Implementing Logit Lens
        #------------------------------------------
        n_layers = model.cfg.n_layers
        gt_probs_per_layer, top_tokens_per_layer = [], []

        for layer in range(n_layers):
            # (a) Extracting Activations
            residual = cache["resid_post", layer][0, -1, :]

            # (b) Normalizing 
            residual_normalized = model.ln_final(residual)

            # (c) Multiplying with Unembedding Matrix (converts from activation space to logit space)
            layer_logits = model.unembed(residual_normalized.unsqueeze(0).unsqueeze(0))
            layer_logits = layer_logits[0,0,:]

            # (d) Convert scores to probabilites
            probs = F.softmax(layer_logits, dim=-1)

            # (e) Storing top tokens probabilites per layer
            top_token_id = probs.argmax().item()  # returns the index of the highest probability, since the index is the token_id
            top_tokens_per_layer.append(top_token_id)

            # (f) Look up the GT Answer's probability
            gt_prob = probs[gt_token_id].item()
            gt_probs_per_layer.append(gt_prob)

        # 8. Saving the Activation File
        activation_data = {
            "qid": qid,
            "prompt": prompt,
            "question": question,
            "gt_answer": gt_answer,
            "gt_token_id": gt_token_id,
            "gt_probs_per_layer": torch.tensor(gt_probs_per_layer),
            "model_answer": model_answer,
            "top_tokens_per_layer": torch.tensor(top_tokens_per_layer),
            "activations": torch.stack([
                cache["resid_post", layer][0, -1, :]
                for layer in range(n_layers)
            ])  # shape [n_layers, d_model]
        }
        torch.save(activation_data, os.path.join(save_dir, f"{qid}.pt"))

        # 9. Tracking results for CSV
        result = {
            "qid": qid,
            "question": question,
            "model_answer": model_answer,
            "gt_answer": gt_answer,
            "gt_token_id": gt_token_id,
            "gt_token_str": model.to_string(gt_token_id),
        }
        for l in range(n_layers):
            result[f"layer_{l}_top_token_id"] = top_tokens_per_layer[l]
            result[f"layer_{l}_top_token_str"] = model.to_string(top_tokens_per_layer[l])
            result[f"layer_{l}_gt_prob"] = gt_probs_per_layer[l]

        results.append(result)

        print(f"  [{idx+1}/{len(dataset)}] {qid} | "
              f"GT: {gt_answer[:30]} | Model: {model_answer[:30]} | "
              f"Peak prob: {max(gt_probs_per_layer):.3f} at layer "
              f"{gt_probs_per_layer.index(max(gt_probs_per_layer))}")

    # 10. Save CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, "logit_lens_results.csv"), index=False)
    print(f"INFO: Results saved to {save_dir}")
    return results_df
        


        

  
