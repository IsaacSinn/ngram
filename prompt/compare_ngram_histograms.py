import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# New imports for dev data processing
from data.charloader import load_chars_from_file
from vocab import START_TOKEN, END_TOKEN

# Load trained models
with open("ngram_model.pkl", "rb") as f:
    model_abs_discount = pickle.load(f)
with open("ngram_vanilla_model.pkl", "rb") as f:
    model_vanilla = pickle.load(f)

# Load dev data (optionally, limit to a few examples for visualization)
dev_data = load_chars_from_file("./data/dev")
dev_data = dev_data[:3]  # limit to first 3 lines for demonstration

# Get probability distribution for the selected context for each model
def get_probs(model, context):
    if context in model.logprob:
        log_probs = model.logprob[context]
        probs = np.exp(log_probs)
        return probs
    else:
        # If context not present, return an array of zeros
        vocab_size = len(model.vocab)
        return np.zeros(vocab_size)

# Function to plot histograms side-by-side for both models
def plot_histograms(indices, probs_abs, probs_vanilla, step_idx, token, line_no):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(indices, probs_abs, color='blue', alpha=0.7)
    plt.title(f"Line {line_no} Step {step_idx} - Absolute Discounting\nToken: {token}")
    plt.xlabel("Token index")
    plt.ylabel("Probability")
    
    plt.subplot(1, 2, 2)
    plt.bar(indices, probs_vanilla, color='green', alpha=0.7)
    plt.title(f"Line {line_no} Step {step_idx} - Vanilla N-gram\nToken: {token}")
    plt.xlabel("Token index")
    plt.ylabel("Probability")
    
    plt.tight_layout()
    plt.show(block=False)
    input("Press any key to continue...")
    plt.close()

# Iterate through each dev line and simulate stepping like dev_ngram()
for line_no, dev_line in enumerate(dev_data, start=1):
    # Prepare input sequence: prepend START_TOKEN and append END_TOKEN
    input_tokens = [START_TOKEN] + dev_line
    # Initialize models' state
    q_abs = model_abs_discount.start()
    q_vanilla = model_vanilla.start()
    # For each token in the input sequence
    for step_idx, token in enumerate(input_tokens, start=1):
        token_idx = model_abs_discount.vocab.numberize(token)
        
        # Step both models
        q_abs, p_abs = model_abs_discount.step(q_abs, token_idx)
        q_vanilla, p_vanilla = model_vanilla.step(q_vanilla, token_idx)
        
        # Convert log prob to probability for model_abs_discount (if necessary)
        if np.max(p_abs) < 0:  # assuming log scale if max < 0
            probs_abs = np.exp(p_abs)
        else:
            probs_abs = p_abs
        
        if np.max(p_vanilla) < 0:
            probs_vanilla = np.exp(p_vanilla)
        else:
            probs_vanilla = p_vanilla
        
        vocab_size = len(model_abs_discount.vocab)
        indices = np.arange(vocab_size)
        
        # Plot the histograms
        plot_histograms(indices, probs_abs, probs_vanilla, step_idx, token, line_no)
