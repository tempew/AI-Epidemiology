"""
SHAP Explanation of GPT-2 Output for a Chess Endgame Prompt - Position 2
This script attributes SHAP scores to input words in a chess endgame prompt,
based on how much each word contributed to GPT-2's confidence in its generated answer.
Designed for interpretability comparisons between SHAP and Logia.
Dependencies: torch, transformers, shap
"""
import warnings
warnings.filterwarnings("ignore", message="loss_type=None.*")
warnings.filterwarnings("ignore", message=".*loss_type.*")
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import shap
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
import random
import numpy as np

# ===============================
# Reproducibility (Optional)
# ===============================
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ===============================
# Model + Tokenizer Setup
# ===============================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ===============================
# Input Prompt - Position 2
# ===============================
prompt = (
    "White king on f6, Black king on e8, White queen on h4, Black queen on a8, "
    "White rook on b3, Black rook on c8, Black pawn on e3. White to move. "
    "What is the best move and why?"
)

# ===============================
# Generate GPT-2 Output
# ===============================
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )
generated_text = tokenizer.decode(
    generated[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True
)
print("\n GPT-2 Output:\n")
print(generated_text)

# ===============================
# SHAP Predict Function
# ===============================
def predict_log_prob(input_texts):
    scores = []
    for text in input_texts:
        input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
        output_ids = tokenizer(generated_text, return_tensors="pt").input_ids
        combined_ids = torch.cat([input_ids, output_ids], dim=1)
        with torch.no_grad():
            outputs = model(combined_ids, labels=combined_ids)
            loss = outputs.loss
            log_prob = -loss.item()
            scores.append(log_prob)
    return torch.tensor(scores).unsqueeze(1)

# ===============================
# Run SHAP Explainer
# ===============================
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(
    predict_log_prob,
    masker=masker,
    algorithm="permutation",
    output_names=["logP(output)"]
)
shap_values = explainer([prompt])

# ===============================
# Extract and aggregate to word level
# ===============================
tokens = shap_values.data[0]
scores = shap_values.values[0]

# Split original prompt into words for mapping
words = prompt.split()
word_scores = {word: 0.0 for word in words}

# Map tokens back to words
for i, (token, score) in enumerate(zip(tokens, scores)):
    token_str = str(token).replace('Ġ', ' ').strip()
    
    # Handle score format
    if hasattr(score, 'shape') and len(score.shape) > 0:
        score_val = float(score[0]) if score.shape[0] > 0 else 0.0
    else:
        score_val = float(score)
    
    # Find which word this token belongs to
    for word in words:
        if token_str.lower() in word.lower() or word.lower() in token_str.lower():
            word_scores[word] += score_val
            break

# ===============================
# Sort by importance (absolute value) and print
# ===============================
ranked_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n Word-Level SHAP Importances (Ranked by Absolute Importance):\n")

with open("shap_output_words_position2.txt", "w", encoding="utf-8") as f:
    f.write("### Prompt ###\n" + prompt + "\n\n")
    f.write("### GPT-2 Output ###\n" + generated_text + "\n\n")
    f.write("### Word-Level SHAP Importances (Ranked) ###\n")
    
    for rank, (word, score) in enumerate(ranked_words, 1):
        line = f"{rank:2d}. {word}: {score:+.4f}"
        print(line)
        f.write(line + "\n")


