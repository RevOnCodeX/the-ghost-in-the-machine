import nbformat as nbf
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
NOTEBOOK_PATH = TASK3_DIR / "notebooks/task3_saliency.ipynb"

def main():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# Task 3: The Smoking Gun (Saliency Mapping)"))
    
    cells.append(nbf.v4.new_markdown_cell("""
## 1. Research Question & Experimental Setup
In Task 2, Tier C demonstrated that a RoBERTa-LoRA model can detect AI text with 99.4% ROC-AUC. 
But **why** does it make these predictions? Is it looking at structure, modern vocabulary, or something else?

This notebook uses **Integrated Gradients** (via Captum) to attribute the model's predictions back to the original text tokens.
"""))
    
    cells.append(nbf.v4.new_markdown_cell("## 2. Generate Prediction Audit & Select Examples"))
    cells.append(nbf.v4.new_code_cell("""
import sys
from pathlib import Path
import pandas as pd
import json

# Adjust path to import saliency modules
sys.path.append(str(Path.cwd().parent / "saliency"))

# Note: In a real run, you'd execute the pipeline. Here we load the results directly since the pipeline was run externally.
print("Loading selected examples...")
with open("../results/selected_examples.json", "r") as f:
    examples = json.load(f)

pd.DataFrame(examples).head()
"""))

    cells.append(nbf.v4.new_markdown_cell("## 3. Token & Word Level Attribution"))
    cells.append(nbf.v4.new_code_cell("""
token_df = pd.read_csv("../results/token_attributions.csv")
word_df = pd.read_csv("../results/word_attributions.csv")

print("Sample Word Attributions:")
word_df.head(10)
"""))

    cells.append(nbf.v4.new_markdown_cell("## 4. Attribution Stability & Validation"))
    cells.append(nbf.v4.new_markdown_cell("""
We performed two major sanity checks:
1. **Stability Check:** Does running Integrated Gradients with 100 steps yield the same token ranking as 50 steps?
2. **Deletion Check:** If we remove the top positively attributed tokens, does the AI probability drop?
"""))
    
    cells.append(nbf.v4.new_code_cell("""
stab_df = pd.read_csv("../results/attribution_stability.csv")
print("Stability Check (Spearman Correlation between 50 and 100 steps):")
print(stab_df['spearman_correlation'].mean())

val_df = pd.read_csv("../results/attribution_validation.csv")
print("\\nValidation Drop in AI Probability (Top Pos, Top Neg, Random):")
val_df.groupby('method')['probability_change'].mean()
"""))

    cells.append(nbf.v4.new_markdown_cell("## 5. Aggregate Author Level Results"))
    cells.append(nbf.v4.new_code_cell("""
top_tokens = pd.read_csv("../results/top_tokens.csv")
print("Top 10 AI-Supporting Tokens Across Dataset:")
print(top_tokens[['top_positive_words', 'mean_positive_attribution']].head(10))
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## 6. Interpretation & Limitations
**OBSERVATION:** The model strongly attributes positive AI predictions to modern adverbs, specific transition phrases, and the lack of stylistic artifacts.
**INTERPRETATION:** The classifier is heavily relying on modern vocabulary clustering rather than structural grammar alone.
**HYPOTHESIS:** Generative AI models struggle to fully suppress their latent modern vocabulary distribution, even when explicitly instructed to mimic a Victorian author.

**LIMITATIONS:**
Positive attribution indicates contribution toward the AI class for the specific model prediction. It does not establish that a token is inherently AI-generated universally.
"""))

    nb['cells'] = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, 'w') as f:
        nbf.write(nb, f)
        
    print(f"Generated {NOTEBOOK_PATH}")

if __name__ == "__main__":
    main()
