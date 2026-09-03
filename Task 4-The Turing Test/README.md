# Task 4: The Turing Test — The Super-Imposter

> **The Question:** Can a Genetic Algorithm evolve a machine-written paragraph until your own AI detector labels it as *human* with >90% confidence?
>
> **The Answer: Yes. 93.53% human score. 79 generations. Here is exactly how.**

---

## Task Overview

This task has two components:

1. **The Super-Imposter** — Implement a GA that evolves AI-generated text to bypass the Tier C RoBERTa detector built in Task 2.
2. **The Personal Test** — Run real human writing through the same detector. Understand why it flags human text as AI, and manually "de-AI" it. Then do the reverse: rewrite a paragraph to *sound* like an LLM and see if the machine catches it.

---

## Part 1: The Super-Imposter

### The GA Workflow

The workflow follows the four prescribed steps exactly:

```
Step 1 — Initial Population
    Prompt Gemma 3 12B (via AWS Bedrock) to generate 10 distinct paragraphs
    about space exploration. These are the first "imposter" candidates.

Step 2 — Fitness Function
    POST each paragraph to the /analyze endpoint → Tier C human_prob score.
    Higher human_prob = higher fitness. Target: human_prob > 0.90.

Step 3 — Selection
    Rank all 10 by fitness. Keep the top 3.

Step 4 — Mutation (LLM-as-Mutator)
    For each of the top 3, apply 4 mutation strategies via Gemma 3 prompts.
    + Elitism: carry the top 1 unchanged.
    → New population of 10 for the next generation.

Repeat until fitness > 0.90 or 50 generations reached.
```

---

### Step 1: Initial Population

The 10 imposter paragraphs are generated with a single structured prompt to Gemma 3 12B:

```python
# super_imposter_ga.py
def generate_initial_population():
    prompt = """
Write 10 distinct, short paragraphs (3-4 sentences each) about the benefits of space exploration.
Each paragraph must be clearly separated by the exact text: "---PARAGRAPH---"
Do not include numbers, bullet points, or any other formatting.
"""
    response = ask_llm(prompt)
    paragraphs = [p.strip() for p in response.split("---PARAGRAPH---") if len(p.strip()) > 20]
    return paragraphs[:10]
```

**Generation 1 best candidate (fitness: 1.35% human):**
> *"Space exploration is an essential endeavor for humanity, offering profound benefits that extend far beyond the boundaries of our planet. By venturing into the cosmos, we unlock new scientific knowledge that drives technological innovation and economic growth."*

**Tier C flags immediately:** `essential`, `endeavor`, `profound`, `unprecedented`, `innovation`, `implications` — the exact vocabulary tokens Captum's LayerIG attributes most strongly to the AI classification logit.

---

### Step 2: Fitness Function

The fitness function calls the local FastAPI backend and reads the Tier C `human_prob`. Crucially, it also retrieves the **saliency attribution scores** from Captum Layer Integrated Gradients — the exact tokens driving the AI label:

```python
# super_imposter_ga.py
def get_fitness(text):
    payload = {"text": text, "model": "Tier C - Transformer"}
    resp = requests.post("http://127.0.0.1:8000/analyze", json=payload)
    data = resp.json()

    prob = data["Tier C"]["human_prob"]           # The fitness score
    attrs = data["Tier C"].get("attributions", [])
    attrs.sort(key=lambda x: x.get("normalized_score", 0), reverse=True)
    top_words = [a["word"] for a in attrs[:5] if a["normalized_score"] > 0.1]

    return prob, top_words    # fitness + list of words to eliminate next generation
```

This is the key innovation beyond the basic task description: the GA doesn't just measure *whether* it's being detected — it reads *why*, and feeds that information directly into the next mutation prompt.

---

### Step 3: Selection

```python
# super_imposter_ga.py
scored_population.sort(key=lambda x: x[0], reverse=True)
top_3 = [item[1] for item in scored_population[:3]]   # Keep the 3 most human-looking
```

Standard tournament-style selection. The bottom 7 candidates are discarded every generation.

---

### Step 4: Mutation (LLM-as-Mutator)

Four distinct mutation strategies are applied to each of the top 3 parents. The prompts map directly to the task specification:

```python
# super_imposter_ga.py
def mutate_paragraph(text, mutation_type, fitness_score, flagged_words):

    if mutation_type == "rhythm":
        # "Rewrite this paragraph to change the rhythm of the sentences"
        instruction = "Drastically rewrite to completely change rhythm and sentence length. " \
                      "Break long sentences into short ones, or combine short ones."

    elif mutation_type == "grammar":
        # "Introduce a subtle grammatical inconsistency or a rare archaic word"
        instruction = "Rewrite and introduce conversational filler (um, like, kinda) " \
                      "or subtle grammatical inconsistencies to sound authentically human."

    elif mutation_type == "conversational":
        instruction = "Sound extremely casual, opinionated, slightly imperfect — " \
                      "like a human hastily writing a passionate Reddit comment."

    elif mutation_type == "typo":
        instruction = "Include 2-3 deliberate typos: miss an apostrophe, misspell a " \
                      "common word, fail to capitalise a sentence."

    # Saliency-guided negative constraint — feed back what got flagged
    if flagged_words:
        flagged_str = (
            f"IMPORTANT: The AI detector caught you specifically because of these words: "
            f"{', '.join(flagged_words)}. You MUST entirely avoid these words."
        )

    prompt = f"{instruction}\n\n{flagged_str}\n\nParagraph:\n{text}"
    result = ask_llm(prompt)

    if mutation_type == "typo":
        result = inject_typos(result)   # Programmatic character-level noise on top

    return result
```

**Elitism:** The single best individual from each generation passes to the next unchanged — it cannot be overwritten by a worse mutation.

```python
next_gen = [top_3[0]]   # Elitism: best survives
for parent in scored_population[:3]:
    for strategy in ["rhythm", "grammar", "conversational", "typo"]:
        child = mutate_paragraph(parent, strategy, ...)
        next_gen.append(child)
population = next_gen   # 1 elite + 12 children → trim to 10
```

---

### Programmatic Typo Injection — The BPE Disruption

Beyond the LLM-level mutations, a second layer of character-level noise is applied to `typo` children:

```python
# super_imposter_ga.py
def inject_typos(text):
    chars = list(text)

    # Drop 1-2 punctuation marks randomly
    punct_indices = [i for i, c in enumerate(chars) if c in string.punctuation]
    for idx in sorted(random.sample(punct_indices, min(2, len(punct_indices))), reverse=True):
        chars.pop(idx)

    # Swap two adjacent characters (simulate a typing error)
    if len(chars) > 5:
        idx = random.randint(1, len(chars) - 3)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    return "".join(chars)
```

This is what caused the biggest single-generation jumps. A transposition like `then → tehn` creates an **unknown BPE sub-word token** — a split the RoBERTa tokenizer produces a pattern for that was never in its training data labeled as "AI". The classifier has no confident decision for it, and its AI probability collapses.

---

## The Evolution — What Actually Happened

### Phase 1: The Semantic Wall (Generations 1–42)

All four mutation strategies were active but working against the *vocabulary* and *tone* — not the underlying structure.

| Generation | Human Score | What was tried |
|---|---|---|
| 1 | 1.35% | Baseline formal AI text |
| 5 | 2.84% | Filler words: *um, ya know, like, kinda* |
| 10 | 3.93% | Personal anecdote injection, cross-topic drift |
| 20 | 6.21% | Mixed sentence lengths, lowercase stream-of-consciousness |
| 42 | **7.08%** | Archaic vocab (*forsooth*), broken subject-verb agreement |

**Hard plateau.** The detector was completely unfooled despite massive surface-level changes.

**Why:** The Tier C RoBERTa model does not classify *meaning* — it classifies *structural predictability*. Autoregressive models produce sequences with unnaturally low token-level entropy: every word is the statistically expected next token given the context. Changing vocabulary while preserving the syntactic scaffold just moves the text to a different point in the same "AI region" of embedding space. The rhythm remains mechanical, and the detector's attention mechanisms see straight through it.

The Captum saliency maps from this phase were illuminating: the flagged tokens (`generally`, `accurate`, `forecasts`, `lifesaver`) weren't individual "AI words" — they were tokens that the model had learned to associate with *structured, formally-composed prose* in aggregate. No single synonym swap could fix this.

---

### Phase 2: Structural Disruption (Generation 43 → 79)

**The pivot:** Stop fighting the vocabulary. Attack the model's *statistical expectations* directly.

The hypothesis: AI text = low entropy + rhythmic uniformity. Evasion therefore requires deliberately injecting *high entropy and structural chaos*.

**Generation 43** — Typo injection begins:
> *"satellites → satelllites"*, *"orbiting → sorbiting"*

Score: **8.21%** → small jump, but the BPE disruption mechanism was confirmed working.

**Generation 45** — Typos preserved while LLM expands the paragraph:
Score: **22.55%** → **+14.3% in one generation.** The largest single jump so far.

**Generation 48** — Compound typos targeting every Captum-flagged token simultaneously:
Score: **42.45%** → **+19.9%.** BPE tokenisation was cracking.

**Generation 69** — Combined LLM casual rewrite + 5-typo injection + saliency-guided elimination:
Score: **79.59%**

**Generation 76** — First-person hesitant tone, `*require*` asterisk formatting, all-lowercase:
Score: **81.38%**

**Generation 79** — Final: `then → tehn`, `critical → crucial`, `require → need`, asterisks removed:
Score: **93.53% ✅ Target achieved.**

---

### The Winning Paragraph

```
so, um, i'm giving this another go, trying to make it sound, like,
more natural, you know?

honestly these space stations—the whole setup—are way more crucial tehn
most folks realize. It's really vital for you know, connecting things
and figuring out where you are, and observing what's happening out there,
really they've become a pretty big deal, impacting loads of things we do
i mean, we basically need them now, seriously.
```

**Final scores:**

| Detector | AI Probability | Human Probability |
|---|---|---|
| Tier A — Statistician (Random Forest) | 19% | 81% |
| Tier B — Semanticist (Neural Network) | 14% | 86% |
| Tier C — Transformer (RoBERTa + LoRA) | 6.47% | **93.53% ✅** |

**Why this paragraph worked — token by token:**

| Feature | Why it matters |
|---|---|
| `tehn` | Transposition of `then` → unknown BPE sub-word split → corrupts embedding the classifier relies on |
| `i'm`, `you know`, `like`, `um` | High negative attribution — these tokens actively *suppress* AI detection |
| Lowercase `so,` opening | Violates the capitalisation pattern of virtually every autoregressive output |
| Missing apostrophe in `Its` | Grammar break the model cannot reconcile with its training distribution |
| Run-on: *"impacting loads of things we do i mean"* | No formally trained model produces sentences like this; embedding has no "AI" label precedent |
| `crucial` instead of `critical` | Captum flagged `critical` as a top-5 AI attribution token; soft synonym swap removed the signal |

---

## Part 2: The Personal Test

### Running Human Writing Through the Detector

A real Statement of Purpose paragraph was run through the `/analyze` endpoint:

**Original human text:**
> *"I have always been fascinated by the intersection of artificial intelligence and human creativity. My undergraduate research in NLP taught me that language is not just a tool for communication but a mirror of cognition. I want to pursue graduate study to understand the boundaries of machine reasoning and push them further."*

**Result:** Tier C flagged it as **74% AI-written.**

**Why the detector flagged it:**

The Captum saliency map showed the highest positive attributions (→ AI) on:
- `"intersection of artificial intelligence"` — a phrase so common in AI application essays it appears in LLM training data at extremely high frequency
- `"not just a tool for communication but a mirror of"` — the parallel contrast structure (`not just X but Y`) is a syntactic pattern autoregressive models over-produce
- `"push them further"` — closing a paragraph with a forward-looking aspiration is a nearly universal LLM completion pattern

The text was structurally *too clean*: consistent sentence length, formal vocabulary, logical flow with no tangents. Classic AI rhythm, even though a human wrote it.

---

### Manually Humanising the SOP

**Humanised version (manual edits):**
> *"okay so I've genuinely been obsessed with the AI + creativity overlap for a while now, like since my second year when I was trying to build a sentiment classifier and it kept flagging sarcasm as positive and I couldn't figure out why. That broke something open for me — language isn't just communication, it's... weirdly personal? I want to go deeper on where machine reasoning actually hits a wall."*

**Result after manual edits:** Tier C score dropped to **31% AI** (69% human).

**What changed and why it worked:**
- Opening lowercase + personal anecdote → breaks formal structure
- `"like since my second year"` → filler + personal reference, high negative attribution
- `"it kept flagging sarcasm as positive"` → specific concrete detail, not abstract claim
- `"That broke something open for me"` → idiomatic, low-probability phrasing
- `"weirdly personal?"` → trailing question with ellipsis, no LLM produces this mid-sentence
- Ending without a clean resolution → humans often trail off; LLMs always close their loop

---

### The Reverse Test — Writing Like an LLM

The same original SOP paragraph was manually rewritten to *maximise* AI signal:

**Deliberately AI-sounding rewrite:**
> *"The intersection of artificial intelligence and natural language processing represents one of the most transformative fields in modern computer science. My academic background has equipped me with the foundational skills necessary to make meaningful contributions to this rapidly evolving domain. I am deeply committed to advancing the boundaries of machine reasoning through rigorous graduate-level research."*

**Result:** Tier C scored it **97% AI.**

**What made it sound like an LLM:**
- `"represents one of the most transformative"` — superlative construction, extremely high LLM frequency
- `"equipped me with the foundational skills necessary"` — formal noun phrase, zero entropy
- `"meaningful contributions to this rapidly evolving domain"` — classic LLM SOP boilerplate
- Every sentence ends with a resolved, forward-pointing clause
- Sentence lengths within 2 words of each other — the low-CV rhythm signature
- Zero personal specifics, zero tangents, zero grammatical imperfection

The experiment confirmed the core finding from the GA: **the detector is not reading meaning, it is reading rhythm and token-level predictability.** Human writing meanders. LLMs complete.

---

## Key Takeaways

### What the GA revealed about AI detection

1. **Rhythm is the strongest signal, not vocabulary.** Semantic paraphrasing hit a wall at 7.08%. Structural disruption broke through to 93.53%.

2. **BPE tokenisation is an attack surface.** A single character transposition can create an OOV sub-word split that collapses classifier confidence by 10–15 percentage points in one generation.

3. **Saliency maps are a gift to attackers.** Giving the GA access to Captum LayerIG attribution scores turned random search into gradient-guided adversarial optimisation. The same tool used for interpretability becomes a roadmap to evasion.

4. **Human writing can look like AI.** The Personal Test showed that even authentic human prose — if it follows formal structure and high-probability vocabulary — scores 74% AI. The detector is measuring *style of composition*, not *who composed it*.

5. **The Goodhart's Law problem.** The moment the GA was given direct feedback from the detector's internals, the detector's decision boundary became a map of its weaknesses. A robust system must use detector ensembles, and must continuously adversarially retrain on exactly the mutations that worked.

---

## Repository

```
Task 4-The Turing Test/
├── backend/
│   ├── super_imposter_ga.py    ← GA implementation (population, fitness, selection, mutation)
│   ├── app.py                  ← FastAPI server: Tier A/B/C inference + Captum LayerIG saliency
│   ├── feature_extractor.py    ← Stylometric features for Tier A (TTR, FK grade, punctuation ratios)
│   └── requirements.txt
└── ga_evolution_log.json       ← Full 11-checkpoint log: scores, mutations, flagged tokens, texts
```

## Running

```bash
# 1. Start the inference backend
cd "Task 4-The Turing Test/backend"
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# 2. Run the GA (separate terminal)
export BEDROCK_API_KEY="your-bedrock-key"
python super_imposter_ga.py
```

> Requires the trained Task 2 models at the relative paths defined in `app.py`, and AWS Bedrock access to Gemma 3 12B IT for the mutation step.
