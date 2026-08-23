# The Ghost in the Machine - Literary Dataset

This repository contains a dataset comparing human-authored literary texts (Charles Dickens and Jane Austen) against AI-generated paragraphs written in a plain, neutral style and an author-mimicking style.

## Pipeline Architecture

The dataset is generated using a scalable multi-provider AI generation pipeline. It supports API provider fallback, quota management, and automatic retries.

### Provider Fallback System
The API router seamlessly shifts between API providers if one hits rate limits (429) or persistent errors. 
The active fallback priority order is:
1. **Gemini** (`gemini-flash-latest`)
2. **OpenRouter** (`nvidia/nemotron-3.5-lightning:free`, `thinkingmachines/inkling-small:free`)
3. **AgentRoute** (`gpt-5.6-sol`)

## Setup Instructions

### 1. Installation
Install the required dependencies using the provided `requirements.txt` (once generated) or using standard python libraries.
```bash
pip install python-dotenv
```

### 2. Configuration & API Keys
We securely use a `.env` file to store credentials. **Do not commit real API keys to GitHub.**

1. Copy the example configuration:
```bash
cp .env.example .env
```
2. Open `.env` and add your API keys:
```text
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
AGENTROUTE_API_KEY=your_agentroute_key
```

### 3. Running Generation

To execute the scalable multi-provider batch generation, simply run:
```bash
python scripts/generate_ai_variations.py
```

To validate the generated JSON dataset output:
```bash
python scripts/validate_dataset.py
```
