import pandas as pd
import numpy as np
import torch
import json
import re
from transformers import AutoTokenizer, AutoModel, pipeline
from pathlib import Path

# -------------------- Config --------------------
output_dir  = Path('/oscar/home/vxlee/scratch/')
caption_dir = Path('/oscar/home/vxlee/data/vxlee/')

MODEL_NAME   = 'meta-llama/Llama-3.2-3B-Instruct'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE        = torch.float16 if DEVICE == 'cuda' else torch.float32

LAYERS_TO_EXTRACT = {
    'low':    2,
    'middle': 14,
    'top':    28
}

# -------------------- Load Captions --------------------
nsd_caption = pd.read_csv(caption_dir / 'nsd_caption.csv')
print(f"Loaded {len(nsd_caption)} captions for {nsd_caption['nsdId'].nunique()} images")

# -------------------- Load Embedding Model --------------------
print(f"\nLoading embedding model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

embed_model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    output_hidden_states=True,
    device_map='auto' if DEVICE == 'cuda' else None
).to(DEVICE)
embed_model.eval()

# -------------------- Load LLM Judge --------------------
print("Loading LLM judge pipeline...")
judge_pipe = pipeline(
    'text-generation',
    model=MODEL_NAME,
    torch_dtype=DTYPE,
    device_map='auto' if DEVICE == 'cuda' else None,
)

# -------------------- Helper Functions --------------------
def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask   = attention_mask.unsqueeze(-1).type_as(hidden_state)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.inference_mode()
def get_embeddings_all_layers(texts: list, batch_size: int = 32, max_length: int = 128) -> dict:
    """Return L2-normalized embeddings for each target layer."""
    layer_vecs = {name: [] for name in LAYERS_TO_EXTRACT}

    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_length, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = embed_model(**inputs)
        hidden_states = outputs.hidden_states

        for name, layer_idx in LAYERS_TO_EXTRACT.items():
            h      = hidden_states[layer_idx]
            pooled = _mean_pool(h, inputs['attention_mask'])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            layer_vecs[name].append(pooled.cpu().float().numpy())

    return {name: np.vstack(vecs) for name, vecs in layer_vecs.items()}

def cosine_similarity_score(recall_emb: np.ndarray, caption_embs: np.ndarray) -> float:
    """
    Compute cosine similarity between recall and each caption embedding,
    then average. Scale from [-1,1] to [0,10].
    """
    sims  = (caption_embs @ recall_emb).flatten()  # [n_captions]
    mean_sim = float(sims.mean())
    return round((mean_sim + 1) / 2 * 10, 2)  # scale to 0-10

def llm_judge_score(patient_recall: str, captions: list) -> tuple[float, str]:
    """
    Use Llama as a judge to score memory recall against reference captions.
    Returns (score 0-10, explanation).
    """
    captions_str = '\n'.join([f"- {c}" for c in captions])
    prompt = f"""You are evaluating a patient's memory recall of an image they were shown.

Reference captions describing the image:
{captions_str}

Patient's recall:
"{patient_recall}"

Score the patient's recall from 0 to 10 based on how accurately it reflects the image content.
- 0-2: Very poor, major details wrong or missing
- 3-4: Poor, some correct elements but significant errors
- 5-6: Moderate, captures general scene but misses key details
- 7-8: Good, most key details correct with minor omissions
- 9-10: Excellent, accurate and detailed recall

Respond ONLY with a JSON object in this exact format:
{{"score": <number 0-10>, "explanation": "<one sentence reason>"}}"""

    response = judge_pipe(
        prompt,
        max_new_tokens=100,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = response[0]['generated_text'][len(prompt):].strip()

    # Parse JSON response
    try:
        match = re.search(r'\{.*?\}', generated, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            score  = float(parsed['score'])
            explanation = parsed.get('explanation', '')
            return round(min(max(score, 0), 10), 2), explanation
    except Exception:
        pass

    # Fallback: try to extract just a number
    nums = re.findall(r'\b([0-9]|10)(?:\.\d+)?\b', generated)
    if nums:
        return round(float(nums[0]), 2), generated
    return 5.0, "Could not parse LLM response"

def score_recall_batch(recalls_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch of patient recalls.

    recalls_df must have columns:
        - nsdId: the image ID the patient was shown
        - patient_recall: the patient's free-text memory description

    Returns the input df with added columns for each layer's score,
    the LLM judge score, and the combined final score.
    """
    results = []

    # Get all unique nsdIds needed
    unique_ids = recalls_df['nsdId'].unique().tolist()

    # Filter captions to only relevant images
    relevant_caps = nsd_caption[nsd_caption['nsdId'].isin(unique_ids)].copy()

    # Embed all captions and all recalls at once
    all_caption_texts = relevant_caps['caption'].tolist()
    all_recall_texts  = recalls_df['patient_recall'].tolist()

    print(f"Embedding {len(all_caption_texts)} captions...")
    caption_embs = get_embeddings_all_layers(all_caption_texts)

    print(f"Embedding {len(all_recall_texts)} patient recalls...")
    recall_embs = get_embeddings_all_layers(all_recall_texts)

    print("Scoring recalls...")
    for idx, row in recalls_df.iterrows():
        nsd_id  = row['nsdId']
        recall  = row['patient_recall']

        # Get caption indices for this image
        cap_mask = relevant_caps['nsdId'].values == nsd_id
        captions = relevant_caps[cap_mask]['caption'].tolist()

        result = {
            'nsdId':          nsd_id,
            'patient_recall': recall,
        }

        # Cosine similarity score per layer
        for layer_name in LAYERS_TO_EXTRACT:
            cap_embs_layer = caption_embs[layer_name][cap_mask]
            rec_emb_layer  = recall_embs[layer_name][idx]
            cos_score = cosine_similarity_score(rec_emb_layer, cap_embs_layer)
            result[f'cosine_score_{layer_name}'] = cos_score

        # LLM judge score
        print(f"  LLM judging recall for nsdId={nsd_id}...")
        llm_score, explanation = llm_judge_score(recall, captions)
        result['llm_judge_score'] = llm_score
        result['llm_explanation'] = explanation

        # Combined final score per layer (average of cosine + LLM)
        for layer_name in LAYERS_TO_EXTRACT:
            combined = round((result[f'cosine_score_{layer_name}'] + llm_score) / 2, 2)
            result[f'final_score_{layer_name}'] = combined

        results.append(result)
        print(f"  nsdId={nsd_id} | "
              f"low={result['final_score_low']} | "
              f"middle={result['final_score_middle']} | "
              f"top={result['final_score_top']} | "
              f"llm={llm_score}")

    return pd.DataFrame(results)

# -------------------- Example Usage --------------------
if __name__ == '__main__':

    # Example batch of patient recalls
    # Replace this with your actual patient data
    patient_recalls = pd.DataFrame([
        {'nsdId': 22555, 'patient_recall': 'There was a room with blue walls and a sink near a door.'},
        {'nsdId': 8440,  'patient_recall': 'I remember seeing two cars, one of them was parked badly.'},
        {'nsdId': 34610, 'patient_recall': 'There was an airplane in the sky.'},
    ])

    print("\nScoring patient recalls...")
    scores_df = score_recall_batch(patient_recalls)

    print("\n========== Results ==========")
    print(scores_df[[
        'nsdId', 'patient_recall',
        'cosine_score_low', 'cosine_score_middle', 'cosine_score_top',
        'llm_judge_score',
        'final_score_low', 'final_score_middle', 'final_score_top',
        'llm_explanation'
    ]].to_string(index=False))

    scores_df.to_csv(output_dir / 'patient_recall_scores.csv', index=False)
    print(f"\nSaved to {output_dir / 'patient_recall_scores.csv'}")
