import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# -------------------- Load Data --------------------
caption_dir = Path('/oscar/home/vxlee/data/vxlee/')
output_dir  = Path('/oscar/home/vxlee/scratch/')

nsd_caption = pd.read_csv(caption_dir / "nsd_caption.csv")
print(f"Loaded {len(nsd_caption)} captions for {nsd_caption['nsdId'].nunique()} images")

# -------------------- Subset Sampling --------------------
N_SUBSET = 100
sample_ids  = nsd_caption["nsdId"].unique()
sample_ids  = sample_ids[:N_SUBSET]
nsd_caption = nsd_caption[nsd_caption["nsdId"].isin(sample_ids)].reset_index(drop=True)
print(f"Using subset: {len(nsd_caption)} captions for {nsd_caption['nsdId'].nunique()} images")

# -------------------- Model Setup --------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Using device: {DEVICE}")

print("Step 1: Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Step 2: Loading Model...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    output_hidden_states=True,
    device_map="auto" if DEVICE == "cuda" else None,
    local_files_only=True
).to(DEVICE)
print("Step 3: Model Loaded Successfully!")

NUM_LAYERS = model.config.num_hidden_layers  # 28 for Llama 3.2 3B
LAYERS_TO_EXTRACT = {
    "low":    2,
    "middle": NUM_LAYERS // 2,
    "top":    NUM_LAYERS - 1
}
print(f"Total transformer layers: {NUM_LAYERS}")
print(f"Extracting layers: {LAYERS_TO_EXTRACT}")

# -------------------- Helper Functions --------------------
def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.inference_mode()
def encode_texts_all_layers(texts, batch_size: int = 32, max_length: int = 128) -> dict:
    """Encode texts and return L2-normalized embeddings for each target layer."""
    layer_vecs = {name: [] for name in LAYERS_TO_EXTRACT}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        for name, layer_idx in LAYERS_TO_EXTRACT.items():
            h = hidden_states[layer_idx]
            pooled = _mean_pool(h, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            layer_vecs[name].append(pooled.cpu().float().numpy())

        torch.cuda.empty_cache()

        if i % (batch_size * 20) == 0:
            print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts...")

    return {name: np.vstack(vecs) for name, vecs in layer_vecs.items()}

# -------------------- Extract Embeddings --------------------
texts   = nsd_caption["caption"].tolist()
nsd_ids = nsd_caption["nsdId"].tolist()

print("\nExtracting embeddings...")
layer_embeddings = encode_texts_all_layers(texts, batch_size=32, max_length=128)

# -------------------- Save Embeddings Immediately --------------------
print("\nSaving embeddings...")
for layer_name, embs in layer_embeddings.items():
    np.save(output_dir / f"caption_embeddings_{layer_name}.npy", embs)
pd.DataFrame({"nsdId": nsd_ids, "caption": texts}).to_csv(output_dir / "caption_metadata.csv", index=False)
print("Embeddings saved!")

# -------------------- Cosine Similarity --------------------
def compute_cosine_similarities(layer_embeddings, nsd_ids, chunk_size=500, n_sample_images=5000, sample_size=500000):
    """
    Within-image: all pairwise cosine similarities between captions of the same image.
    Across-image: all pairwise cosine similarities between captions of DIFFERENT images
                  at the caption level — NO averaging before similarity computation.
                  This follows mentor's advice: compute all pairwise similarities first, then average.
    """
    nsd_ids_arr = np.array(nsd_ids)
    unique_ids  = np.unique(nsd_ids_arr)
    results     = []

    for layer_name, embs in layer_embeddings.items():
        print(f"\nComputing similarities for layer: {layer_name}")

        # ---- Within-image: all pairwise caption similarities per image ----
        print(f"  Computing within-image pairwise similarities...")
        within_sims = []
        for nsd_id in unique_ids:
            idx = np.where(nsd_ids_arr == nsd_id)[0]
            if len(idx) < 2:
                continue
            vecs       = embs[idx].astype(np.float32)       # [n_captions, D]
            sim_matrix = vecs @ vecs.T                       # [n_captions, n_captions]
            upper      = sim_matrix[np.triu_indices(len(vecs), k=1)]
            within_sims.append(upper)
        within_sims = np.concatenate(within_sims).astype(np.float32)
        np.save(output_dir / f"within_sims_{layer_name}.npy", within_sims)
        print(f"  Within-image pairs: {len(within_sims)}")

        # ---- Across-image: pairwise caption similarities between different images ----
        # Use one caption per image to preserve per-image variance (no averaging)
        print(f"  Computing across-image pairwise similarities...")
        if len(unique_ids) > n_sample_images:
            sampled_ids = np.random.choice(unique_ids, n_sample_images, replace=False)
            print(f"  Sampling {n_sample_images} images out of {len(unique_ids)}...")
        else:
            sampled_ids = unique_ids

        # Take first caption per image — preserves variance, no collapsing
        caption_indices = [np.where(nsd_ids_arr == nsd_id)[0][0] for nsd_id in sampled_ids]
        caption_embs    = embs[caption_indices].astype(np.float32)  # [n_images, D]

        # Re-normalize after indexing
        norms        = np.linalg.norm(caption_embs, axis=1, keepdims=True).clip(min=1e-9)
        caption_embs = caption_embs / norms

        # Compute all pairwise similarities in chunks — no averaging before similarity
        across_sims   = []
        across_sample = []
        collected     = 0
        n             = len(caption_embs)

        for i in range(0, n, chunk_size):
            chunk     = caption_embs[i : i + chunk_size]
            sim_block = (chunk @ caption_embs.T).astype(np.float32)  # [chunk, n_images]

            for row_idx in range(len(chunk)):
                global_idx = i + row_idx
                # Upper triangle only — avoids self-similarity and duplicate pairs
                row = sim_block[row_idx, global_idx + 1:]
                across_sims.append(row)

                # Collect sample for plotting
                if collected < sample_size:
                    take = min(sample_size - collected, len(row))
                    across_sample.append(row[:take])
                    collected += take

            if i % 1000 == 0:
                print(f"    {i}/{n} images processed...")

        across_sims   = np.concatenate(across_sims).astype(np.float32)
        across_sample = np.concatenate(across_sample).astype(np.float32)
        np.save(output_dir / f"across_sims_sample_{layer_name}.npy", across_sample)
        print(f"  Across-image pairs: {len(across_sims)}")

        results.append({
            "layer":                    layer_name,
            "layer_idx":                LAYERS_TO_EXTRACT[layer_name],
            "within_image_mean_cosine": float(within_sims.mean()),
            "within_image_std_cosine":  float(within_sims.std()),
            "across_image_mean_cosine": float(across_sims.mean()),
            "across_image_std_cosine":  float(across_sims.std()),
            "n_within_pairs":           len(within_sims),
            "n_across_pairs":           len(across_sims),
        })
        print(f"  Layer {layer_name} complete.")

    return pd.DataFrame(results)

summary_df = compute_cosine_similarities(layer_embeddings, nsd_ids)
print("\n========== Results ==========")
print(summary_df.to_string(index=False))

summary_df.to_csv(output_dir / "layer_similarity_summary.csv", index=False)

# -------------------- Plot Distributions --------------------
print("\nPlotting distributions...")

layer_names   = ['low', 'middle', 'top']
layer_labels  = [
    'Low (Layer 2)',
    f'Middle (Layer {NUM_LAYERS // 2})',
    f'Top (Layer {NUM_LAYERS - 1})'
]
colors_within = ['#2196F3', '#4CAF50', '#F44336']
colors_across = ['#90CAF9', '#A5D6A7', '#EF9A9A']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Cosine Similarity Distributions by Layer\n(Llama-3.2-3B-Instruct)', fontsize=14, fontweight='bold')

for ax, layer_name, label, c_within, c_across in zip(
        axes, layer_names, layer_labels, colors_within, colors_across):

    within = np.load(output_dir / f"within_sims_{layer_name}.npy")
    across = np.load(output_dir / f"across_sims_sample_{layer_name}.npy")

    ax.hist(within, bins=80, alpha=0.7, color=c_within,
            label=f'Within-image\n(μ={within.mean():.3f}, σ={within.std():.3f})', density=True)
    ax.hist(across, bins=80, alpha=0.7, color=c_across,
            label=f'Across-image\n(μ={across.mean():.3f}, σ={across.std():.3f})', density=True)

    ax.axvline(within.mean(), color=c_within, linestyle='--', linewidth=1.5)
    ax.axvline(across.mean(), color=c_across,  linestyle='--', linewidth=1.5)

    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Cosine Similarity', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 1.0)

plt.tight_layout()
plt.savefig(output_dir / 'similarity_distributions_llama.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_dir / 'similarity_distributions_llama.png'}")
print("\nDone! All files saved.")
