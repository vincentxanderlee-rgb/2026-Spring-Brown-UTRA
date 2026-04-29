import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- Load Metadata --------------------
output_dir = Path('/oscar/home/vxlee/scratch/')

meta    = pd.read_csv(output_dir / 'caption_metadata.csv')
nsd_ids = meta['nsdId'].values
print(f"Loaded metadata: {len(nsd_ids)} captions")

LAYERS_TO_EXTRACT = {
    'low':    2,
    'middle': 6,
    'top':    12
}

# -------------------- Cosine Similarity (Pairwise, One Layer at a Time) --------------------
def compute_similarities_for_layer(layer_name, layer_idx, embs, nsd_ids, chunk_size=500, n_sample_images=10000, sample_size=500000):
    """
    Within-image: all pairwise cosine similarities between captions of the same image.
    Across-image: pairwise cosine similarities between captions of DIFFERENT images
                  using a sample of images — NO averaging before similarity computation.
                  Uses running mean/std to avoid storing all pairs in memory.
    """
    nsd_ids_arr = np.array(nsd_ids)
    unique_ids  = np.unique(nsd_ids_arr)

    # Within-image: all pairwise caption similarities per image
    print(f"  Computing within-image pairwise similarities...")
    within_sims = []
    for nsd_id in unique_ids:
        idx = np.where(nsd_ids_arr == nsd_id)[0]
        if len(idx) < 2:
            continue
        vecs       = embs[idx].astype(np.float32)
        sim_matrix = vecs @ vecs.T
        upper      = sim_matrix[np.triu_indices(len(vecs), k=1)]
        within_sims.append(upper)
    within_sims = np.concatenate(within_sims).astype(np.float32)
    np.save(output_dir / f'within_sims_{layer_name}.npy', within_sims)
    print(f"  Within-image pairs: {len(within_sims)}")

    # Across-image: sample images to keep memory usage manageable
    print(f"  Computing across-image pairwise similarities...")
    if len(unique_ids) > n_sample_images:
        sampled_ids = np.random.choice(unique_ids, n_sample_images, replace=False)
        print(f"  Sampling {n_sample_images} images out of {len(unique_ids)}...")
    else:
        sampled_ids = unique_ids

    # Take first caption per image — preserves variance, no averaging collapse
    caption_indices = [np.where(nsd_ids_arr == nsd_id)[0][0] for nsd_id in sampled_ids]
    caption_embs    = embs[caption_indices].astype(np.float32)

    norms        = np.linalg.norm(caption_embs, axis=1, keepdims=True).clip(min=1e-9)
    caption_embs = caption_embs / norms

    # Free full embedding matrix — no longer needed
    del embs

    # Running mean/std to avoid storing all pairs in memory
    # Also collect a small sample for plotting
    across_sum    = 0.0
    across_sum_sq = 0.0
    across_count  = 0
    across_sample = []
    collected     = 0
    n             = len(caption_embs)

    for i in range(0, n, chunk_size):
        chunk     = caption_embs[i : i + chunk_size]
        sim_block = (chunk @ caption_embs.T).astype(np.float32)

        for row_idx in range(len(chunk)):
            global_idx = i + row_idx
            row = sim_block[row_idx, global_idx + 1:]
            across_sum    += row.sum()
            across_sum_sq += (row ** 2).sum()
            across_count  += len(row)

            # Collect sample for plotting
            if collected < sample_size:
                take = min(sample_size - collected, len(row))
                across_sample.append(row[:take])
                collected += take

        if i % 1000 == 0:
            print(f"    {i}/{n} images done...")

    across_sample = np.concatenate(across_sample).astype(np.float32)
    np.save(output_dir / f'across_sims_sample_{layer_name}.npy', across_sample)
    print(f"  Across-image pairs computed: {across_count}")

    del caption_embs

    across_mean = across_sum / across_count
    across_std  = np.sqrt(across_sum_sq / across_count - across_mean ** 2)

    return {
        'layer':                    layer_name,
        'layer_idx':                layer_idx,
        'within_image_mean_cosine': float(within_sims.mean()),
        'within_image_std_cosine':  float(within_sims.std()),
        'across_image_mean_cosine': float(across_mean),
        'across_image_std_cosine':  float(across_std),
        'n_within_pairs':           len(within_sims),
        'n_across_pairs':           across_count,
    }

# -------------------- Process One Layer at a Time --------------------
results = []

for layer_name, layer_idx in LAYERS_TO_EXTRACT.items():
    print(f"\nProcessing layer: {layer_name} (idx={layer_idx})")

    emb_path = output_dir / f'caption_embeddings_{layer_name}.npy'
    print(f"  Loading {emb_path}...")
    embs = np.load(emb_path)
    print(f"  Loaded: {embs.shape}")

    result = compute_similarities_for_layer(layer_name, layer_idx, embs, nsd_ids)
    results.append(result)

    del embs
    print(f"  Layer {layer_name} done.")

# -------------------- Save Results --------------------
summary_df = pd.DataFrame(results)
print("\n========== Results ==========")
print(summary_df.to_string(index=False))

summary_df.to_csv(output_dir / 'layer_similarity_summary.csv', index=False)

# -------------------- Plot Distributions --------------------
print("\nPlotting distributions...")

layer_names   = ['low', 'middle', 'top']
layer_labels  = ['Low (Layer 2)', 'Middle (Layer 6)', 'Top (Layer 12)']
colors_within = ['#2196F3', '#4CAF50', '#F44336']
colors_across = ['#90CAF9', '#A5D6A7', '#EF9A9A']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Cosine Similarity Distributions by Layer\n(paraphrase-mpnet-base-v2)', fontsize=14, fontweight='bold')

for ax, layer_name, label, c_within, c_across in zip(
        axes, layer_names, layer_labels, colors_within, colors_across):

    within = np.load(output_dir / f'within_sims_{layer_name}.npy')
    across = np.load(output_dir / f'across_sims_sample_{layer_name}.npy')

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
plt.savefig(output_dir / 'similarity_distributions_mpnet.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_dir / 'similarity_distributions_mpnet.png'}")
print("\nDone! All results saved.")
