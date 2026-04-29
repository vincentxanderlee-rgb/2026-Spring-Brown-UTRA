import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# -------------------- Load Data --------------------
caption_dir = Path('/oscar/home/vxlee/data/vxlee/')
output_dir  = Path('/oscar/home/vxlee/scratch/')

nsd_caption = pd.read_csv(caption_dir / 'nsd_caption.csv')
print(f"Loaded {len(nsd_caption)} captions for {nsd_caption['nsdId'].nunique()} images")

# -------------------- Model Setup --------------------
MODEL_NAME = 'sentence-transformers/paraphrase-mpnet-base-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float16 if DEVICE == 'cuda' else torch.float32
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    output_hidden_states=True
).to(DEVICE)
model.eval()

NUM_LAYERS = model.config.num_hidden_layers  # 12
LAYERS_TO_EXTRACT = {
    'low':    2,
    'middle': NUM_LAYERS // 2,
    'top':    NUM_LAYERS
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
def encode_and_save(texts, batch_size: int = 64, max_length: int = 128):
    """Encode texts and save checkpoints to disk to avoid OOM."""
    all_vecs = {name: [] for name in LAYERS_TO_EXTRACT}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt'
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        for name, layer_idx in LAYERS_TO_EXTRACT.items():
            h = hidden_states[layer_idx]
            pooled = _mean_pool(h, inputs['attention_mask'])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vecs[name].append(pooled.cpu().float().numpy())

        torch.cuda.empty_cache()

        # Save checkpoint every ~20000 captions to free memory
        if (i // batch_size) % 312 == 311:
            print(f"  Saving checkpoint at {i}...")
            for name in LAYERS_TO_EXTRACT:
                chunk = np.vstack(all_vecs[name])
                np.save(output_dir / f'caption_embeddings_{name}_chunk_{i}.npy', chunk)
                all_vecs[name] = []

        if i % (batch_size * 20) == 0:
            print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts...")

    # Save remaining
    for name in LAYERS_TO_EXTRACT:
        if all_vecs[name]:
            chunk = np.vstack(all_vecs[name])
            np.save(output_dir / f'caption_embeddings_{name}_final.npy', chunk)

    print("Encoding complete! Merging chunks...")

    # Merge chunks one layer at a time to save memory
    for name in LAYERS_TO_EXTRACT:
        chunks = sorted(output_dir.glob(f'caption_embeddings_{name}_chunk_*.npy'))
        final  = output_dir / f'caption_embeddings_{name}_final.npy'

        all_chunks = [np.load(c) for c in chunks]
        if final.exists():
            all_chunks.append(np.load(final))

        merged = np.vstack(all_chunks)
        np.save(output_dir / f'caption_embeddings_{name}.npy', merged)
        print(f"  Merged {name}: {merged.shape}")

        for c in chunks:
            c.unlink()
        if final.exists():
            final.unlink()

        del merged, all_chunks

# -------------------- Run --------------------
texts   = nsd_caption['caption'].tolist()
nsd_ids = nsd_caption['nsdId'].tolist()

print("\nExtracting embeddings...")
encode_and_save(texts, batch_size=64, max_length=128)

pd.DataFrame({'nsdId': nsd_ids, 'caption': texts}).to_csv(
    output_dir / 'caption_metadata.csv', index=False
)
print("Done! Embeddings saved.")
