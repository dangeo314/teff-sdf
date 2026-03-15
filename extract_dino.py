import os
import io
import sys
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import torch
import torchvision.transforms as T

sys.path.append('dino-vit-features')
from extractor import ViTExtractor

def extract_dino_features(
    image_zip='datasets/cars_128.zip',
    output_zip='datasets/in_the_wild/shapenetcars_dinov1_stride4_pca16_nomask_5k.zip',
    model_type='dino_vits8',
    stride=4,
    n_pca=3,
    device='cuda',
    batch_size=128,
    pca_batch=512,
):
    print("Loading DINO model...")
    extractor = ViTExtractor(model_type, stride, device=device)

    transform = T.Compose([
        T.Resize((128, 128)),
        T.Pad(2),  # pad 128 → 132
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    with zipfile.ZipFile(image_zip, 'r') as zf:
        all_fnames = sorted([f for f in zf.namelist()
                             if f.endswith('.png') or f.endswith('.jpg')])
    N = len(all_fnames)
    print(f"Found {N} images")

    def iter_feats(fnames, zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for i in range(0, len(fnames), batch_size):
                batch_fnames = fnames[i:i+batch_size]
                imgs = []
                for fname in batch_fnames:
                    with zf.open(fname) as f:
                        img = Image.open(f).convert('RGB')
                        imgs.append(transform(img))
                batch = torch.stack(imgs).to(device)
                with torch.no_grad():
                    # shape: [B, 1, num_patches, feat_dim]
                    feats = extractor.extract_descriptors(
                        batch, layer=11, facet='key', bin=False
                    )
                    feats = feats.squeeze(1).cpu().numpy()  # [B, P, D]
                for j in range(len(batch_fnames)):
                    yield feats[j]  # [P, D]

    # --- Pass 1: fit IncrementalPCA ---
    print("Pass 1: Fitting IncrementalPCA...")
    ipca = IncrementalPCA(n_components=n_pca)

    patch_buffer = []
    for feat in tqdm(iter_feats(all_fnames, image_zip), total=N):
        patch_buffer.append(feat)
        if len(patch_buffer) >= pca_batch:
            chunk = np.concatenate(patch_buffer, axis=0)  # [pca_batch*P, D]
            ipca.partial_fit(chunk)
            patch_buffer = []
    if patch_buffer:
        chunk = np.concatenate(patch_buffer, axis=0)
        ipca.partial_fit(chunk)

    # --- Normalization stats on sample ---
    print("Computing normalization stats...")
    sample_feats = []
    for i, feat in enumerate(iter_feats(all_fnames[:500], image_zip)):
        sample_feats.append(ipca.transform(feat))
        if i >= 499:
            break
    sample_feats = np.concatenate(sample_feats, axis=0)
    p1  = np.percentile(sample_feats, 1,  axis=0)  # [3]
    p99 = np.percentile(sample_feats, 99, axis=0)  # [3]

    # --- Pass 2: transform and save ---
    print("Pass 2: Transforming and saving...")
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)

    H = W = None
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_STORED) as zf_out:
        for fname, feat in tqdm(
            zip(all_fnames, iter_feats(all_fnames, image_zip)), total=N
        ):
            P, D = feat.shape
            if H is None:
                H = W = int(P ** 0.5)
                print(f"Spatial resolution: {H}x{W}")

            pca_feat = ipca.transform(feat)  # [P, 3]
            pca_feat = np.clip((pca_feat - p1) / (p99 - p1 + 1e-6) * 2 - 1, -1, 1)
            pca_feat = pca_feat.reshape(H, W, n_pca).astype(np.float32)  # [32, 32, 3] HWC

            npy_fname = fname.rsplit('.', 1)[0] + '.npy'
            buf = io.BytesIO()
            np.save(buf, pca_feat)
            zf_out.writestr(npy_fname, buf.getvalue())

    print(f"Done! Saved {N} files. Each shape: [{H}, {W}, {n_pca}]")

if __name__ == '__main__':
    extract_dino_features()