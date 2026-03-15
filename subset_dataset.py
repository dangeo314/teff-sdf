import os
import io
import json
import zipfile
import random
from tqdm import tqdm

def subset_dataset(
    input_zip='datasets/cars_128.zip',
    output_zip='datasets/cars_128_20k.zip',
    n=20000,
    seed=10086,
):
    random.seed(seed)

    with zipfile.ZipFile(input_zip, 'r') as zf_in:
        all_fnames = sorted([f for f in zf_in.namelist()
                             if f.endswith('.png') or f.endswith('.jpg')])
        print(f"Total images: {len(all_fnames)}")

        selected = sorted(random.sample(all_fnames, n))
        print(f"Sampling {len(selected)} images...")

        labels = {}
        if 'dataset.json' in zf_in.namelist():
            with zf_in.open('dataset.json') as f:
                data = json.load(f)
                if data['labels'] is not None:
                    labels = {x[0]: x[1] for x in data['labels']}

        os.makedirs(os.path.dirname(output_zip), exist_ok=True) if os.path.dirname(output_zip) else None
        with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_STORED) as zf_out:
            for fname in tqdm(selected):
                data = zf_in.read(fname)
                zf_out.writestr(fname, data)

            subset_labels = [[f, labels[f]] for f in selected if f in labels]
            metadata = {'labels': subset_labels if subset_labels else None}
            zf_out.writestr('dataset.json', json.dumps(metadata))

    print(f"Done! Saved {n} images to {output_zip}")

if __name__ == '__main__':
    subset_dataset()