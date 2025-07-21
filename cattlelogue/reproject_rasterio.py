import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
source_dir = 'inputs/Cattle'
output_dir = 'inputs/Cattle_Reprojected'
target_path = 'inputs/GLW4-2020.D-DA.CTL.tif'

# Open target to get desired CRS and transform
with rasterio.open(os.path.join(BASE_DIR, target_path)) as target:
    target_crs = target.crs
    target_transform = target.transform
    target_shape = target.shape

# Open source to read data
for filename in tqdm(os.listdir(os.path.join(BASE_DIR, source_dir))):
    if not filename.endswith('.tif'):
        continue

    source_path = os.path.join(BASE_DIR, source_dir, filename)
    output_path = os.path.join(BASE_DIR, output_dir, filename)
    print(f"Reprojecting {source_path} to {output_dir}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(source_path) as src:
        source_crs = src.crs
        source_data = src.read(1)

        # Create destination array
        dst_data = np.empty(target_shape, dtype=src.dtypes[0])

        # Reproject
        reproject(
            source=source_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=source_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )
        print(f"Reprojected {source_path} to {output_path}")
        # Save result
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=dst_data.shape[0],
            width=dst_data.shape[1],
            count=1,
            dtype=dst_data.dtype,
            crs=target_crs,
            transform=target_transform
        ) as dst:
            dst.write(dst_data, 1)