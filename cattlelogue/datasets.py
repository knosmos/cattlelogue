import netCDF4 as nc
import numpy as np
import cv2
import os
from rich import print
from tifffile import imread

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TIMESERIES_NC_PATHS = [
    "inputs/ts_Amon_GISS-E2-1-G_ssp460_r1i1p1f2_gn_20150116-21001216.nc",
    "inputs/pr_Amon_GISS-E2-1-G_ssp460_r1i1p1f2_gn_20150116-21001216.nc",
    "inputs/prsn_Amon_GISS-E2-1-G_ssp460_r1i1p1f2_gn_20150116-21001216.nc",
    "inputs/huss_Amon_GISS-E2-1-G_ssp460_r1i1p1f2_gn_20150116-21001216.nc",
]
FIXEDDATA_NC_PATHS = ["inputs/orog_fx_CanESM5_ssp460_r1i1p1f1_gn.nc"]
LIVESTOCK_DENSITY_PATH = "inputs/GLW4-2020.D-DA.CTL.tif"
file_error_flag = False
for path in TIMESERIES_NC_PATHS + FIXEDDATA_NC_PATHS + [LIVESTOCK_DENSITY_PATH]:
    full_path = os.path.join(BASE_PATH, path)
    if not os.path.exists(full_path):
        print(
            f"I can't find the dataset file at {path}.",
            "Make sure the dataset is downloaded and placed in the correct directory.",
        )
        file_error_flag = True
if file_error_flag:
    raise FileNotFoundError(
        "One or more dataset files are missing. Please check the paths and ensure all files are present."
    )


def fourier(timeseries):
    n = len(timeseries)
    freq = np.fft.fftfreq(n)
    fourier_transform = np.fft.fft(timeseries)
    return np.abs(fourier_transform), freq


def upscale_to_glw4(glw4_shape, cmip_data):
    # Assuming cmip_data is a 3D array and glw4_shape is the target shape
    cmip_data_resized = cv2.resize(
        cmip_data, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return cmip_data_resized


def year_to_index(year):
    return (year - 2015) * 12


def process_timeseries_data(timeseries_fname, shape, year=2015):
    start, end = year_to_index(year), year_to_index(year + 1)
    with nc.Dataset(timeseries_fname) as ds:
        for var_name, data in ds.variables.items():
            if var_name in timeseries_fname.split("/")[-1].split("_"):
                cmip_data = cv2.merge(data[start:end])
                cmip_data = fourier(cmip_data)[0]
                cmip_data_resized = upscale_to_glw4(shape, cmip_data)
                print(
                    f"Processed timeseries variable {var_name} with shape {cmip_data_resized.shape}",
                )
                return cmip_data_resized


def process_fixed_data(fixed_data, shape):
    with nc.Dataset(fixed_data) as ds:
        for var_name, data in ds.variables.items():
            if var_name in fixed_data.split("/")[-1].split("_"):
                cmip_data = data[:]
                cmip_data_resized = upscale_to_glw4(shape, cmip_data)
                cmip_data_resized = cmip_data_resized[:, :, np.newaxis]
                print(
                    f"Processed fixed variable {var_name} with shape {cmip_data_resized.shape}"
                )
                return cmip_data_resized


def load_glw4_data():
    glw4_path = os.path.join(BASE_PATH, LIVESTOCK_DENSITY_PATH)
    glw4_data = imread(glw4_path, key=3)
    # FAO plots the prime meridian at the center whereas CMIP data places it at the left edge
    glw4_data = np.roll(glw4_data, glw4_data.shape[1] // 2, axis=1)
    return glw4_data, glw4_data.shape


def build_dataset(year=2015):
    """
    Builds a dataset for livestock density prediction using GLW4 data and CMIP6 timeseries data.
    The dataset includes feature vectors from CMIP6 timeseries data and livestock density data from GLW4.
    Timeseries data is processed to extract Fourier features. The feature vectors are rescaled to
    match the GLW4 grid shape.
    """

    glw4_data, glw4_shape = load_glw4_data()
    datasets = []

    # Process timeseries data
    for path in TIMESERIES_NC_PATHS:
        datasets.append(
            process_timeseries_data(
                os.path.join(BASE_PATH, path), glw4_shape, year=year
            )
        )

    # Process fixed data
    for path in FIXEDDATA_NC_PATHS:
        datasets.append(process_fixed_data(os.path.join(BASE_PATH, path), glw4_shape))

    datasets = [data.astype(np.float64) for data in datasets]
    merged_datasets = cv2.merge(datasets)
    feature_vector_size = merged_datasets.shape[2]
    features = merged_datasets.reshape(-1, feature_vector_size)
    return {
        "features": features,
        "livestock_density": glw4_data.flatten(),
        "glw4_shape": glw4_shape,
    }
