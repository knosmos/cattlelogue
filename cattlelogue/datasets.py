import netCDF4 as nc
import numpy as np
import cv2
import os
from rich import print
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
import ee

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

WORLDCEREAL_ID = "ESA/WorldCereal/2021/MODELS/v100"
HUMAN_MODIF_ID = "CSP/HM/GlobalHumanModification"
PASTURE_WCH_ID = "projects/global-pasture-watch/assets/ggc-30m/v1/cultiv-grassland_p"
SCALE_FACTOR = 10

""" LOCALLY STORED DATASETS """


def fourier(timeseries) -> tuple[np.ndarray, np.ndarray]:
    n = len(timeseries)
    freq = np.fft.fftfreq(n)
    fourier_transform = np.fft.fft(timeseries)
    return np.abs(fourier_transform), freq


def upscale_to_glw4(glw4_shape, cmip_data) -> np.ndarray:
    # Assuming cmip_data is a 3D array and glw4_shape is the target shape
    if glw4_shape[0] > cmip_data.shape[0] or glw4_shape[1] > cmip_data.shape[1]:
        cmip_data_resized = cv2.resize(
            cmip_data, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_CUBIC
        )
    else:
        cmip_data_resized = cv2.resize(
            cmip_data, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_AREA
        )
    return cmip_data_resized


def year_to_index(year) -> int:
    return (year - 2015) * 12


def process_timeseries_data(timeseries_fname, shape, year=2015) -> np.ndarray:
    start, end = year_to_index(year), year_to_index(year + 1)
    with nc.Dataset(timeseries_fname) as ds:
        for var_name, data in ds.variables.items():
            if var_name in timeseries_fname.split("/")[-1].split("_"):
                cmip_data = cv2.merge(data[start:end])
                cmip_data = np.flipud(cmip_data)
                cmip_data = np.roll(cmip_data, shift=540 // 2, axis=1)
                # cmip_data = fourier(cmip_data)[0]
                cmip_data_resized = upscale_to_glw4(shape, cmip_data)
                print(
                    f"Processed timeseries variable {var_name} with shape {cmip_data_resized.shape}",
                )
                return cmip_data_resized


def process_fixed_data(fixed_data, shape) -> np.ndarray:
    with nc.Dataset(fixed_data) as ds:
        for var_name, data in ds.variables.items():
            if var_name in fixed_data.split("/")[-1].split("_"):
                cmip_data = data[:]
                cmip_data = np.flipud(cmip_data)
                cmip_data = np.roll(cmip_data, shift=540 // 2)
                cmip_data_resized = upscale_to_glw4(shape, cmip_data)
                cmip_data_resized = cmip_data_resized[:, :, np.newaxis]
                print(
                    f"Processed fixed variable {var_name} with shape {cmip_data_resized.shape}"
                )
                return cmip_data_resized


def load_glw4_data() -> tuple[np.ndarray, tuple[int, int]]:
    glw4_path = os.path.join(BASE_PATH, LIVESTOCK_DENSITY_PATH)
    glw4_data = imread(glw4_path, key=3)
    # FAO plots the prime meridian at the center whereas CMIP data places it at the left edge
    glw4_data = np.roll(glw4_data, glw4_data.shape[1] // 2, axis=1)
    return glw4_data, glw4_data.shape


""" EARTH ENGINE DATASETS """


def initialize_earth_engine() -> None:
    try:
        ee.Authenticate()
        ee.Initialize()
        print("Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        raise


def load_worldcereal_data() -> np.ndarray:
    if os.path.exists(os.path.join(BASE_PATH, "inputs/worldcereal_data.npy")):
        print("Loading cached WorldCereal data from numpy file.")
        return np.load(os.path.join(BASE_PATH, "inputs/worldcereal_data.npy"))
    cereal_data = ee.ImageCollection(WORLDCEREAL_ID).select("classification").mosaic()
    cereal_npy = ee.data.computePixels(
        {
            "expression": cereal_data,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {
                    "width": 360 * SCALE_FACTOR,
                    "height": 180 * SCALE_FACTOR,
                },
                "affineTransform": {
                    "scaleX": 1 / SCALE_FACTOR,
                    "shearX": 0,
                    "shearY": 0,
                    "scaleY": -1 / SCALE_FACTOR,
                    "translateY": 90,
                },
            },
        }
    )
    data = cereal_npy["classification"]
    np.save(os.path.join(BASE_PATH, "inputs/worldcereal_data.npy"), data)
    print(f"WorldCereal data loaded with shape {data.shape}")
    return data


def load_human_modification_index() -> np.ndarray:
    if os.path.exists(os.path.join(BASE_PATH, "inputs/human_modification.npy")):
        print("Loading cached Human Modification Index data from numpy file.")
        return np.load(os.path.join(BASE_PATH, "inputs/human_modification.npy"))
    human_modification = ee.ImageCollection(HUMAN_MODIF_ID).select("gHM")
    human_modif_npy = ee.data.computePixels(
        {
            "expression": human_modification.first(),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {
                    "width": 360 * SCALE_FACTOR,
                    "height": 180 * SCALE_FACTOR,
                },
                "affineTransform": {
                    "scaleX": 1 / SCALE_FACTOR,
                    "shearX": 0,
                    "translateX": -180,
                    "shearY": 0,
                    "scaleY": -1 / SCALE_FACTOR,
                    "translateY": 90,
                },
                "crsCode": "EPSG:4326",
            },
        }
    )
    data = human_modif_npy["gHM"]
    np.save(os.path.join(BASE_PATH, "inputs/human_modification.npy"), data)
    print(f"Human Modification Index data loaded with shape {data.shape}")
    return data


def load_pasture_watch_data() -> np.ndarray:
    if os.path.exists(os.path.join(BASE_PATH, "inputs/pasture_watch.npy")):
        print("Loading cached Pasture Watch data from numpy file.")
        return np.load(os.path.join(BASE_PATH, "inputs/pasture_watch.npy"))
    pasture_watch = (
        ee.ImageCollection(PASTURE_WCH_ID)
        .select("probability")
        .filterDate("2015-01-01", "2016-01-01")
        .first()
    )
    pasture_watch = pasture_watch.mask(pasture_watch.gt(0))
    pasture_watch_npy = ee.data.computePixels(
        {
            "expression": pasture_watch,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {
                    "width": 360 * SCALE_FACTOR,
                    "height": 180 * SCALE_FACTOR,
                },
                "affineTransform": {
                    "scaleX": 1 / SCALE_FACTOR,
                    "shearX": 0,
                    "translateX": -180,
                    "shearY": 0,
                    "scaleY": -1 / SCALE_FACTOR,
                    "translateY": 90,
                },
            },
        }
    )
    data = pasture_watch_npy["probability"]
    np.save(os.path.join(BASE_PATH, "inputs/pasture_watch.npy"), data)
    print(f"Pasture Watch data loaded with shape {data.shape}")
    return data


def build_dataset(year=2015, process_ee=True) -> dict:
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

    # Process EE data
    if process_ee:
        initialize_earth_engine()

        worldcereal_data = load_worldcereal_data()
        worldcereal_data = upscale_to_glw4(glw4_shape, worldcereal_data)

        human_modification_index = load_human_modification_index()
        human_modification_index = upscale_to_glw4(glw4_shape, human_modification_index)

        pasture_watch_data = load_pasture_watch_data()
        pasture_watch_data = upscale_to_glw4(glw4_shape, pasture_watch_data)

        worldcereal_data = worldcereal_data.reshape(-1, 1)
        human_modification_index = human_modification_index.reshape(-1, 1)
        pasture_watch_data = pasture_watch_data.reshape(-1, 1)

    return {
        "features": features,
        "livestock_density": glw4_data.flatten(),
        "worldcereal_data": worldcereal_data if process_ee else None,
        "human_modification_index": human_modification_index if process_ee else None,
        "pasture_watch_data": pasture_watch_data if process_ee else None,
        "glw4_shape": glw4_shape,
    }
