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
    "inputs/tasmax_Amon_CanESM5_ssp460_r1i1p1f1_gn_20150116-21001216.nc",
    "inputs/tasmin_Amon_CanESM5_ssp460_r1i1p1f1_gn_20150116-21001216.nc",
]
HISTORICAL_NC_PATHS = [
    "inputs/ts_Amon_E3SM-1-1-ECA_historical_r1i1p1f1_gr_19500116-20141216.nc",
    "inputs/pr_Amon_E3SM-1-1-ECA_historical_r1i1p1f1_gr_19500116-20141216.nc",
    "inputs/prsn_Amon_E3SM-1-1-ECA_historical_r1i1p1f1_gr_19500116-20141216.nc",
    "inputs/huss_Amon_E3SM-1-1-ECA_historical_r1i1p1f1_gr_19500116-20141216.nc",
    "inputs/tasmax_Amon_CanESM5_historical_r1i1p1f1_gn_19500116-20141216.nc",
    "inputs/tasmin_Amon_CanESM5_historical_r1i1p1f1_gn_19500116-20141216.nc",
]
FIXEDDATA_NC_PATHS = ["inputs/orog_fx_CanESM5_ssp460_r1i1p1f1_gn.nc"]
LIVESTOCK_DENSITY_PATH = "inputs/GLW4-2020.D-DA.CTL.tif"
HISTORICAL_LIVESTOCK_PREFIX = "inputs/Cattle_Reprojected/Cattl_"
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

CITY_DISTAN_ID = "Oxford/MAP/accessibility_to_cities_2015_v1_0"
LANDFORMS_ID = "CSP/ERGo/1_0/Global/ALOS_landforms"
WATER_AVAIL_ID = "WRI/Aqueduct_Water_Risk/V4/future_annual"

SCALE_FACTOR = 10

""" LOCALLY STORED DATASETS """


def fourier(timeseries) -> tuple[np.ndarray, np.ndarray]:
    n = len(timeseries)
    freq = np.fft.fftfreq(n)
    fourier_transform = np.fft.fft(timeseries)
    return np.abs(fourier_transform), freq


def time_average(timeseries) -> np.ndarray:
    return np.mean(timeseries, axis=2)


def upscale_to_glw4(glw4_shape, cmip_data) -> np.ndarray:
    # Assuming cmip_data is a 3D array and glw4_shape is the target shape
    cmip_data = cmip_data.astype(np.float32)
    if glw4_shape[0] > cmip_data.shape[0] or glw4_shape[1] > cmip_data.shape[1]:
        cmip_data_resized = cv2.resize(
            cmip_data, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_CUBIC
        )
    else:
        cmip_data_resized = cv2.resize(
            cmip_data, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_AREA
        )
    return cmip_data_resized


def year_to_index(year, st_year=2015) -> int:
    return (year - st_year) * 12


def process_timeseries_data(timeseries_fname, shape, year=2015, st_year=2015) -> np.ndarray:
    start, end = year_to_index(year, st_year), year_to_index(year + 1, st_year)
    with nc.Dataset(timeseries_fname) as ds:
        for var_name, data in ds.variables.items():
            if var_name in timeseries_fname.split("/")[-1].split("_"):
                cmip_data = cv2.merge(data[start:end])
                cmip_data = np.flipud(cmip_data)
                cmip_data = np.roll(cmip_data, shift=cmip_data.shape[1] // 2, axis=1)
                cmip_data = fourier(cmip_data)[0]
                # cmip_data = time_average(cmip_data)
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
                cmip_data = np.roll(cmip_data, shift=cmip_data.shape[1] // 2, axis=1)
                cmip_data_resized = upscale_to_glw4(shape, cmip_data)
                cmip_data_resized = cmip_data_resized[:, :, np.newaxis]
                print(
                    f"Processed fixed variable {var_name} with shape {cmip_data_resized.shape}"
                )
                return cmip_data_resized


def load_glw4_data(resolution=1) -> tuple[np.ndarray, tuple[int, int]]:
    glw4_path = os.path.join(BASE_PATH, LIVESTOCK_DENSITY_PATH)
    import tifffile
    glw4_data = imread(glw4_path, key=resolution)
    metadata = tifffile.TiffFile(glw4_path).pages[0].geotiff_tags
    print(metadata)
    # FAO plots the prime meridian at the center whereas CMIP data places it at the left edge
    # glw4_data = np.roll(glw4_data, glw4_data.shape[1] // 2, axis=1)
    return glw4_data, glw4_data.shape


def load_aglw_data(year, resolution=1) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load the AGLW data for a given year and resolution.
    The data is expected to be stored in a TIFF file.
    """
    aglw_path = os.path.join(BASE_PATH, f"{HISTORICAL_LIVESTOCK_PREFIX}{year}.tif")
    aglw_data = imread(aglw_path, key=0)

    # AGLW_TIE = [-168.83835765026407, 84.21705788620514, 0.0]
    # GLW4_TIE = [-180.0, 90.0, 0.0]
    # TIE_DIFF = [AGLW_TIE[0] - GLW4_TIE[0], AGLW_TIE[1] - GLW4_TIE[1], 0.0]
    # TIE_DIFF_PERCENT = [TIE_DIFF[0] / 360.0, TIE_DIFF[1] / 180.0, 0.0]
    # # so GLW4 and AGLW very annoyingly have different model tie points
    # # we need to adjust the AGLW data to match GLW4's tie points
    # aglw_data = np.roll(aglw_data, shift=int(aglw_data.shape[1] * TIE_DIFF_PERCENT[0]), axis=1)
    # aglw_data = np.roll(aglw_data, shift=-int(aglw_data.shape[0] * TIE_DIFF_PERCENT[1]), axis=0)
    
    import tifffile
    metadata = tifffile.TiffFile(aglw_path).pages[0].geotiff_tags
    print(metadata)
    # hack so my computer doesn't bluescreen
    #SHAPE = (540, 1080)
    #aglw_data = cv2.resize(aglw_data, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    print(f"Loaded AGLW data for year {year} with shape {aglw_data.shape}")
    return aglw_data, aglw_data.shape


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
        data = np.load(os.path.join(BASE_PATH, "inputs/worldcereal_data.npy"))
        # hack to rotate data
        data = np.roll(data, shift=data.shape[1] // 2, axis=1)
        return data
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


def load_city_distance_data() -> np.ndarray:
    if os.path.exists(os.path.join(BASE_PATH, "inputs/city_distance.npy")):
        print("Loading cached City Distance data from numpy file.")
        return np.load(os.path.join(BASE_PATH, "inputs/city_distance.npy"))
    city_distance = ee.Image(CITY_DISTAN_ID).select("accessibility")
    city_distance_npy = ee.data.computePixels(
        {
            "expression": city_distance,
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
    data = city_distance_npy["accessibility"]
    np.save(os.path.join(BASE_PATH, "inputs/city_distance.npy"), data)
    print(f"City Distance data loaded with shape {data.shape}")
    return data

def load_landforms_data() -> np.ndarray:
    if os.path.exists(os.path.join(BASE_PATH, "inputs/landforms.npy")):
        print("Loading cached Landforms data from numpy file.")
        return np.load(os.path.join(BASE_PATH, "inputs/landforms.npy"))
    landforms = ee.Image(LANDFORMS_ID).select("constant")
    landforms_npy = ee.data.computePixels(
        {
            "expression": landforms,
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
    data = landforms_npy["constant"]
    np.save(os.path.join(BASE_PATH, "inputs/landforms.npy"), data)
    print(f"Landforms data loaded with shape {data.shape}")
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


def load_water_avail_data(year) -> np.ndarray:
    """
    WRI only provides water avail data for the years 2030, 2050, and 2080.
    We therefore need to interpolate the data for the years in between,
    and extrapolate for the years before 2030 and after 2080.
    We will assume linear trend in both cases.

    We will assume "business as usual" scenario, since it matches
    most closely with the SSP4 scenario used in the CMIP6 data.

    The data is stored as a featurecollection, with data points
    for each hydrological basin. We need to rasterize this data
    in order to use it in our model.
    """
    
    files = [
        "inputs/water_avail_bau30_ba_x_r.npy",
        "inputs/water_avail_bau50_ba_x_r.npy",
        "inputs/water_avail_bau80_ba_x_r.npy",
    ]

    for file in files:
        if not os.path.exists(os.path.join(BASE_PATH, file)):
            print(f"File {file} does not exist. Downloading data...")
            os.makedirs(os.path.dirname(os.path.join(BASE_PATH, file)), exist_ok=True)

            dataset = ee.FeatureCollection(WATER_AVAIL_ID)
            props = {
                "2030": "bau30_ba_x_r",
                "2050": "bau50_ba_x_r",
                "2080": "bau80_ba_x_r",
            }

            for prop in props.values():
                img = dataset.reduceToImage(
                    properties=[prop],
                    reducer=ee.Reducer.mean(),
                )

                SCALE_FACTOR = 5
                pixels = ee.data.computePixels(
                    {
                        "expression": img,
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
                water_avail = np.array(pixels).reshape(
                    (180 * SCALE_FACTOR, 360 * SCALE_FACTOR)
                )
                np.save(os.path.join(BASE_PATH, f"inputs/water_avail_{prop}.npy"), water_avail)
            
            break
    
    # Interpolate and extrapolate the data
    water_avail_2030 = np.load(os.path.join(BASE_PATH, "inputs/water_avail_bau30_ba_x_r.npy"))["mean"]
    water_avail_2050 = np.load(os.path.join(BASE_PATH, "inputs/water_avail_bau50_ba_x_r.npy"))["mean"]
    water_avail_2080 = np.load(os.path.join(BASE_PATH, "inputs/water_avail_bau80_ba_x_r.npy"))["mean"]

    if year < 2030:
        # Extrapolate backwards
        # slope = (water_avail_2050 - water_avail_2030) / (2050 - 2030)
        # water_avail = water_avail_2030 + slope * (year - 2030)
        water_avail = water_avail_2030
    elif year > 2080:
        # Extrapolate forwards
        slope = (water_avail_2080 - water_avail_2050) / (2080 - 2050)
        water_avail = water_avail_2080 + slope * (year - 2080)
    else:
        # Interpolate between 2030 and 2050 or 2050 and 2080
        if year < 2050:
            slope = (water_avail_2050 - water_avail_2030) / (2050 - 2030)
            water_avail = water_avail_2030 + slope * (year - 2030)
        else:
            slope = (water_avail_2080 - water_avail_2050) / (2080 - 2050)
            water_avail = water_avail_2050 + slope * (year - 2050)
    
    return water_avail


""" RF FIRST STAGE RESULTS """


def load_rf_results(prefix) -> np.ndarray:
    """
    Load the results of the Random Forest model inference.
    The results are expected to be stored in a numpy file.
    """
    FILE_DIR = os.path.join(BASE_PATH, "outputs")
    outputs = {}
    for file in os.listdir(FILE_DIR):
        if file.startswith(prefix) and file.endswith(".npy"):
            file_path = os.path.join(FILE_DIR, file)
            year = int(file.split("_")[1].split(".")[0])
            outputs[year] = np.load(file_path)
    return outputs


def build_dataset(year=2015, process_ee=True, flatten=True) -> dict:
    """
    Builds a dataset for livestock density prediction using GLW4 data and CMIP6 timeseries data.
    The dataset includes feature vectors from CMIP6 timeseries data and livestock density data from GLW4.
    Timeseries data is processed to extract Fourier features. The feature vectors are rescaled to
    match the GLW4 grid shape.
    """

    # if year < 2015:
    #     glw4_data, glw4_shape = load_aglw_data(year)
    # else:
    glw4_data, glw4_shape = load_glw4_data(resolution=1)
    aglw_data = None
    if year <= 2015:
        aglw_data, _ = load_aglw_data(year, resolution=1)
        aglw_data = upscale_to_glw4(glw4_shape, aglw_data)
    datasets = []

    # Process timeseries data
    if year >= 2015:
        for path in TIMESERIES_NC_PATHS:
            datasets.append(
                process_timeseries_data(
                    os.path.join(BASE_PATH, path), glw4_shape, year=year
                )
            )
    else:
        for path in HISTORICAL_NC_PATHS:
            datasets.append(
                process_timeseries_data(
                    os.path.join(BASE_PATH, path), glw4_shape, year=year, st_year=1950
                )
            )

    # Process fixed data
    for path in FIXEDDATA_NC_PATHS:
        datasets.append(process_fixed_data(os.path.join(BASE_PATH, path), glw4_shape))

    datasets = [data.astype(np.float64) for data in datasets]
    merged_datasets = cv2.merge(datasets)
    feature_vector_size = merged_datasets.shape[2]
    if flatten:
        features = merged_datasets.reshape(-1, feature_vector_size)
    else:
        features = merged_datasets

    # Process EE data
    if process_ee:
        initialize_earth_engine()

        worldcereal_data = load_worldcereal_data()
        worldcereal_data = upscale_to_glw4(glw4_shape, worldcereal_data)

        human_modification_index = load_human_modification_index()
        human_modification_index = upscale_to_glw4(glw4_shape, human_modification_index)

        pasture_watch_data = load_pasture_watch_data()
        pasture_watch_data = upscale_to_glw4(glw4_shape, pasture_watch_data)

        city_distance_data = load_city_distance_data()
        print(f"City distance data shape: {city_distance_data.shape}")
        city_distance_data = upscale_to_glw4(glw4_shape, city_distance_data)

        landforms_data = load_landforms_data()
        landforms_data = upscale_to_glw4(glw4_shape, landforms_data)

        water_risk_data = load_water_avail_data(year)
        water_risk_data = upscale_to_glw4(glw4_shape, water_risk_data)

        if flatten:
            worldcereal_data = worldcereal_data.reshape(-1, 1)
            human_modification_index = human_modification_index.reshape(-1, 1)
            pasture_watch_data = pasture_watch_data.reshape(-1, 1)
            city_distance_data = city_distance_data.reshape(-1, 1)
            landforms_data = landforms_data.reshape(-1, 1)
            water_risk_data = water_risk_data.reshape(-1, 1)
        else:
            worldcereal_data = worldcereal_data[:, :, np.newaxis]
            human_modification_index = human_modification_index[:, :, np.newaxis]
            pasture_watch_data = pasture_watch_data[:, :, np.newaxis]
            city_distance_data = city_distance_data[:, :, np.newaxis]
            landforms_data = landforms_data[:, :, np.newaxis]
            water_risk_data = water_risk_data[:, :, np.newaxis] 

        # features = np.hstack((features, city_distance_data),)
        features = np.concatenate((features, city_distance_data, landforms_data, water_risk_data), axis=-1)

    ret = {
        "features": features,
        "livestock_density": glw4_data.flatten() if flatten else glw4_data,
        "worldcereal_data": worldcereal_data if process_ee else None,
        "human_modification_index": human_modification_index if process_ee else None,
        "pasture_watch_data": pasture_watch_data if process_ee else None,
        "glw4_shape": glw4_shape,
    }
    if aglw_data is not None:
        ret["aglw_data"] = aglw_data.flatten() if flatten else aglw_data
    else:
        ret["aglw_data"] = None
    return ret