import os
import numpy as np
import ee

WATER_AVAIL_ID = "WRI/Aqueduct_Water_Risk/V4/future_annual"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def initialize_earth_engine():
    """
    Initialize the Earth Engine API.
    This function should be called before using any Earth Engine functions.
    """
    try:
        ee.Initialize()
        print("Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        raise

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
        slope = (water_avail_2050 - water_avail_2030) / (2050 - 2030)
        water_avail = water_avail_2030 + slope * (year - 2030)
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

# initialize_earth_engine()
# load_water_avail_data(2030)

import matplotlib.pyplot as plt
def plot_water_avail():
    """
    Plot the water avail data for a given year.
    """
    water_avail = load_water_avail_data(2100) - load_water_avail_data(1961)
    print("avg water delta:", np.mean(water_avail))
    plt.imshow(water_avail, cmap='viridis', origin='upper')
    # set vlim
    plt.clim(-100, 100)
    plt.colorbar(label='Water Availability Change')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

plot_water_avail()