"""
Just a test script for visualizing Earth Engine data (sanity check to determine transforms)
"""

import numpy as np
import matplotlib.pyplot as plt
import ee

ee.Authenticate()
ee.Initialize()

SCALE_FACTOR = 10

# HUMAN_MODIF_ID = "projects/global-pasture-watch/assets/ggc-30m/v1/cultiv-grassland_p"
# human_modification = (
#     ee.ImageCollection(HUMAN_MODIF_ID)
#     .select("probability")
#     .filterDate("2015-01-01", "2016-01-01")
#     .first()
# )
# human_modification = human_modification.mask(human_modification.gt(15))
# human_modif_npy = ee.data.computePixels(
#     {
#         "expression": human_modification,
#         "fileFormat": "NUMPY_NDARRAY",
#         "grid": {
#             "dimensions": {"width": 360 * SCALE_FACTOR, "height": 180 * SCALE_FACTOR},
#             "affineTransform": {
#                 "scaleX": 1 / SCALE_FACTOR,
#                 "shearX": 0,
#                 "translateX": -180,
#                 "shearY": 0,
#                 "scaleY": -1 / SCALE_FACTOR,
#                 "translateY": 90,
#             },
#             # "crsCode": "EPSG:4326",
#         },
#     }
# )

CITY_DISTAN_ID = "Oxford/MAP/accessibility_to_cities_2015_v1_0"
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

fig = plt.figure(figsize=(10, 5))
#human_modif_npy_band = human_modif_npy["probability"]
# human_modif_npy_band[human_modif_npy_band < 0] = -1  # Mask out invalid data
plt.imshow(
    data, cmap="turbo", interpolation="nearest", vmin=0, vmax=100
)
plt.show()
