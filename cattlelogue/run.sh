py -m cattlelogue.train --ground_truth pasture_watch_data --output pasture_model.joblib
py -m cattlelogue.train --ground_truth worldcereal_data --output crop_model.joblib

py -m cattlelogue.pasture_visual --start_year 2015 --end_year 2015 --model_path cattlelogue/outputs/pasture_model.joblib
py -m cattlelogue.pasture_visual --start_year 1961 --end_year 1961 --model_path cattlelogue/outputs/pasture_model.joblib

py -m cattlelogue.crop_visual --start_year 1961 --end_year 1961 --model_path cattlelogue/outputs/crop_model.joblib
py -m cattlelogue.crop_visual --start_year 2015 --end_year 2015 --model_path cattlelogue/outputs/crop_model.joblib

py -m cattlelogue.train_xg_livestock_hyperopt
py -m cattlelogue.livestock_visual_ada --start_year 1961 --end_year 1961 --model_path cattlelogue/outputs/livestock_model_2.joblib
py -m cattlelogue.eval