# This Script is based on https://github.com/TimNagle-McNaughton/Boulder_Identification
# Note: we used Python 2.7 with arcpy to run it

import arcpy
import os
import shutil
from arcpy.sa import *
from arcpy import env
import glob


def run(workspace, rast_to_process, output_folder):

    try:
        os.makedirs(workspace)
    except WindowsError:
        pass

    # ArcGIS setup
    arcpy.env.workspace = workspace
    arcpy.env.overwriteOutput = True
    # the minimum pixel iteration value (inclusive)
    rast_min = 70
    # the maximum pixel threshold value (not inclusive)
    rast_max = 210
    # the inverval between threshold values (e.g. min = 0, max = 100, step = 10 -> 10 files: 0, 10, 20, etc.)
    rast_step = 10
    # the maximum value for the 'dark' brightness interval
    dark_interval_max = 400
    # the maxmimum value for the 'medium interval'
    medium_interval_max = 500

    # make a directory for the junk files
    if not os.path.exists(os.path.join(output_folder, "junk\\")):
        os.makedirs(os.path.join(output_folder, "junk\\"))
    # set junk output strings
    junk_path = os.path.join(output_folder, "junk\\")

    # make a directory for the selected threshold files
    if not os.path.exists(os.path.join(output_folder, "threshold\\")):
        os.makedirs(os.path.join(output_folder, "threshold\\"))
    # set the threshold strings
    thresh_path = os.path.join(output_folder, "threshold\\")

    # make a directory for the final files
    if not os.path.exists(os.path.join(output_folder, "final\\")):
        os.makedirs(os.path.join(output_folder, "final\\"))
    # set the final strings
    final_path = os.path.join(output_folder, "final/")

    print("Setup complete... \n")

    # apply mean filter
    neighborhood = NbrCircle(20)
    avg20_rast = arcpy.sa.FocalStatistics(rast_to_process, neighborhood, "MEAN",  "")

    junk_name = os.path.splitext(os.path.basename(rast_to_process))[0]

    # save the output to the junk folder
    avg20_out = junk_name +  "_20_rad_average.tif"
    avg20_rast.save(avg20_out)

    print("20 pixel average complete...\n")

    dark_mask = arcpy.Raster(avg20_out) < (dark_interval_max + 1)
    medium_mask = (arcpy.Raster(avg20_out) > dark_interval_max) & (arcpy.Raster(avg20_out) < (medium_interval_max + 1))
    light_mask = arcpy.Raster(avg20_out) > medium_interval_max

    #save the output to the intermediate folder
    dark_mask_out = junk_name + "_dark_mask.tif"
    medium_mask_out = junk_name + "_medium_mask.tif"
    light_mask_out = junk_name + "_light_mask.tif"

    dark_mask.save(dark_mask_out)
    medium_mask.save(medium_mask_out)
    light_mask.save(light_mask_out)

    print("Dark, medium, and light masks generated...\n")

    # generate the brightness rasters
    dark = arcpy.Raster(dark_mask_out) * arcpy.Raster(rast_to_process)
    medium = arcpy.Raster(medium_mask_out) * arcpy.Raster(rast_to_process)
    light = arcpy.Raster(light_mask_out) * arcpy.Raster(rast_to_process)

    # save the output to the intermediate folder
    dark_out = junk_name + "_dark_zero.tif"
    medium_out = junk_name + "_medium_zero.tif"
    light_out = junk_name + "_light_zero.tif"
    dark.save(dark_out)
    medium.save(medium_out)
    light.save(light_out)

    print("Dark, medium, and light rasters generated...\n")

    # generate the null values in the rasters
    dark_mask_null = SetNull(dark_out, dark_out, "VALUE < 1")
    medium_mask_null = SetNull(medium_out, medium_out, "VALUE < 1")
    light_mask_null = SetNull(light_out, light_out, "VALUE < 1")

    # save the output
    dark_mask_null_out = junk_path + "null_dark.tif"
    medium_mask_null_out = junk_path + "null_medium.tif"
    light_mask_null_out = junk_path + "null_light.tif"
    dark_mask_null.save(dark_mask_null_out)
    medium_mask_null.save(medium_mask_null_out)
    light_mask_null.save(light_mask_null_out)

    print("Final dark, medium, and light rasters generated...\n")

    # apply a 2x2 range filter
    interval_list = [dark_mask_null_out, medium_mask_null_out, light_mask_null_out]
    for interval in interval_list:
        neighborhood = NbrRectangle(2)
        rast_range = arcpy.sa.FocalStatistics(interval, neighborhood, "RANGE", "")

        # make a better basename
        basename = arcpy.Describe(interval).basename[5:]

        # save the output #
        range_out = junk_path + basename + "_2x2_range.tif"
        rast_range.save(range_out)

        ### peform raster calculation ###
        # iteratively select 'boulder' pixels above the threshold set above
        for value in range(rast_min, rast_max, rast_step):
            calc_above = arcpy.Raster(range_out) > value

            # save the output
            calc_out = thresh_path + basename + "_" + str(value) + ".tif"
            calc_above.save(calc_out)

    print("Threshold pixel values for generated...\n")

    arcpy.CheckInExtension("Spatial");

    # get variables
    dark_upper_value = 160# Conservative
    dark_lower_value = 80# Liberal

    medium_upper_value = 160# Conservative
    medium_lower_value = 80# Liberal

    light_upper_value = 160# Conservative
    light_lower_value = 80# Liberal

    # select the type of minimum bounding geometry
    # RECTANGLE_BY_AREA, RECTANGLE_BY_WIDTH, CONVEX_HULL, CIRCLE, ENVELOPE
    user_selected_MBG = "RECTANGLE_BY_AREA"

    # generate list of the dark rasters
    arcpy.env.workspace = thresh_path
    dark_raster_list = arcpy.ListRasters("*dark*" + str(dark_upper_value) + "*")
    dark_raster_list.extend(arcpy.ListRasters("*dark*" + str(dark_lower_value) + "*"))

    # generate list of the medium rasters
    arcpy.env.workspace = thresh_path
    medium_raster_list = arcpy.ListRasters("*medium*" + str(medium_upper_value) + "*")
    medium_raster_list.extend(arcpy.ListRasters("*medium*" + str(medium_lower_value) + "*"))

    # generate list of the light rasters
    arcpy.env.workspace = thresh_path
    light_raster_list = arcpy.ListRasters("*light*" + str(light_upper_value) + "*")
    light_raster_list.extend(arcpy.ListRasters("*light*" + str(light_lower_value) + "*"))

    # generate list of polygons
    arcpy.env.workspace = junk_path
    # process_list = arcpy.ListFeatureClasses("*_s.shp")

    print("Selection of boulder values completed...")

    #generate another list of polygons
    arcpy.env.workspace = junk_path
    # one_list = arcpy.ListFeatureClasses("*_1.shp")

    print("Bounding geometry completed...")

    # generate a list of MBG polygons
    arcpy.env.workspace = final_path
    bound_list = arcpy.ListFeatureClasses("*_bd.shp")

    print("Area added...")

    # generate a list of the final files
    arcpy.env.workspace = final_path
    feat_list = arcpy.ListFeatureClasses("*_bd.shp")

    print("Points generated...")

    # generate a list of the final files
    arcpy.env.workspace = final_path
    file_list = arcpy.ListFeatureClasses("*_pnt.shp")

    # add XY points
    for points in file_list:
        arcpy.management.AddXY(points)

    print("Coordinates added...")


    arcpy.env.workspace = final_path
    file_list = arcpy.ListFeatureClasses()


    print("Final tables generated...")

if __name__ == '__main__':

    base_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\SSW-ROCK\rough\image'

    works = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Arcpy\test\rough'

    files = os.listdir(base_path)
    for file in files:
        if file.endswith(".png"):
            file_name = file.split(".")[0]
            rast_to_process = os.path.join(base_path, file)
            workspace = os.path.join(works, file_name)

            output_folder = os.path.join(works, file_name + '_output')
            run(workspace, rast_to_process, output_folder)

    #
    # workspace = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Arcpy\test\1'
    # rast_to_process = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\SSW-ROCK\flat\image\1.png'
    # # where your data will be saved
    # output_folder = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Arcpy\test\1_output'
    #
    # run(workspace, rast_to_process, output_folder)

