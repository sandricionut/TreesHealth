
import os
import sys
import arcgis
import arcpy


if __name__ == "__main__":

    path_aerial_image = r""
    path_out_training_ds = r""
    path_out_trained_model = r""
    path_vector_mapped_trees = r""
    path_vector_detected_trees = r""


    # Export training sample
    try:
        arcpy.ia.ExportTrainingDataForDeepLearning()
        print(arcpy.GetMessages())
    except arcpy.ExecuteError:
        print(arcpy.GetMessages())

    # Train deep learning model
    ds_trees = arcgis.learn.prepare_data(path=path_out_training_ds)
    m_trees = arcgis.learn.MaskRCNN(data=ds_trees, backbone="Restnet50")
    m_trees.lr_find()
    m_trees.fit()
    m_trees.save(path_out_trained_model)


#     Detect trees
    arcpy.ia.DetectObjectsUsingDeepLearning(in_raster=path_aerial_image,
                                            out_detected_objects=path_vector_detected_trees)




