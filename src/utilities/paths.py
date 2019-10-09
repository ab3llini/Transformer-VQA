import os
import re


def __robust_respath_search():
    """
    Resolve the path for resources from anywhere in the code.
    :return: The real path of the resources
    """
    curpath = os.path.realpath(__file__)
    basepath = curpath
    while os.path.split(basepath)[1] != 'src':
        newpath = os.path.split(basepath)[0]
        if newpath == basepath:
            print("ERROR: unable to find source from path " + curpath)
            break
        basepath = os.path.split(basepath)[0]
    return os.path.join(os.path.split(basepath)[0], "resources")


# ######### RESOURCES DIRECTORIES DEFINITION ###########

RESPATH = __robust_respath_search()
MODELS_FOLDER = 'models'
DATA_FOLDER = 'data'


def resources_path(*paths):
    """
    Very base function for resources path management.
    Return the complete path from resources given a sequence of directories
    eventually terminated by a file, and makes all necessary subdirectories
    :param paths: a sequence of paths to be joined starting from the base of resources
    :return: the complete path from resources (all necessary directories are created)
    """
    p = os.path.join(RESPATH, *paths)
    if os.path.splitext(p)[1] != '':
        basep = os.path.split(p)[0]
    else:
        basep = p
    os.makedirs(basep, exist_ok=True)
    return p


# ############################## BASE DIRECTORY-RELATIVE PATHS ###############

def models_path(*paths):
    """
    Builds the path starting where all model data should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(MODELS_FOLDER, *paths)


def data_path(*paths):
    return resources_path(DATA_FOLDER, *paths)


if __name__ == '__main__':
    print(resources_path('models', 'bert', 'runs'))
