import numpy as np
import glob

# datacube = np.load('DataCubes/09_05_2025/data_cube_20250509_151112.npy', allow_pickle=True)

# print(datacube.shape)

list = sorted(glob.glob('DataCubes/**/*.npy', recursive=True))
print(list)

# # Gets the datacubes list from files and put them in a list for combobox to view
# def datacube_getter():
#     list = os.listdir('/DataCubes')
#     print(list)