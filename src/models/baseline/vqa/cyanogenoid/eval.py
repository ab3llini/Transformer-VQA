import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities import paths
from models.baseline.vqa.cyanogenoid.model import Net

if __name__ == '__main__':
    checkpoint = paths.resources_path('models/baseline/vqa/cyanogenoid/2017-08-04_00.55.19.pth')

    model = Net()
