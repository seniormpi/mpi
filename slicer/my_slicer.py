import sys
import os.path
import argparse

# import pyximport; pyximport.install()
from .slicermpi import Slicer
from .stl_data import StlData


def slice_new(stl_path, mat):
    stl = StlData()
    stl.read_file(stl_path)
    print("Read {0} ({4} facets, {1:.1f} x {2:.1f} x {3:.1f})".format(
        stl_path,
        stl.points.maxx - stl.points.minx,
        stl.points.maxy - stl.points.miny,
        stl.points.maxz - stl.points.minz,
        len(stl.facets),
    ))

    slicer = Slicer([stl])

    slicer.load_configs()

    outfile = os.path.splitext(stl_path)[0] + ".gcode"
    return slicer.slice_to_file(outfile)

# vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
