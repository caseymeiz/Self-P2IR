import argparse
import os

import numpy as np
import vtk


def _add_data_arrays(data_holder, arrays):
    for name, data in arrays.items():
        array = np.asarray(data)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        n, components = array.shape
        vtk_data = vtk.vtkFloatArray()
        vtk_data.SetName(name)
        vtk_data.SetNumberOfComponents(components)
        vtk_data.SetNumberOfTuples(n)
        for i in range(n):
            vtk_data.SetTuple(i, array[i])
        data_holder.AddArray(vtk_data)


def write_pc(path, points_np, point_data):
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    for i in range(len(points_np)):
        points.InsertPoint(i, points_np[i])
        cells.InsertNextCell(1, [i])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(cells)
    _add_data_arrays(polydata.GetPointData(), point_data)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()
    print(path)


def write_surface(path, points_np, tris_np, point_data):
    points = vtk.vtkPoints()
    for i in range(len(points_np)):
        points.InsertNextPoint(points_np[i])

    tri_array = vtk.vtkCellArray()
    for tri in tris_np:
        vtk_tri = vtk.vtkTriangle()
        for i in range(3):
            vtk_tri.GetPointIds().SetId(i, int(tri[i]))
        tri_array.InsertNextCell(vtk_tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(tri_array)
    _add_data_arrays(polydata.GetPointData(), point_data)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="outputs/posenet_scene")
    args = parser.parse_args()

    arrays = np.load(os.path.join(args.scene_dir, "scene_arrays.npz"))
    write_surface(
        os.path.join(args.scene_dir, "mesh_before.vtk"),
        arrays["mesh_before"],
        arrays["faces"],
        {},
    )
    write_surface(
        os.path.join(args.scene_dir, "mesh_after.vtk"),
        arrays["mesh_after"],
        arrays["faces"],
        {},
    )
    write_pc(os.path.join(args.scene_dir, "point_cloud.vtk"), arrays["point_cloud"], {})


if __name__ == "__main__":
    main()
