import argparse
import os

os.environ.setdefault("PARAVIEW_USE_OFFSCREEN", "1")
os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")

from paraview.simple import (
    GetActiveViewOrCreate,
    LegacyVTKReader,
    Render,
    ResetCamera,
    SaveScreenshot,
    Show,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--points-path", required=True)
    parser.add_argument("--png-path", required=True)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.png_path), exist_ok=True)

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [args.width, args.height]
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0

    mesh_path = args.mesh_path
    points_path = args.points_path
    mesh = LegacyVTKReader(FileNames=[mesh_path])
    mesh_display = Show(mesh, view)
    mesh_display.Representation = "Surface"
    mesh_display.DiffuseColor = [0.38, 0.63, 0.70]
    mesh_display.Opacity = 0.42

    points = LegacyVTKReader(FileNames=[points_path])
    points_display = Show(points, view)
    points_display.Representation = "Points"
    points_display.DiffuseColor = [0.95, 0.43, 0.18]
    points_display.PointSize = 1.0

    mesh.UpdatePipeline()
    points.UpdatePipeline()
    mesh_bounds = mesh.GetDataInformation().GetBounds()
    point_bounds = points.GetDataInformation().GetBounds()
    xmin = min(mesh_bounds[0], point_bounds[0])
    xmax = max(mesh_bounds[1], point_bounds[1])
    ymin = min(mesh_bounds[2], point_bounds[2])
    ymax = max(mesh_bounds[3], point_bounds[3])
    zmin = min(mesh_bounds[4], point_bounds[4])
    zmax = max(mesh_bounds[5], point_bounds[5])
    center = [
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        0.5 * (zmin + zmax),
    ]
    x_extent = xmax - xmin
    z_extent = zmax - zmin
    max_extent = max(x_extent, ymax - ymin, z_extent)
    aspect = view.ViewSize[0] / view.ViewSize[1]

    ResetCamera(view)
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = center
    view.CameraPosition = [center[0], center[1] - 3.0 * max_extent, center[2]]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelScale = 0.55 * max(z_extent, x_extent / aspect)

    Render(view)
    SaveScreenshot(args.png_path, view, ImageResolution=[args.width, args.height])
    print(args.png_path)


if __name__ == "__main__":
    main()
