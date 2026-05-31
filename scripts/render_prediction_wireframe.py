from paraview.simple import *


input_vtk = "snapshot/real-last/test/overlays/predicted_deformation_0000.vtk"
output_png = "snapshot/real-last/test/overlays/predicted_deformation_wireframe.png"

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1400, 1000]
view.Background = [1.0, 1.0, 1.0]

mesh = LegacyVTKReader(FileNames=[input_vtk])

surface = Show(mesh, view)
surface.Representation = "Surface"
surface.DiffuseColor = [0.0, 0.65, 0.25]
surface.Opacity = 0.22

wireframe = Show(mesh, view)
wireframe.Representation = "Wireframe"
wireframe.DiffuseColor = [0.0, 0.25, 0.08]
wireframe.LineWidth = 1.3

ResetCamera(view)
view.CameraParallelProjection = 1
view.OrientationAxesVisibility = 0

Render(view)
SaveScreenshot(output_png, view)
print(output_png)
