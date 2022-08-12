""" Complete test module for motiontrack 
"""

from test_read_file import test_read_file
test_read_file()
print("Test module: 'read_data.py' :".ljust(60)+"SUCCESS")

from test_combine_observables import test_combine_observables
test_combine_observables()
print("Test module: 'combine_observables.py' :".ljust(60)+"SUCCESS")

from test_spatial_match import test_spatial_match
test_spatial_match()
print("Test module: 'spatial_match.py' :".ljust(60)+"SUCCESS")

from test_tracking_loop import test_tracking_loop
test_tracking_loop()
print("Test module: 'ukf.py' :".ljust(60)+"SUCCESS")

from test_geometry_projection import show_camera_viewpoints,\
    create_example_view_rotation
show_camera_viewpoints()
create_example_view_rotation()
print("Test module: 'geometry.py' :".ljust(60)+"SUCCESS")

