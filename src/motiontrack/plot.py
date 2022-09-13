import sys

import numpy as np
import pyqtgraph as pg

class PlotMatch:
    def __init__(self, name):

        # Set white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Create the main application instance
        self.app = pg.mkQApp()

        # Create the view
        self.view = pg.PlotWidget()
        self.view.resize(800, 600)
        self.view.setWindowTitle(name)
        self.view.setAspectLocked(True)
        self.view.show()

        self.blob_ob = pg.ScatterPlotItem(pen=pg.mkPen(width=20, color='r'),
                                              symbol='o', size=1)
        self.blob_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=10, color='k'),
                                           symbol='o', size=1)

        #  self.blob_ass = pg.PlotCurveItem(pen=pg.mkPen(width=1), connect='pairs')
        self.frame_pr = pg.PlotCurveItem(pen=pg.mkPen(width=1))
    
        self.view.addItem(self.blob_ob)
        self.view.addItem(self.blob_pr)
        #  self.view.addItem(self.blob_ass)
        self.view.addItem(self.frame_pr)

    def update_observation(self,blob_x):
        self.blob_ob.setData(blob_x[0], blob_x[1])
        self.app.processEvents()

    def update_projection(self, blob_data):
        blob_x = blob_data.points
        self.blob_pr.setData(blob_x[:,0], blob_x[:,1])
        self.app.processEvents()

    #  def update_assocation(self, pair_data):
        #  self.blob_ass.setData(pair_data[:,0], pair_data[:,1])
        #  self.app.processEvents()

    def update_mesh(self, mesh, angles):
        n_surf = mesh.shape[0]
        mesh = np.concatenate((mesh, mesh[:,0,:].reshape(n_surf,1,3)),axis=1)
        mesh = mesh.reshape(-1,3)
        connect = np.array([True, True, True, False]*n_surf)
        self.frame_pr.setData(mesh[:,0], mesh[:,1], connect=connect)
        self.app.processEvents()

    def close(self):
        self.view.close()


