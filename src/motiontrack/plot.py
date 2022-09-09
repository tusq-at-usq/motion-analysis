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

        self.blob_ob = pg.ScatterPlotItem(pen=pg.mkPen(width=10, color='r'),
                                              symbol='o', size=1)
        self.blob_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=10, color='k'),
                                           symbol='o', size=1)
        #  self.frame_pr = pg.PlotCurveItem(pen=pg.mkPen(width=1))

        self.view.addItem(self.blob_ob)
        self.view.addItem(self.blob_pr)
        #  self.view.addItem(self.frame_pr)

    def update_observation(self,blob_x):
        self.blob_ob.setData(blob_x[0],blob_x[1])
        self.app.processEvents()

    def update_projection(self, blob_data):
        blob_x = blob_data.points
        self.blob_pr.setData(blob_x[:,0],blob_x[:,1])
        self.app.processEvents()

    def close(self):
        sys.exit()


#  plot = PlotMatch('test')
#  plot.update()
#  input()
#  plot.update()
#  input()
#  plot.close()

#  # Generate random points
#  n = 100
#  data1 = np.random.normal(size=(2, n))
#  data2 = np.random.normal(size=(2, n))
#  data3 = np.random.normal(size=(2, 10))

#  # Create the scatter plot and add it to the view
#  pos1 = [{'pos': data1[:, i]} for i in range(n)]
#  pos2 = [{'pos': data2[:, i]} for i in range(n)]
#  pos3 = [{'pos': data3[:, i]} for i in range(10)]
#  self.scatter2.setData(pos2)
#  self.outline.setData(data3[0],data3[1])


#  Convert data array into a list of dictionaries with the x,y-coordinates

#  now = pg.ptime.time()
#  print("Plot time: {} sec".format(pg.ptime.time() - now))

#  Gracefully exit the application
