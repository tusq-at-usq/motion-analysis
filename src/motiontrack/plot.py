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
        pg.setConfigOptions(antialias=True)

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

    #  def update_projection(self, blob_data):
        #  blob_x = blob_data.points
        #  self.blob_pr.setData(blob_x[:,0], blob_x[:,1])
        #  self.app.processEvents()

    #  def update_assocation(self, pair_data):
        #  self.blob_ass.setData(pair_data[:,0], pair_data[:,1])
        #  self.app.processEvents()

    #  def update_mesh(self, mesh, angles):
        #  n_surf = mesh.shape[0]
        #  mesh = np.concatenate((mesh, mesh[:,0,:].reshape(n_surf,1,3)),axis=1)
        #  mesh = mesh.reshape(-1,3)
        #  connect = np.array([True, True, True, False]*n_surf)
        #  self.frame_pr.setData(mesh[:,0], mesh[:,1], connect=connect)
        #  self.app.processEvents()

    #  def close(self):
        #  self.view.close()

class PlotTrack:
    def __init__(self, x_dict):
        self.x_dict = x_dict

        # Set white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Create the main application instance
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(show=True, title="State tracking")

        self.p1 = self.win.addPlot(title="Position")
        self.p2 = self.win.addPlot(title="Orientation")

        self.win.nextRow()

        self.p3 = self.win.addPlot(title="Velocity")
        self.p4 = self.win.addPlot(title="Angular velocity")

        self.x = pg.PlotCurveItem(pen=pg.mkPen(1, width=1))
        self.y = pg.PlotCurveItem(pen=pg.mkPen(2, width=1))
        self.z = pg.PlotCurveItem(pen=pg.mkPen(3, width=1))

        self.q0 = pg.PlotCurveItem(pen=pg.mkPen(4, width=1))
        self.q1 = pg.PlotCurveItem(pen=pg.mkPen(5, width=1))
        self.q2 = pg.PlotCurveItem(pen=pg.mkPen(6, width=1))
        self.q3 = pg.PlotCurveItem(pen=pg.mkPen(7, width=1))

        self.u = pg.PlotCurveItem(pen=pg.mkPen(1, width=1))
        self.v = pg.PlotCurveItem(pen=pg.mkPen(2, width=1))
        self.w = pg.PlotCurveItem(pen=pg.mkPen(3, width=1))

        self.p = pg.PlotCurveItem(pen=pg.mkPen(8, width=1))
        self.q = pg.PlotCurveItem(pen=pg.mkPen(9, width=1))
        self.r = pg.PlotCurveItem(pen=pg.mkPen(10, width=1))

        self.x_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color=1),
                                              symbol='o', size=1)
        self.y_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color=2),
                                              symbol='o', size=1)
        self.z_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color=3),
                                              symbol='o', size=1)

        self.q0_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color=4),
                                              symbol='o', size=1)
        self.q1_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color=5),
                                              symbol='o', size=1)
        self.q2_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color=6),
                                              symbol='o', size=1)
        self.q3_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color=7),
                                              symbol='o', size=1)

        self.p1.addItem(self.x)
        self.p1.addItem(self.y)
        self.p1.addItem(self.z)
        self.p1.addItem(self.x_pr)
        self.p1.addItem(self.y_pr)
        self.p1.addItem(self.z_pr)

        self.p2.addItem(self.q0)
        self.p2.addItem(self.q1)
        self.p2.addItem(self.q2)
        self.p2.addItem(self.q3)
        self.p2.addItem(self.q0_pr)
        self.p2.addItem(self.q1_pr)
        self.p2.addItem(self.q2_pr)
        self.p2.addItem(self.q3_pr)

        self.p3.addItem(self.u)
        self.p3.addItem(self.v)
        self.p3.addItem(self.w)

        self.p4.addItem(self.p)
        self.p4.addItem(self.q)
        self.p4.addItem(self.r)

    def update_state(self, x, t):
        x = np.array(x).T
        t = np.array(t)
        self.x.setData(t, x[self.x_dict['x']])
        self.y.setData(t, x[self.x_dict['y']])
        self.z.setData(t, x[self.x_dict['z']])

        self.q0.setData(t, x[self.x_dict['q0']])
        self.q1.setData(t, x[self.x_dict['q1']])
        self.q2.setData(t, x[self.x_dict['q2']])
        self.q3.setData(t, x[self.x_dict['q3']])

        self.u.setData(t, x[self.x_dict['u']])
        self.v.setData(t, x[self.x_dict['v']])
        self.w.setData(t, x[self.x_dict['w']])

        self.p.setData(t, x[self.x_dict['p']])
        self.q.setData(t, x[self.x_dict['q']])
        self.r.setData(t, x[self.x_dict['r']])

        self.app.processEvents()

    def update_priori(self, x_, t):
        x_ = np.array(x_).T
        t = np.array(t)
        if x_.shape[1] > 1:
            self.x_pr.setData(t, x_[self.x_dict['x']])
            self.y_pr.setData(t, x_[self.x_dict['y']])
            self.z_pr.setData(t, x_[self.x_dict['z']])

            self.q0_pr.setData(t, x_[self.x_dict['q0']])
            self.q1_pr.setData(t, x_[self.x_dict['q1']])
            self.q2_pr.setData(t, x_[self.x_dict['q2']])
            self.q3_pr.setData(t, x_[self.x_dict['q3']])

    def update_observation(self, x, t):
        x = np.array(x).T
        t = np.array(t)
        if x.shape[1] > 1:
            self.x_pr.setData(t, x[0])
            self.y_pr.setData(t, x[1])
            self.z_pr.setData(t, x[2])

            self.q0_pr.setData(t, x[3])
            self.q1_pr.setData(t, x[4])
            self.q2_pr.setData(t, x[5])
            self.q3_pr.setData(t, x[6])

    def load_true_data(self, x_true, t):
        x_true = np.array(x_true).T
        t = np.array(t)

        self.p1.plot(t, x_true[self.x_dict['x']])
        self.p1.plot(t, x_true[self.x_dict['y']])
        self.p1.plot(t, x_true[self.x_dict['z']])

        self.p2.plot(t, x_true[self.x_dict['q0']])
        self.p2.plot(t, x_true[self.x_dict['q1']])
        self.p2.plot(t, x_true[self.x_dict['q2']])
        self.p2.plot(t, x_true[self.x_dict['q3']])

        self.p3.plot(t, x_true[self.x_dict['u']])
        self.p3.plot(t, x_true[self.x_dict['v']])
        self.p3.plot(t, x_true[self.x_dict['w']])

        self.p4.plot(t, x_true[self.x_dict['p']])
        self.p4.plot(t, x_true[self.x_dict['q']])
        self.p4.plot(t, x_true[self.x_dict['r']])

        self.app.processEvents()

    def close(self):
        self.view.close()