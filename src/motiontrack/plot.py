import sys
import numpy as np
import PyQt6
from pyqtgraph import PlotWidget, plot, QtCore
import pyqtgraph as pg

class PlotMatch:
    def __init__(self, name, resolution=None):

        # Set white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major') # best performance

        # Create the main application instance
        self.app = pg.mkQApp()
        #  self.app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(antialias=True)

        # Create the view
        self.view = pg.PlotWidget()

        self.y = 1024
        if resolution != None:
            self.view.resize(resolution[0], resolution[1])
            self.view.setXRange(0, resolution[0])
            self.view.setYRange(0, resolution[1])
            self.y = resolution[1]

        self.view.setWindowTitle(name)
        self.view.setAspectLocked(True)
        self.view.show()

        self.blob_ob = pg.ScatterPlotItem(pen=pg.mkPen(width=20, color='r'),
                                              symbol='o', size=1)
        self.blob_pr = pg.ScatterPlotItem(pen=pg.mkPen(width=10, color='b'),
                                           symbol='o', size=1)

        self.blob_CoM = pg.ScatterPlotItem(pen=pg.mkPen(width=15, color='g'),
                                           symbol='o', size=1)

        self.frame_pr = pg.PlotCurveItem(pen=pg.mkPen(width=1))
        self.edges = pg.PlotCurveItem(pen=pg.mkPen(width=2, color='orange'))

        self.image = pg.ImageItem()

        self.view.addItem(self.image)
        self.view.addItem(self.blob_ob)
        self.view.addItem(self.blob_pr)
        self.view.addItem(self.blob_CoM)
        self.view.addItem(self.frame_pr)
        self.view.addItem(self.edges)

    def invert(self,Y):
        # Invert Y axis from downwards-positive openCV axis system
        Y_ = -1*Y + self.y
        return Y_

    def update_CoM(self,blob_cent):
        self.blob_CoM.setData(blob_cent[0], blob_cent[1])
        self.app.processEvents()

    def update_observation(self,blob_x):
        self.blob_ob.setData(blob_x[0], self.invert(blob_x[1]))
        self.app.processEvents()

    def update_projection(self, blob_data):
        blob_x = blob_data.points
        self.blob_pr.setData(blob_x[:,0], self.invert(blob_x[:,1]))
        self.app.processEvents()

    def update_mesh(self, mesh, angles):
        mesh_dim = mesh.shape[2]
        n_surf = mesh.shape[0]
        mesh = np.concatenate((mesh, mesh[:,0,:].reshape(n_surf,1,mesh_dim)),axis=1)
        mesh = mesh.reshape(-1,mesh_dim)
        connect = np.array([True, True, True, False]*n_surf)
        self.frame_pr.setData(mesh[:,0], self.invert(mesh[:,1]), connect=connect)
        self.app.processEvents()

    def update_edges(self, line_st, line_en):

        #  np.ravel([A,B],'F')
        x = np.ravel((line_st[:,0],line_en[:,0]),'F')
        y = np.ravel((line_st[:,1],line_en[:,1]),'F')

        connect = np.array([True, False]*int(len(x)/2))
        #  self.edges.setData(x,self.invert(y),connect=connect[:-1])    
        self.edges.setData(x,self.invert(y),connect=connect)    
        self.app.processEvents()

    def update_image(self, im):
        self.image.setImage(np.flip(im,axis=0))
        #  self.image.setImage((im))
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


        sm_pen = pg.mkPen(0.2, style=QtCore.Qt.PenStyle.DashLine, width=1)
        #  sm_pen = pg.mkPen(0.2, style=Qt.Qt.DashLine, width=1)
        self.x_sm = pg.PlotCurveItem(pen=sm_pen)
        self.y_sm = pg.PlotCurveItem(pen=sm_pen)
        self.z_sm = pg.PlotCurveItem(pen=sm_pen)

        self.q0_sm = pg.PlotCurveItem(pen=sm_pen)
        self.q1_sm = pg.PlotCurveItem(pen=sm_pen)
        self.q2_sm = pg.PlotCurveItem(pen=sm_pen)
        self.q3_sm = pg.PlotCurveItem(pen=sm_pen)

        self.u_sm = pg.PlotCurveItem(pen=sm_pen)
        self.v_sm = pg.PlotCurveItem(pen=sm_pen)
        self.w_sm = pg.PlotCurveItem(pen=sm_pen)

        self.p_sm = pg.PlotCurveItem(pen=sm_pen)
        self.q_sm = pg.PlotCurveItem(pen=sm_pen)
        self.r_sm = pg.PlotCurveItem(pen=sm_pen)

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
        self.p1.addItem(self.x_sm)
        self.p1.addItem(self.y_sm)
        self.p1.addItem(self.z_sm)
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
        self.p2.addItem(self.q0_sm)
        self.p2.addItem(self.q1_sm)
        self.p2.addItem(self.q2_sm)
        self.p2.addItem(self.q3_sm)

        self.p3.addItem(self.u)
        self.p3.addItem(self.v)
        self.p3.addItem(self.w)
        self.p3.addItem(self.u_sm)
        self.p3.addItem(self.v_sm)
        self.p3.addItem(self.w_sm)

        self.p4.addItem(self.p)
        self.p4.addItem(self.q)
        self.p4.addItem(self.r)
        self.p4.addItem(self.p_sm)
        self.p4.addItem(self.q_sm)
        self.p4.addItem(self.r_sm)

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

        self.u.setData(t, x[self.x_dict['v_x']])
        self.v.setData(t, x[self.x_dict['v_y']])
        self.w.setData(t, x[self.x_dict['v_z']])

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
            self.x_pr.setData(t, x[self.x_dict['x']])
            self.y_pr.setData(t, x[self.x_dict['y']])
            self.z_pr.setData(t, x[self.x_dict['z']])

            self.q0_pr.setData(t, x[self.x_dict['q0']])
            self.q1_pr.setData(t, x[self.x_dict['q1']])
            self.q2_pr.setData(t, x[self.x_dict['q2']])
            self.q3_pr.setData(t, x[self.x_dict['q3']])

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

    def load_smoothed_data(self, x_smoothed, t):
        x_smoothed = np.array(x_smoothed).T
        t = np.array(t)
        pen = pg.mkPen(0.2, style=QtCore.Qt.DashLine, width=1)

        self.x_sm.setData(t, x_smoothed[self.x_dict['x']])
        self.y_sm.setData(t, x_smoothed[self.x_dict['y']])
        self.z_sm.setData(t, x_smoothed[self.x_dict['z']])

        self.q0_sm.setData(t, x_smoothed[self.x_dict['q0']])
        self.q1_sm.setData(t, x_smoothed[self.x_dict['q1']])
        self.q2_sm.setData(t, x_smoothed[self.x_dict['q2']])
        self.q3_sm.setData(t, x_smoothed[self.x_dict['q3']])

        self.u_sm.setData(t, x_smoothed[self.x_dict['v_x']])
        self.v_sm.setData(t, x_smoothed[self.x_dict['v_y']])
        self.w_sm.setData(t, x_smoothed[self.x_dict['v_z']])

        self.p_sm.setData(t, x_smoothed[self.x_dict['p']])
        self.q_sm.setData(t, x_smoothed[self.x_dict['q']])
        self.r_sm.setData(t, x_smoothed[self.x_dict['r']])

        self.app.processEvents()


    def close(self):
        self.win.close()
