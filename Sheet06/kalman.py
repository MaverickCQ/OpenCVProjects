import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.psi = psi
        self.sigma_p = sigma_p
        self.phi = phi
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None
        self.tau = tau
        self.wt = (tau)+psi.shape[0]
        for i,j in range(psi.shape[0],psi.shape[1]):
            self.psi[i][j] = self.psi[i][j]*tau
                               
        
        #self.xt= 
        self.psi_t = np.transpose(psi)
        self.phi_t = np.transpose(phi)
        
        

    def init(self, init_state):
        self.state = init_state        
        self.convariance = np.identity(self.wt)
        #print(self.convariance)
        pass

    def track(self, xt):
        # to do
        #print(self.convariance.shape)
        #print(self.psi.shape)
        sigma_plus = self.sigma_p + np.dot( np.dot(self.sigma_p,self.convariance[0:4,0:4]), self.psi_t)
        #print(sigma_plus)
        #print(sigma_plus.shape)
        #print(self.phi.shape)
        kalman = np.dot(np.dot(sigma_plus, self.phi_t),\
                        np.linalg.inv(self.sigma_m + \
                         np.dot(np.dot(self.phi, sigma_plus), self.phi_t)))
        #print("-------------------------")
        #print(kalman)
        mean_plus = np.dot (self.psi, self.state)
        #print("xt----------", xt)
        #mean_t = xt - np.dot(self.phi, mean_plus)
        #print("mean_p-----",mean_plus)
       # print(mean_t.shape)
        #mean_plus = mean_plus + np.dot(self.psi, mean_t)
        #print("mean_plus-----",mean_plus)
        #measurement incorporation
        mean_t = mean_plus + np.dot(kalman, (xt - np.dot(self.phi, mean_plus)))
        #print(mean_t)
        #print(sigma_plus.shape)
        #print(self.phi.shape)
        #print(kalman.shape)
        new = np.identity(self.wt)
        sigma_t = np.dot(new[0:4,0:4]- np.dot(kalman[0:4,0:4], self.phi), sigma_plus)
        #print(sigma_t)
        self.convariance  = sigma_t
        self.state = mean_t
        
        pass

    def get_current_location(self):
        # to do
        return self.state
        pass

def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    return track

def main():
    init_state = np.array([1, 0, 0, 0])

    psi = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    phi = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
    sm = 0.05
    sigma_m = np.array([[sm, 0],
                        [0, sm]])


    tracker = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=0)
    tracker.init(init_state)

    fixed_lag_smoother = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=5)
    fixed_lag_smoother.init(init_state)

    track = perform_tracking(tracker)
    track_smoothed = perform_tracking(fixed_lag_smoother)

    plt.figure()
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.plot([x[0] for x in track_smoothed], [x[1] for x in track_smoothed])

    plt.show()


if __name__ == "__main__":
    main()
