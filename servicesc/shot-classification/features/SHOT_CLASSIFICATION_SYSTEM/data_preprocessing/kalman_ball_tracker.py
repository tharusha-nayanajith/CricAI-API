import numpy as np

class KalmanBallTracker:
    """
    Constant-acceleration ball tracker
    State: [x, y, vx, vy]
    """
    def __init__(self):
        self.state = None
        self.P = np.eye(4) * 1000

        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.R = np.eye(2) * 10
        self.Q = np.eye(4) * 0.1

    def update(self, measurement: np.ndarray):
        if self.state is None:
            self.state = np.array([measurement[0], measurement[1], 0, 0])
            return

        # Prediction
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Correction
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def predict(self):
        if self.state is None:
            return None

        self.state = self.F @ self.state
        return self.state[:2]