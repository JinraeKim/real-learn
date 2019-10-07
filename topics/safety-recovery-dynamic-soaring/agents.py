from utils import GPRegression


class EstimateDisturbanceBound():
    def __init__(self, env):
        self.GP0 = GPRegression([],
                               [])
        self.GP1 = GPRegression([],
                               [])
        self.system = env.systems['aircraft']
        self.time_series = np.array([])
    
    def get(self, state, t):
        d = self.obs_disturbance(state, type='true')

        self.GP0.get_data(x, d[0])
        self.GP1.get_data(x, d[1])
        self.time_series = np.concatenate((self.time_series, [t]))

    def obs_disturbance(self, state, type='true'):
        if type == 'true':
            # true d
            (_, Wy, _), (_, (_, _, dWydz), _) = self.system.wind.get(state)
            d = torch.tensor([Wy, dWydz])
        
        if type == 'est':
            pass
        # estimated d by numerical diff.

        if type == 'test':
            pass
        # d only for test
        # import math
        # d = (torch.sin(x * (2 * math.pi))
        #      + torch.randn(x.size()) * 0.2)

        return d

    def train(self):
        self.GP0.train()
        self.GP1.train()

    def predict(self, x):
        _, observed_pred0 = self.GP0.eval(x)
        _, observed_pred1 = self.GP1.eval(x)
        return observed_pred0, observed_pred1


class Agent():
    def __init__(self, env):
        self.disturbance_bound = EstimateDisturbanceBound(env)

    def get(self):
        # the below constant control input should be replaced by an appropriate
        # one produced by a control law.
        return [0, 10, 0]


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from envs import DynamicSoaringEnv

    time_step = 0.01
    time_series = np.arange(0, 10, time_step)

    env = DynamicSoaringEnv(
        initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
        dt=time_step
    )
    obs = env.reset()
    obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))

    agent = Agent(env)
    disturbance_series = np.array([])  # just for true data

    for i, t in enumerate(time_series):
        controls = agent.get()
        
        next_obs, reward, done, info = env.step(controls)

        if done:
            break

        obs = next_obs
        obs_series = np.vstack((obs_series, obs))

        x = torch.from_numpy(obs).type(dtype=torch.float)
        if i % 10 == 0:
            agent.disturbance_bound.get(x, t)
            agent.disturbance_bound.train()
        disturbance_series = np.append(
            disturbance_series,
            agent.disturbance_bound.obs_disturbance(x, type='true').numpy()
        )
        # observed_pred = agent.disturbance_bound.predict(x)

    disturbance_series = disturbance_series.reshape(-1, 2)
    time_series = time_series[:obs_series.shape[0]]
    observed_pred0, observed_pred1 = \
            agent.disturbance_bound.predict(
                agent.disturbance_bound.GP1.train_x
            )

    # Plot the estimated disturbance bound
    with torch.no_grad():
        # Initialize a plot
        fig = plt.figure()
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2)

        # Get upper and lower confidence bounds
        lower0, upper0 = observed_pred0.confidence_region()
        lower1, upper1 = observed_pred1.confidence_region()

        # Plot all data
        ax0.plot(
            time_series,
            disturbance_series[:, 0],
            'k*'
        )
        ax1.plot(
            time_series,
            disturbance_series[:, 1],
            'k*'
        )

        # Plot training data as black stars
        ax0.plot(
            agent.disturbance_bound.time_series,
            # agent.disturbance_bound.GP.train_x.numpy(), 
            agent.disturbance_bound.GP0.train_y.numpy(),
            'k*'
        )
        ax1.plot(
            agent.disturbance_bound.time_series,
            # agent.disturbance_bound.GP.train_x.numpy(), 
            agent.disturbance_bound.GP1.train_y.numpy(),
            'k*'
        )

        # Plot predictive means as blue line
        ax0.plot(
            agent.disturbance_bound.time_series,
            # agent.disturbance_bound.GP.train_x.numpy(),
            observed_pred0.mean.numpy(), 
            'b'
        )
        ax1.plot(
            agent.disturbance_bound.time_series,
            # agent.disturbance_bound.GP.train_x.numpy(),
            observed_pred1.mean.numpy(), 
            'b'
        )

        # Shade between the lower and upper confidence bounds
        ax0.fill_between(
            agent.disturbance_bound.time_series,
            lower0.numpy(), 
            upper0.numpy(), 
            alpha=0.5
        )
        ax1.fill_between(
            agent.disturbance_bound.time_series,
            lower1.numpy(), 
            upper1.numpy(), 
            alpha=0.5
        )

        ax0.set_ylim([-20, 20])
        # ax0.set_xlabel('time (s)')
        ax0.set_ylabel('disturbance0')
        ax0.legend(['True', 'Observed Data', 'Mean', 'Confidence'])
        ax1.set_ylim([-10, 10])
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('disturbance1')
        ax0.legend(['True', 'Observed Data', 'Mean', 'Confidence'])

        plt.show()
