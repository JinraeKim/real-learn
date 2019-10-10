from utils import GPRegression


class EstimateDisturbanceBound():
    def __init__(self, env, disturbance_length=2):
        self.disturbance_length = disturbance_length
        self.GP = []
        for i in range(disturbance_length):
            self.GP.append(GPRegression([], []))

        self.system = env.systems['aircraft']
        self.time_series = np.array([])
    
    def put(self, state, t):
        d = self.obs_disturbance(state, type='true')

        for i in range(self.disturbance_length):
            self.GP[i].put(x, d[i])

        self.time_series = np.concatenate((self.time_series, [t]))

    def obs_disturbance(self, state, type='true'):
        if type == 'true':
            # true d
            (_, Wy, _), (_, (_, _, dWydz), _) = self.system.wind.get(state)
            d = torch.tensor([Wy, dWydz])
        
        if type == 'est':
            pass
        # estimated d by numerical diff.
        
        return d

    def train(self):
        for i in range(self.disturbance_length):
            self.GP[i].train()

    def predict(self, x):
        observed_pred = [None] * self.disturbance_length
        for i in range(self.disturbance_length):
            _, observed_pred[i] = self.GP[i].eval(x)
        return observed_pred


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
    time_series = np.arange(0, 5, time_step)

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
            agent.disturbance_bound.put(x, t)
        if i & 100 == 0:
            agent.disturbance_bound.train()
        disturbance_series = np.append(
            disturbance_series,
            agent.disturbance_bound.obs_disturbance(x, type='true').numpy()
        )

    disturbance_series = disturbance_series.reshape(-1, 2)
    time_series = time_series[:obs_series.shape[0]]
    disturbance_series = disturbance_series.reshape(-1, 2)
    observed_pred = \
            agent.disturbance_bound.predict(
                agent.disturbance_bound.GP[0].train_x
            )

    # Plot the estimated disturbance bound
    with torch.no_grad():
        fig = plt.figure()
        N = agent.disturbance_bound.disturbance_length
        ax = [None] * N
        lower = [None] * N
        upper = [None] * N

        for i in range(agent.disturbance_bound.disturbance_length):
            # Initialize a plot
            ax[i] = fig.add_subplot(2, 1, i+1)

            # Get upper and lower confidence bounds
            lower[i], upper[i] = observed_pred[i].confidence_region()

            # Plot all data
            ax[i].plot(
                time_series,
                disturbance_series[:, i],
                'k'
            )

            # Plot training data as black stars
            ax[i].plot(
                agent.disturbance_bound.time_series,
                # agent.disturbance_bound.GP.train_x.numpy(), 
                agent.disturbance_bound.GP[i].train_y.numpy(),
                'r*'
            )

            # Plot predictive means as blue line
            ax[i].plot(
                agent.disturbance_bound.time_series,
                # agent.disturbance_bound.GP.train_x.numpy(),
                observed_pred[i].mean.numpy(), 
                'b'
            )

            # Shade between the lower and upper confidence bounds
            ax[i].fill_between(
                agent.disturbance_bound.time_series,
                lower[i].numpy(), 
                upper[i].numpy(), 
                alpha=0.5
            )

            ax[i].set_ylim([-20, 20])
            ax[i].set_ylabel('disturbance'+str(i))
            ax[i].legend(['All Data', 'Observed Data', 'Mean', 'Confidence'])

        plt.show()
