from utils import GPRegression


class EstimateDisturbanceBound():
    def __init__(self, env):
        self.GP = GPRegression([],
                               [])
        self.system = env.systems['aircraft']
    
    def get(self, state):
        # true d
        (_, Wy, _), (_, (_, _, dWydz), _) = self.system.wind.get(state)
        d = torch.tensor([Wy, dWydz])
        
        # estimated d by numerical diff.

        # naive d
        # import math
        # d = (torch.sin(x * (2 * math.pi))
        #      + torch.randn(x.size()) * 0.2)

        self.GP.get_data(x, d[0])

    def train(self):
        self.GP.train()

    def predict(self, x):
        pred, observed_pred = self.GP.eval(x)
        return observed_pred 


class Agent():
    def __init__(self, env):
        self.disturbance_bound = EstimateDisturbanceBound(env)

    def get(self):
        # the below constant control input should be replaced by an appropriate
        # one produced by a control law.
        return [0, 1, 0]


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from envs import DynamicSoaringEnv

    time_step = 0.01
    time_series = np.arange(0, 1, time_step)

    env = DynamicSoaringEnv(
        initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
        dt=time_step
    )
    obs = env.reset()
    obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))

    agent = Agent(env)

    for i, t in enumerate(time_series):
        controls = agent.get()
        
        next_obs, reward, done, info = env.step(controls)

        if done:
            break

        obs = next_obs
        obs_series = np.vstack((obs_series, obs))

        x = torch.from_numpy(obs).type(dtype=torch.float)
        agent.disturbance_bound.get(x)
        agent.disturbance_bound.train()
        # observed_pred = agent.disturbance_bound.predict(x)


    x_series = torch.from_numpy(
        obs_series.reshape((-1, 6)).squeeze()
    ).type(dtype=torch.float)
    observed_pred = agent.disturbance_bound.predict(x_series)
    # Plot the estimated disturbance bound
    with torch.no_grad():
        # Initialize a plot
        f, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(
            range(len(agent.disturbance_bound.GP.train_x)),
            # agent.disturbance_bound.GP.train_x.numpy(), 
            agent.disturbance_bound.GP.train_y.numpy(),
            'k*'
        )
        # Plot predictive means as blue line
        ax.plot(
            range(len(agent.disturbance_bound.GP.train_x)),
            # agent.disturbance_bound.GP.train_x.numpy(),
            observed_pred.mean.numpy(), 
            'b'
        )
        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            range(len(agent.disturbance_bound.GP.train_x)),
            # agent.disturbance_bound.GP.train_x.numpy(), 
            lower.numpy(), 
            upper.numpy(), 
            alpha=0.5
        )
        ax.set_ylim([8, 9])
        ax.set_xlabel('n th data (state)')
        ax.set_ylabel('disturbance')
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
