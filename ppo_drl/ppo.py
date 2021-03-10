from network import FeedForwardNN

class PPO:
    def __init__(self, env):
        # Extract environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.hidden_dim = (self.act_dim + self.obs_dim)//2
        self.hidden_dim = 64

        # Initialize actor and critic networks
        self.actor  = FeedForwardNN(self.obs_dim, self.act_dim, self.hidden_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1, self.hidden_dim)

        self._init_hyperparameters()

    
    def _init_hyperparameters(self):
        # Default values for the hyperparameters
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
    

    def rollout(self):
        # Collects batch data
        batch_obs  = []
        batch_acts = []
        batch_rews = []
        batch_rtgs = [] # Rewards to-go
        batch_lens = [] # Episodic lengths in batch
        batch_log_probs = [] # Log-Prob of each action
        
        

    def learn(self, total_timesteps):
        t_so_far = 0

        while t_so_far < total_timesteps:



