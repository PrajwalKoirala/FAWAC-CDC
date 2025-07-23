import copy
from fsrl.utils import DummyLogger, WandbLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm.auto import trange

TensorBatch = List[torch.Tensor]



def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)



class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        cost_scale: float = 1.0,
        reward_scale: float = 1.0,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._costs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device
        self._cost_returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.cost_scale = cost_scale
        self.reward_scale = reward_scale

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None]) * self.reward_scale
        self._costs[:n_transitions] = self._to_tensor(data["costs"][..., None]) * self.cost_scale
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None]) + self._to_tensor(data["timeouts"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")



    def sample(self, batch_size: int) -> TensorBatch:
        indices = torch.randint(0, self._size, (batch_size,))
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        costs = self._costs[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, costs, next_states, dones]


def discounted_cumsum(x, gamma: float):
    """
    Calculate the discounted cumulative sum of x (can be rewards or costs).
    """
    cumsum = torch.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SafeActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 max_action: float = 1.0, device: str = "cuda"):
        super().__init__()
        self.actor = MLP(
            [state_dim, hidden_dim, hidden_dim, action_dim], nn.ReLU, output_activation_fn=nn.Tanh
        )
        self.device = device
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state) * self.max_action
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().reshape(1, -1)



class lagrangian_fxn(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, max_lag: float = 10.0):
        super().__init__()
        self.lagrangian = MLP(
            [state_dim, hidden_dim, hidden_dim, 1], nn.ReLU, output_activation_fn=nn.Softplus
        )
        self.max_lag = max_lag

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.lagrangian(state), min=0.0, max=self.max_lag)





class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
        n_hidden: int = 2, type: str = "reward"
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        self.type = type
        if type not in ["reward", "cost"]:
            raise ValueError("Invalid q function type")
        self.init_weights()
        
    def init_weights(self):
        def init_(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
        self.q1.apply(init_)
        self.q2.apply(init_)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.type == "reward":
            return torch.min(*self.both(state, action))
        elif self.type == "cost":
            return torch.clamp(torch.max(*self.both(state, action)), min=0.0)


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)
        # self.init_weights()

    def init_weights(self):
        def init_(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
        self.v.apply(init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.v(state), min=0.0)


def FAWAC_IQL_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    max_action = config["max_action"]
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    device = config["device"]
    actor_lr = config["actor_lr"]
    qf_lr = config["qf_lr"]
    vf_lr = config["vf_lr"]
    max_lag = config["lag_max"]
    # networks
    actor = SafeActor(state_dim, action_dim, max_action=max_action, device=device).to(device)
    q_network = TwinQ(state_dim, action_dim).to(device)
    v_network = ValueFunction(state_dim).to(device)
    cost_q_network = TwinQ(state_dim, action_dim, type="cost").to(device)
    cost_v_network = ValueFunction(state_dim).to(device)
    lagrangian = lagrangian_fxn(state_dim, max_lag=max_lag).to(device)
    # optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=qf_lr)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=vf_lr)
    cost_q_optimizer = torch.optim.Adam(cost_q_network.parameters(), lr=qf_lr)
    cost_v_optimizer = torch.optim.Adam(cost_v_network.parameters(), lr=vf_lr)
    lagrangian_optimizer = torch.optim.Adam(lagrangian.parameters(), lr=actor_lr/100)
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "lagrangian": lagrangian,
        "actor_optimizer": actor_optimizer,
        "lagrangian_optimizer": lagrangian_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "cost_q_network": cost_q_network,
        "cost_q_optimizer": cost_q_optimizer,
        "cost_v_network": cost_v_network,
        "cost_v_optimizer": cost_v_optimizer,
        "device": device,
    }
    return kwargs


class FAWAC_IQL_Trainer:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        lagrangian: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        lagrangian_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        cost_q_network: nn.Module,
        cost_q_optimizer: torch.optim.Optimizer,
        cost_v_network: nn.Module,
        cost_v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        discount: float = 0.99,
        tau: float = 0.005,
        exp_adv_max_cost: float = 1000.0,
        exp_adv_max_reward: float = 200.0,
        device: str = "cpu",
        logger: WandbLogger = DummyLogger(),
        episode_len: int = 300,
        safe_qc_vc_threshold: float = 0.02,
        lag_max: float = 5.0,
        cost_planning_steps: float = 100.0,
        obj_method: str = "penalty",
        cost_limit: int = 10
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.lagrangian = lagrangian
        self.lagrangian_optimizer = lagrangian_optimizer
        self.iql_tau = iql_tau
        self.rew_beta = beta
        self.discount = discount
        self.tau = tau
        self.exp_adv_max_cost = exp_adv_max_cost
        self.exp_adv_max_reward = exp_adv_max_reward
        self.logger = logger
        self.episode_len = episode_len
        self.safe_qc_vc_threshold = safe_qc_vc_threshold

        # cost critics
        self.cost_qf = cost_q_network
        self.cost_q_target = copy.deepcopy(self.cost_qf).requires_grad_(False).to(device)
        self.cost_q_optimizer = cost_q_optimizer
        self.cost_vf = cost_v_network
        self.cost_v_optimizer = cost_v_optimizer
        #
        self.max_lag = lag_max
        self.cost_planning_steps = cost_planning_steps
        self.obj_method = obj_method
        self.cost_limit = float(cost_limit)
        self.cost_thresh = self.cost_limit*self.discount**self.cost_planning_steps

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

    def _update_cost_v(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        log_dict: Dict,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_cost_q = self.cost_q_target(states, actions)
            target_cost_q = torch.clamp(target_cost_q, min=0.0)
        cost_v = self.cost_vf(states)
        cost_v = torch.clamp(cost_v, min=0.0)
        cost_adv = -(target_cost_q - cost_v)

        # safety mask
        safety_threshold = self.safe_qc_vc_threshold
        safety_mask = (cost_v < safety_threshold).float() * (target_cost_q < safety_threshold).float()
        log_dict["safety_mask_mean"] = safety_mask.mean().item()
        log_dict["cost_v_mean"] = cost_v.mean().item()
        log_dict["cost_v_max"] = torch.max(cost_v).item()
        log_dict["cost_q_mean"] = target_cost_q.mean().item()
        log_dict["cost_q_max"] = torch.max(target_cost_q).item()
        log_dict["costs_mean"] = costs.mean().item()

        
        cost_v_loss = asymmetric_l2_loss(cost_adv, self.iql_tau)
        log_dict["cost_v_loss"] = cost_v_loss.item()
        self.cost_v_optimizer.zero_grad()
        cost_v_loss.backward()
        self.cost_v_optimizer.step()

        return cost_adv, safety_mask.detach()

    def _update_cost_q(
        self,
        next_cost_v: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        log_dict: Dict,
    ):
        targets = costs + (1.0 - dones.float()) * self.discount * next_cost_v.detach()
        cost_qs = self.cost_qf.both(states, actions)
        cost_q_loss = sum(F.mse_loss(q, targets) for q in cost_qs) / len(cost_qs)
        log_dict["cost_q_loss"] = cost_q_loss.item()
        self.cost_q_optimizer.zero_grad()
        cost_q_loss.backward()
        self.cost_q_optimizer.step()
        soft_update(self.cost_q_target, self.cost_qf, self.tau)
        

    def _update_policy(
        self,
        rew_adv: torch.Tensor,
        cost_adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        #
        if self.obj_method == "penalty":
            qc_actions = self.cost_qf(observations, actions).detach()
            vc = self.cost_vf(observations).detach()
            # lag = ((qc_actions - self.cost_thresh)>0).float() * ((vc - self.cost_thresh) > 0).float() * self.max_lag
            lag = ((vc - self.cost_thresh) > 0).float() * self.max_lag
        else:
            lag = self.lagrangian(observations)
        lag = lag.detach()
        #        
        net_adv = rew_adv + lag * cost_adv # cost_adv already has a negative sign
        exp_adv = torch.exp(self.rew_beta * net_adv).detach().clamp(max=self.exp_adv_max_reward)
        actions_out = self.actor(observations)
        #
        bc_losses = torch.sum((actions_out - actions) ** 2, dim=1)
        policy_loss = torch.mean(exp_adv * bc_losses)
        #
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        # logging weights
        max_exp_adv = torch.max(exp_adv)
        log_dict["weights/max_vae_exp_adv"] = max_exp_adv.item()
        min_exp_adv = torch.min(exp_adv)
        log_dict["weights/min_vae_exp_adv"] = min_exp_adv.item()
        mean_exp_adv = torch.mean(exp_adv)
        log_dict["weights/mean_vae_exp_adv"] = mean_exp_adv.item()
        log_dict["bc_loss"] = torch.mean(bc_losses).item()
        log_dict["actor_loss"] = policy_loss.item()
        # logging lagrangian
        max_lag = torch.max(lag)
        log_dict["weights/max_lag"] = max_lag.item()
        min_lag = torch.min(lag)
        log_dict["weights/min_lag"] = min_lag.item()
        mean_lag = torch.mean(lag)
        log_dict["weights/mean_lag"] = mean_lag.item()
        median_lag = torch.median(lag)
        log_dict["weights/median_lag"] = median_lag.item()



    def _update_lagrangian(
        self,
        observations: torch.Tensor,
        log_dict: Dict
    ):
        lag_values = self.lagrangian(observations)
        pi_actions = self.actor(observations)
        cost_q = self.cost_qf(observations, pi_actions)
        cost_v = self.cost_vf(observations)
        # lag_loss = - lag_values * (cost_q - self.cost_thresh)
        lag_loss = - lag_values * (cost_v - self.cost_thresh)
        lag_loss = torch.mean(lag_loss)
        # logging
        log_dict["lag_loss"] = lag_loss.item()
        max_lag_in_the_batch = torch.max(lag_values)
        min_lag_in_the_batch = torch.min(lag_values)
        mean_lag_in_the_batch = torch.mean(lag_values)
        median_lag_in_the_batch = torch.median(lag_values)
        log_dict["lag_log/max_lag_in_the_batch"] = max_lag_in_the_batch.item()
        log_dict["lag_log/min_lag_in_the_batch"] = min_lag_in_the_batch.item()
        log_dict["lag_log/mean_lag_in_the_batch"] = mean_lag_in_the_batch.item()
        log_dict["lag_log/median_lag_in_the_batch"] = median_lag_in_the_batch.item()
        max_cost_q_in_the_batch = torch.max(cost_q)
        min_cost_q_in_the_batch = torch.min(cost_q)
        mean_cost_q_in_the_batch = torch.mean(cost_q)
        median_cost_q_in_the_batch = torch.median(cost_q)
        log_dict["lag_log/max_cost_q_in_the_batch"] = max_cost_q_in_the_batch.item()
        log_dict["lag_log/min_cost_q_in_the_batch"] = min_cost_q_in_the_batch.item()
        log_dict["lag_log/mean_cost_q_in_the_batch"] = mean_cost_q_in_the_batch.item()
        log_dict["lag_log/median_cost_q_in_the_batch"] = median_cost_q_in_the_batch.item()
        max_cost_v_in_the_batch = torch.max(cost_v)
        min_cost_v_in_the_batch = torch.min(cost_v)
        mean_cost_v_in_the_batch = torch.mean(cost_v)
        median_cost_v_in_the_batch = torch.median(cost_v)
        log_dict["lag_log/max_cost_v_in_the_batch"] = max_cost_v_in_the_batch.item()
        log_dict["lag_log/min_cost_v_in_the_batch"] = min_cost_v_in_the_batch.item()
        log_dict["lag_log/mean_cost_v_in_the_batch"] = mean_cost_v_in_the_batch.item()
        log_dict["lag_log/median_cost_v_in_the_batch"] = median_cost_v_in_the_batch.item()
        #
        self.lagrangian_optimizer.zero_grad()
        lag_loss.backward()
        self.lagrangian_optimizer.step()


    
    def train_one_step(
        self, observations: torch.Tensor, next_observations: torch.Tensor, 
        actions: torch.Tensor, rewards: torch.Tensor, costs: torch.Tensor,
        dones: torch.Tensor
    ):
        self.total_it += 1
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
            next_cost_v = self.cost_vf(next_observations)
            next_cost_v = torch.clamp(next_cost_v, min=0.0)
            #
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        #
        # Update Cost critics
        cost_adv, safety_mask = self._update_cost_v(observations, actions, costs, log_dict)
        costs = costs.squeeze(dim=-1)
        self._update_cost_q(next_cost_v, observations, actions, costs, dones, log_dict)
        if self.total_it % 2 == 0:
            # Update policy
            self._update_policy(adv, cost_adv, observations, actions, log_dict)
            # Update lagrangian
            if self.obj_method == "statewise_lagrangian":
                self._update_lagrangian(observations, log_dict)
        self.logger.store(**log_dict)
    


    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "lagrangian": self.lagrangian.state_dict(),
            "lagrangian_optimizer": self.lagrangian_optimizer.state_dict(),
            "cost_qf": self.cost_qf.state_dict(),
            "cost_q_optimizer": self.cost_q_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.actor.load_state_dict(state_dict["actor"])
        self.lagrangian.load_state_dict(state_dict["lagrangian"])

        self.actor.load_state_dict(state_dict["actor"])
        self.lagrangian.load_state_dict(state_dict["lagrangian"])
        self.lagrangian_optimizer.load_state_dict(state_dict["lagrangian_optimizer"])

        self.cost_qf.load_state_dict(state_dict["cost_qf"])
        self.cost_q_optimizer.load_state_dict(state_dict["cost_q_optimizer"])
        self.cost_q_target = copy.deepcopy(self.cost_qf)

        self.total_it = state_dict["total_it"]

    @torch.no_grad()
    def evaluate(self, env, eval_episodes: int) -> Tuple:
        self.actor.eval()
        rets, costs, lengths = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating FAWAC ...", leave=False):
            ret, cost, length = self.rollout(env)
            rets.append(ret)
            costs.append(cost)
            lengths.append(length)
        self.actor.train()
        return np.mean(rets), np.mean(costs), np.mean(lengths), np.std(rets), np.std(costs), np.std(lengths)
    
    @torch.no_grad()
    def rollout(self, env) -> Tuple[float, float, int]:
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for _ in range(self.episode_len):
            act = self.actor.act(obs, self.device)
            act = act.flatten()
            obs_next, reward, terminated, truncated, info = env.step(act)
            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]
            if terminated or truncated:
                break
            obs = obs_next
        return episode_ret, episode_cost, episode_len
    

