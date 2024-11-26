"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple
import re

import flax.traverse_util
from optax._src import base
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl.agents.drq.networks import DrQDoubleCritic, DrQPolicy, ActivationTrackDrQDoubleCritic
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOpt
from jaxrl.agents.drq import weight_recyclers


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    next_observations = batched_random_crop(key, batch.next_observations)

    batch = batch._replace(observations=observations,
                           next_observations=next_observations)

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            soft_critic=True)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # Use critic conv layers in actor:
    new_actor_params = actor.params.copy(
        add_or_replace={'SharedEncoder': new_critic.params['SharedEncoder']})
    actor = actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class DrQLearner(object):

    def __init__(self,
                 seed: int,
                 track: bool,
                 replay_buffer,
                 redo: bool,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 reset_interval: int = 200_000,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 batch_size: int = 512,
                 batch_size_statistics: int = 256,
                 dead_neurons_threshold: float = 0.1,
                 dormancy_logging_period: int = 2_000,
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 0.1):

        action_dim = actions.shape[-1] # q-r: 12 h-h: 4
        # print(action_dim, observations.shape) # (1, 84, 84, 9)

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = DrQPolicy(hidden_dims, action_dim, cnn_features,
                              cnn_strides, cnn_padding, latent_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                                    cnn_padding, latent_dim)
        # critic = Model.create(critic_def,
        #                       inputs=[critic_key, observations, actions],
        #                       tx=optax.adam(learning_rate=critic_lr))
        critic = ModelDecoupleOpt.create(critic_def,
                                         inputs=[critic_key, observations, actions],
                                         tx=optax.adam(learning_rate=critic_lr),
                                         tx_enc=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng
        self.step = 0

        import flax
        param_dict = flax.traverse_util.flatten_dict(critic.params, sep='/')
        layer_list = list(param_dict.keys())
        layer_list = [l[l.find('/')+1:l.rfind('/')] for l in layer_list]
        reset_layer_list = list(dict.fromkeys(layer_list))
        reset_layer_list = [l for l in reset_layer_list if 'final' not in l and l != '']
        # reset_layer_list = reset_layer_list[reset_start_layer_idx:]
        if redo:
            self.critic_weight_recycler = weight_recyclers.NeuronRecycler(reset_layer_list, 
                                                                          track, 
                                                                          dead_neurons_threshold=dead_neurons_threshold, 
                                                                          dormancy_logging_period=dormancy_logging_period, 
                                                                          prune_dormant_neurons=False, 
                                                                          reset_period=reset_interval)
        else:
            self.critic_weight_recycler = weight_recyclers.NeuronRecycler(reset_layer_list, 
                                                                          track, 
                                                                          dead_neurons_threshold=dead_neurons_threshold, 
                                                                          dormancy_logging_period=dormancy_logging_period, 
                                                                          )

        self.replay_buffer = replay_buffer
        self.batch_size_statistics = batch_size_statistics
        self.redo = redo

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)

        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def get_intermediates(self, network, online_params):
        batch = self.replay_buffer.sample(self.batch_size_statistics)
        _, state = network.apply(
            {'params': online_params},
            batch.observations,
            batch.actions,
            capture_intermediates=lambda l, _: l.name is not None and 'act' in l.name,
            mutable=['intermediates'],
        )
        return state['intermediates']

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0)
        
        # is_intermediated = self.critic_weight_recycler.is_intermediated_required(self.step)
        # intermediates = (
        #     self.get_intermediates(new_critic, new_critic.params) if is_intermediated else None
        # )
        # critic_intermediates = intermediates#['CriticHead']
        # self.critic_weight_recycler.maybe_log_deadneurons(
        #     self.step, critic_intermediates
        # )
        # actor_intermediates = (
        #     self.get_intermediates(self.actor, actor_online_params) if is_intermediated else None
        # )
        # self.actor_weight_recycler.maybe_log_deadneurons(
        #     self.step, actor_intermediates
        # ) # TODO log actor dormancy statistics

        self.rng = new_rng
        if self.redo:
            self.rng, key = jax.random.split(self.rng)
            # redone_enc_params, redone_enc_opt_state = self.encoder_weight_recycler.maybe_update_weights(
            #     self.step, encoder_intermediates, new_critic.params['SharedEncoder'], key, new_critic.opt_state_enc
            # )
            # opt_state = flax.core.FrozenDict()
            # print(new_critic.params.keys())
            # print(new_critic.params['/dense-1_layernorm_tanh'].keys())
            redone_critic_params, redone_critic_opt_state = self.critic_weight_recycler.maybe_update_weights(
                self.step, critic_intermediates, new_critic.params, key, [new_critic.opt_state_enc, new_critic.opt_state_head]
            )
            # new_params = flax.core.unfreeze(new_critic.params)
            # new_params['SharedEncoder'], new_params['CriticHead'] = redone_enc_params, redone_critichead_params
            new_critic = new_critic.replace(params=redone_critic_params, 
                                            opt_state_enc=redone_critic_opt_state[0],
                                            opt_state_head=redone_critic_opt_state[1])

        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
