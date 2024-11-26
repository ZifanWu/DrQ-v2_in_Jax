from typing import Tuple

import jax
import flax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


def update(key: PRNGKey, stddev: float, encoder: Model, actor: Model, critic: Model, target_critic: Model,
            batch: Batch, discount: float, stddev_clip: float,
           soft_critic: bool) -> Tuple[Model, InfoDict]:
    next_encodings = encoder(batch.next_observations)
    dist = actor(next_encodings, stddev)
    next_actions = dist.sample(seed=key, clip=stddev_clip)
    next_qs = target_critic(next_encodings, next_actions) # (2, B)
    next_q1, next_q2 = next_qs[0], next_qs[1]
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    current_encodings = encoder(batch.observations)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # encodings = encoder.apply({'params': encoder_params}, batch.observations)
        # encodings = encoder(batch.observations)
        qs = critic.apply({'params': critic_params}, current_encodings, batch.actions)
        q1, q2 = qs[0], qs[1]
        
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'r': batch.rewards.mean(),
        }
    def encoder_loss_fn(encoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        encodings = encoder.apply({'params': encoder_params}, batch.observations)
        # encodings = encoder(batch.observations)
        qs = critic(encodings, batch.actions)
        q1, q2 = qs[0], qs[1]
        
        encoder_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return encoder_loss, {
            'encoder_loss': encoder_loss,
        }
    new_encoder, encoder_info = encoder.apply_gradient(encoder_loss_fn)
    new_critic, critic_info = critic.apply_gradient(critic_loss_fn)

    return new_encoder, new_critic, {**encoder_info, **critic_info}