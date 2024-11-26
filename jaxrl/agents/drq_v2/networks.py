from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from jaxrl.networks.common import default_init
from jaxrl.networks.critic_net import DoubleCritic, ActivationTrackDoubleDistributionalCritic, DistributionalCritic, ActivationTrackDoubleCritic
from jaxrl.networks.policies import NormalTanhPolicy, NormalTanhDeterministicPolicy


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        layer_count = 0
        for features, stride in zip(self.features, self.strides):
            layer = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding,
                        name='conv{}'.format(layer_count))
            x = layer(x)
            x = nn.relu(x)
            x = IdentityLayer(name=f'{layer.name}_act')(x)
            layer_count += 1

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


class DrQDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return DoubleCritic(self.hidden_dims)(x, actions)


class DrQv2DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int = 50

    @nn.compact
    def __call__(self, encodings: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(self.latent_dim)(encodings)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        x = jnp.concatenate([x, actions], axis=-1)

        return ActivationTrackDoubleCritic(self.hidden_dims)(x)


class DrQDistributionalSingleCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_logits: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return DistributionalCritic(self.hidden_dims, self.n_logits, name='CriticHead')(x, actions)


class DrQPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.Distribution:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NormalTanhPolicy(self.hidden_dims, self.action_dim)(x,
                                                                   temperature)


class DrQv2Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    latent_dim: int = 50

    @nn.compact
    def __call__(self,
                 encodings: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.Distribution:

        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(encodings)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NormalTanhDeterministicPolicy(self.hidden_dims, self.action_dim)(x)