from __future__ import annotations

import torch

try:
    from tensordict import TensorDictBase
except ImportError:  # pragma: no cover - older setups without tensordict
    TensorDictBase = ()


GO2_JUMP_OBS_PERMUTATION = (
    -0.0001,
    -1,
    2,
    -3,
    -4,
    -5,
    6,
    -7,
    -8,
    9,
    -10,
    -14,
    15,
    16,
    -11,
    12,
    13,
    -20,
    21,
    22,
    -17,
    18,
    19,
    -26,
    27,
    28,
    -23,
    24,
    25,
    -32,
    33,
    34,
    -29,
    30,
    31,
    -38,
    39,
    40,
    -35,
    36,
    37,
    -44,
    45,
    46,
    -41,
    42,
    43,
)

GO2_JUMP_ACTION_PERMUTATION = (-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8)

GO2_JUMP_POLICY_FRAME_DIM = 47
GO2_JUMP_POLICY_FRAME_STACK = 10


def _signed_permutation_tensors(
    permutation: tuple[float, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = []
    signs = []
    for value in permutation:
        signs.append(-1.0 if value < 0 else 1.0)
        abs_value = abs(value)
        index = 0 if abs_value < 0.5 else int(abs_value)
        indices.append(index)
    return (
        torch.tensor(indices, device=device, dtype=torch.long),
        torch.tensor(signs, device=device, dtype=torch.float),
    )


def _apply_signed_permutation(tensor: torch.Tensor, permutation: tuple[float, ...]) -> torch.Tensor:
    indices, signs = _signed_permutation_tensors(permutation, device=tensor.device)
    return tensor.index_select(-1, indices) * signs


def _mirror_policy_obs_tensor(obs: torch.Tensor) -> torch.Tensor:
    obs_frames = obs.reshape(-1, GO2_JUMP_POLICY_FRAME_STACK, GO2_JUMP_POLICY_FRAME_DIM)
    mirrored_frames = _apply_signed_permutation(obs_frames, GO2_JUMP_OBS_PERMUTATION)
    return mirrored_frames.reshape(obs.shape[0], -1)


def augment_go2_jump_symmetry(
    env,
    obs: torch.Tensor | TensorDictBase | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | TensorDictBase | None, torch.Tensor | None]:
    del env

    augmented_obs = None
    augmented_actions = None

    if obs is not None:
        if isinstance(obs, TensorDictBase):
            mirrored_obs = obs.clone()
            if obs_type == "policy":
                mirrored_obs["policy"] = _mirror_policy_obs_tensor(obs["policy"])
            else:
                mirrored_obs["policy"] = obs["policy"].clone()
            augmented_obs = torch.cat((obs, mirrored_obs), dim=0)
        else:
            if obs_type == "policy":
                mirrored_obs = _mirror_policy_obs_tensor(obs)
            else:
                # Critic observations do not use a dedicated mirror mapping here.
                # Duplicating them preserves the augmentation contract expected by RSL-RL.
                mirrored_obs = obs.clone()
            augmented_obs = torch.cat((obs, mirrored_obs), dim=0)

    if actions is not None:
        mirrored_actions = _apply_signed_permutation(actions, GO2_JUMP_ACTION_PERMUTATION)
        augmented_actions = torch.cat((actions, mirrored_actions), dim=0)

    return augmented_obs, augmented_actions
