"""A general interface for an agent that runs bilevel planning."""

from prpl_utils.gym_agent import Agent
from typing import TypeVar, Hashable
from gymnasium.spaces import Space

_O = TypeVar("_O", bound=Hashable)
_A = TypeVar("_A", bound=Hashable)


class BilevelPlanningAgent(Agent[_O, _A]):
    """A general interface for an agent that runs bilevel planning."""

    def __init__(self, env_models: None, seed: int) -> None:
        self._env_models = env_models
        super().__init__(seed)
    
    def _get_action(self) -> _A:
        import ipdb; ipdb.set_trace()
