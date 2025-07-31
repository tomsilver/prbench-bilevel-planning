"""Common datastructures."""

from dataclasses import dataclass
from typing import TypeVar, Hashable, Generic, Callable
from gymnasium.spaces import Space
from relational_structs import Type, Predicate
from bilevel_planning.structs import LiftedSkill, RelationalAbstractGoal, RelationalAbstractState, LiftedOperator


# These distinctions are primarily necessary because we are assuming in bilevel-planning
# that states and actions are hashable.
_O = TypeVar("_O")
_E = TypeVar("_E")
_X = TypeVar("_X", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


@dataclass(frozen=True)
class BilevelPlanningEnvModels(Generic[_O, _E, _X, _U]):
    """Holds all bilevel planning models for one environment."""

    observation_space: Space[_O]
    executable_space: Space[_E]
    state_space: Space[_X]
    action_space: Space[_U]
    transition_fn: Callable[[_X, _U], _X]
    types: set[Type]
    predicates: set[Predicate]
    observation_to_state: Callable[[_O], _X]
    action_to_executable: Callable[[_U], _E]
    state_abstractor: Callable[[_X], RelationalAbstractState]
    goal_abstractor: Callable[[_X], RelationalAbstractGoal]
    skills: set[LiftedSkill]

    @property
    def operators(self) -> set[LiftedOperator]:
        """Access the lifted operators from the lifted skills."""
        return {s.operator for s in self.skills}
