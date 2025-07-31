"""Common datastructures."""

from dataclasses import dataclass
from typing import TypeVar, Hashable, Generic, Callable
from gymnasium.spaces import Space
from relational_structs import Type, Predicate
from bilevel_planning.structs import LiftedSkill, RelationalAbstractGoal, RelationalAbstractState, LiftedOperator


_X = TypeVar("_X", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


@dataclass(frozen=True)
class BilevelPlanningEnvModels(Generic[_X, _U]):
    """Holds all bilevel planning models for one environment."""

    state_space: Space[_X]
    action_space: Space[_U]
    transition_fn: Callable[[_X, _U], _X]
    types: set[Type]
    predicates: set[Predicate]
    state_abstractor: Callable[[_X], RelationalAbstractState]
    goal_abstractor: Callable[[_X], RelationalAbstractGoal]
    skills: set[LiftedSkill]

    @property
    def operators(self) -> set[LiftedOperator]:
        """Access the lifted operators from the lifted skills."""
        return {s.operator for s in self.skills}
