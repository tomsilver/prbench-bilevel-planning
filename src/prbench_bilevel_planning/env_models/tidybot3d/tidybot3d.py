"""TidyBot3D bilevel planning models.

This module provides bilevel planning models for TidyBot3D environment
by adapting the regular observation space to object-centric format.
"""

from prbench_bilevel_planning.env_models.tidybot3d.object_centric_adapter import create_tidybot3d_bilevel_planning_models

# Re-export the function with the expected name
create_bilevel_planning_models = create_tidybot3d_bilevel_planning_models

__all__ = ["create_bilevel_planning_models"] 