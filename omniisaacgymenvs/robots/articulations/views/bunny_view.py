
from typing import Optional

from omni.isaac.core.prims import GeometryPrimView


class BunnyView(GeometryPrimView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "BunnyView",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name
        )