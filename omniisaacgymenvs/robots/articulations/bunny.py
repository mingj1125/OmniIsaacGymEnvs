# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Optional
import os

from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.utils.stage import add_reference_to_stage

import numpy as np
import torch


class Bunny(GeometryPrim):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Bunny",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            path = os.getcwd()
            self._usd_path = path + "/usd_data/bunny.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        self._position = torch.tensor([0.0, 0.0, 1.8]) if translation is None else translation
        self._orientation = torch.tensor([0.70711, 0.70711, 0.0, 0.0]) if orientation is None else orientation


        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation
        )