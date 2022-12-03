from gettext import translation
from omniisaacgymenvs.robots.articulations.bunny import Bunny
from omniisaacgymenvs.robots.articulations.views.bunny_view import BunnyView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.table import Table
from omniisaacgymenvs.robots.articulations.views.table_view import TableView

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
from omni.isaac.core.prims import RigidPrimView

from omni.isaac.core.utils.stage import get_current_stage
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import UsdGeom, Gf, Vt, UsdPhysics, PhysxSchema, Usd, UsdShade, Sdf
import omni.usd
import omni.kit

import numpy as np
import torch
import math

class MovingOnTheTabTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 15
        self._num_actions = 3

        self._table_position = torch.tensor([0, 0, 0.4])
        self._crackerbox_position = torch.tensor([0, 0, 0.87])
        self._bunny_position = torch.tensor([0.0, 0., 1.8])

        RLTask.__init__(self, name=name, env=env)

        max_thrust = 0.8
        self.force_lower_limits = -max_thrust * torch.ones(3, device=self._device, dtype=torch.float32)
        self.force_upper_limits = max_thrust * torch.ones(3, device=self._device, dtype=torch.float32)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        return

    def set_up_scene(self, scene) -> None:
        self.get_table()
        self.get_crackerbox()
        # self.get_bunny()
        self.get_target()
        super().set_up_scene(scene)
        self._tables = TableView(prim_paths_expr="/World/envs/.*/Table", name="table_view")
        # self._bunnies = BunnyView(prim_paths_expr="/World/envs/.*/Bunny", name="bunny_view")
        self._crackerboxes = RigidPrimView(prim_paths_expr="/World/envs/.*/CrackerBox", name="cracker_box_view", reset_xform_properties=False)
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball")
        scene.add(self._tables)
        scene.add(self._crackerboxes)
        scene.add(self._balls)
        return

    def get_table(self):
        table = Table(prim_path=self.default_zero_env_path + "/Table", name="table", translation=self._table_position)    
        self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))
    
    def get_crackerbox(self):
 
        crackerbox = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/CrackerBox", 
            translation=self._crackerbox_position, 
            name="crackerbox",
            size=0.1,
            color=torch.tensor([0.9, 0.6, 0.2]),
        )
        self._sim_config.apply_articulation_settings("crackerbox", get_prim_at_path(crackerbox.prim_path), self._sim_config.parse_actor_config("crackerbox"))
    
    def set_softbody(
        self, mesh_path: Sdf.Path, material_path: Sdf.Path, mass: float, resolution: int
    ):
        stage = get_current_stage()
        success = omni.kit.commands.execute(
            "AddDeformableBodyComponentCommand",
            skin_mesh_path=mesh_path,
            voxel_resolution=resolution,
            solver_position_iteration_count=22,
            self_collision=False,
            vertex_velocity_damping=0.005,
            sleep_damping=0.001,
            sleep_threshold=0.001,
            settling_threshold=0.001,
        )

        prim = stage.GetPrimAtPath(mesh_path)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        assert physxCollisionAPI.CreateRestOffsetAttr().Set(0.01)
        assert physxCollisionAPI.CreateContactOffsetAttr().Set(0.1)

        massAPI = UsdPhysics.MassAPI.Apply(prim)
        massAPI.CreateMassAttr().Set(mass)

        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(mesh_path), material_path)

        assert success

    def get_bunny(self):
        bunny = Bunny(prim_path=self.default_zero_env_path+"/Bunny", name="bunny", translation=self._bunny_position,)
        # if Bunny() is just a mesh prim without predefined soft body properties, set properties manually
        stage = get_current_stage()
        skin_mesh = UsdGeom.Mesh.Get(stage, self.default_zero_env_path+"/Bunny/bunny/bun_zipper_res3")
        assert deformableUtils.add_physx_deformable_body(stage, skin_mesh.GetPath(), collision_simplification=True, simulation_hexahedral_resolution=20,
             self_collision=False,)
        deformable_material_path = omni.usd.get_stage_next_free_path(stage, "/deformableBodyMaterial", True)
        assert deformableUtils.add_deformable_body_material(
            stage,
            deformable_material_path,
            youngs_modulus=100.0,
            poissons_ratio=0.49,
            damping_scale=0.0,
            dynamic_friction=0.5,
        )
        self.set_softbody(
                skin_mesh.GetPath(),
                deformable_material_path,
                0.01,
                42,
            )

    def get_target(self):
        radius = 0.02
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball", 
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:
        self.boxes_pos, self.boxes_rot = self._crackerboxes.get_world_poses(clone=False)
        self.boxes_velocities = self._crackerboxes.get_velocities(clone=False)

        boxes_positions = self.boxes_pos - self._env_pos
        boxes_linvels = self.boxes_velocities[:, :3]
        boxes_angvels = self.boxes_velocities[:, 3:]

        self.obs_buf[..., 0:3] = boxes_positions
        self.obs_buf[..., 3:6] = boxes_angvels / math.pi
        self.obs_buf[..., 6:9] = boxes_linvels
        self.obs_buf[..., 9:12] = self.target_positions
        self.obs_buf[..., 12:15] = boxes_positions-self.target_positions

        observations = {
            self._crackerboxes.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations
    
    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.clone().to(self._device)

        force_action_speed_scale = 10
        self.forces += self.dt * force_action_speed_scale * actions[:, 0:3]
        self.forces[:] = tensor_clamp(self.forces, self.force_lower_limits, self.force_upper_limits)

        # clear actions for reset envs
        self.forces[reset_env_ids] = 0.0
        
        # _, boxes_quat = self._crackerboxes.get_world_poses(clone=False)
        # boxes_quat = boxes_quat.reshape(self._num_envs, 4)

        # self.forces_world_frame = quat_apply(boxes_quat, self.forces)
        self.forces_world_frame = self.forces

        # apply actions
        self._crackerboxes.apply_forces(self.forces_world_frame)

    def post_reset(self):
        # control tensors
        self.forces = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device, requires_grad=False)
        self.forces_world_frame = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device, requires_grad=False)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device)
        self.target_positions[:, 2] = 0.85

        self.boxes_pos, self.boxes_rot = self._crackerboxes.get_world_poses(clone=False)
        self.boxes_velocities = self._crackerboxes.get_velocities(clone=False)
        self.initial_boxes_pos, self.initial_boxes_rot = self.boxes_pos.clone(), self.boxes_rot.clone()

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        boxes_pos = self.initial_boxes_pos.clone()
        boxes_pos[env_ids, 0] += torch_rand_float(-0.3, .3, (num_resets, 1), device=self._device).view(-1)
        boxes_pos[env_ids, 1] += torch_rand_float(-0.3, .3, (num_resets, 1), device=self._device).view(-1)
        boxes_pos[env_ids, 2] += torch_rand_float(-0.0, .05, (num_resets, 1), device=self._device).view(-1)
        boxes_velocities = self.boxes_velocities.clone()
        boxes_velocities[env_ids] = 0

        # apply resets
        self._crackerboxes.set_world_poses(boxes_pos[env_ids], self.initial_boxes_rot[env_ids].clone(), indices=env_ids)
        self._crackerboxes.set_velocities(boxes_velocities[env_ids], indices=env_ids)

        self._balls.set_world_poses(positions=self.target_positions[:, 0:3] + self._env_pos)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:
        boxes_positions = self.boxes_pos - self._env_pos
        boxes_angvels = self.boxes_velocities[:, 3:]
        boxes_linvels = self.boxes_velocities[:, 0:3]

        # distance to target
        target_dist = torch.sqrt(torch.square(self.target_positions- boxes_positions).sum(-1))
        pos_reward = 1.0 / (1.0 + 20*target_dist*target_dist)

        boxes_speed = torch.sqrt(torch.square(boxes_linvels).sum(-1))
        # speed_reward = 1.0 / (1.0 + boxes_speed*boxes_speed)
        speed_reward = 0.05 * torch.exp(-0.5 * boxes_speed)
        self.target_dist = target_dist
        self.boxes_positions = boxes_positions
        
        # height reward
        target_heighthdist = torch.sqrt(torch.square(self.target_positions[..., 2]- boxes_positions[..., 2]).sum(-1))
        heighth_reward = 0.1 * (torch.exp(-1.0 * target_heighthdist)+0.05*(boxes_positions[..., 2]-self.target_positions[..., 2]))

        # spinning
        spinnage = torch.square(boxes_angvels).sum(-1)
        spinnage_reward = 0.01 * torch.exp(-1.0 * spinnage)
        
        # forces reward
        effort = torch.square(self.forces).sum(-1)
        effort_reward = 0.01 * torch.exp(-1.0 * effort)

        # basic metric
        # rew = pos_reward
        # rew = pos_reward + pos_reward*speed_reward

        # rewards combination
        rew =  pos_reward + 0.1*heighth_reward * speed_reward + pos_reward * (spinnage_reward + heighth_reward - effort_reward) 
        rew = torch.clip(rew, 0.0, None)
        self.rew_buf[:] = rew
    
    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 2.0, ones, die)
        die = torch.where(self.boxes_positions[..., 2] < 0.3, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
