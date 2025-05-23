#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for loading MuJoCo models."""

from copy import copy
import os
from typing import Dict, Optional

from d4rl.kitchen.adept_envs.simulation import module
from d4rl.kitchen.adept_envs.simulation.renderer import (
    DMRenderer,
    MjPyRenderer,
    RenderMode,
)
import numpy as np
import xml.etree.ElementTree as ET


class MujocoSimRobot:
    """Class that encapsulates a MuJoCo simulation.

    This class exposes methods that are agnostic to the simulation backend.
    Two backends are supported:
    1. mujoco_py - MuJoCo v1.50
    2. dm_control - MuJoCo v2.00
    """

    def __init__(
        self,
        model_file: str,
        use_dm_backend: bool = False,
        camera_settings: Optional[Dict] = None,
    ):
        """Initializes a new simulation.

        Args:
            model_file: The MuJoCo XML model file to load.
            use_dm_backend: If True, uses DM Control's Physics (MuJoCo v2.0) as
              the backend for the simulation. Otherwise, uses mujoco_py (MuJoCo
              v1.5) as the backend.
            camera_settings: Settings to initialize the renderer's camera. This
              can contain the keys `distance`, `azimuth`, and `elevation`.
        """
        self._use_dm_backend = use_dm_backend

        if not os.path.isfile(model_file):
            raise ValueError(
                "[MujocoSimRobot] Invalid model file path: {}".format(model_file)
            )
        self.model_file = model_file
        if self._use_dm_backend:
            dm_mujoco = module.get_dm_mujoco()
            if model_file.endswith(".mjb"):
                self.sim = dm_mujoco.Physics.from_binary_path(model_file)
            else:
                self.sim = dm_mujoco.Physics.from_xml_path(model_file)
            self.model = self.sim.model
            self._patch_mjlib_accessors(self.model, self.sim.data)
            self.renderer = DMRenderer(self.sim, camera_settings=camera_settings)
        else:  # Use mujoco_py
            mujoco_py = module.get_mujoco_py()
            self.model = mujoco_py.load_model_from_path(model_file)
            self.sim = mujoco_py.MjSim(self.model)
            self.renderer = MjPyRenderer(self.sim, camera_settings=camera_settings)

        self.data = self.sim.data

    def close(self):
        """Cleans up any resources being used by the simulation."""
        self.renderer.close()

    def save_binary(self, path: str):
        """Saves the loaded model to a binary .mjb file."""
        if os.path.exists(path):
            raise ValueError("[MujocoSimRobot] Path already exists: {}".format(path))
        if not path.endswith(".mjb"):
            path = path + ".mjb"
        if self._use_dm_backend:
            self.model.save_binary(path)
        else:
            with open(path, "wb") as f:
                f.write(self.model.get_mjb())

    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        if self._use_dm_backend:
            return module.get_dm_mujoco().wrapper.mjbindings.mjlib
        else:
            return module.get_mujoco_py_mjlib()

    def _patch_mjlib_accessors(self, model, data):
        """Adds accessors to the DM Control objects to support mujoco_py API.
        obtained from https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi
        """
        mjlib = self.get_mjlib()
        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(
                model.ptr, mjlib.mju_str2Type(type_name.encode()), name.encode()
            )
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(type_name, name))
            return obj_id

        def id2name(type_name, id):
            obj_name = mjlib.mj_id2name(
                model.ptr, mjlib.mju_str2Type(type_name.encode()), id
            )
            return obj_name

        if not hasattr(model, "body_name2id"):
            model.body_name2id = lambda name: name2id("body", name)

        if not hasattr(model, "geom_name2id"):
            model.geom_name2id = lambda name: name2id("geom", name)

        if not hasattr(model, "geom_id2name"):
            model.geom_id2name = lambda id: id2name("geom", id)

        if not hasattr(model, "site_name2id"):
            model.site_name2id = lambda name: name2id("site", name)

        if not hasattr(model, "joint_name2id"):
            model.joint_name2id = lambda name: name2id("joint", name)

        if not hasattr(model, "actuator_name2id"):
            model.actuator_name2id = lambda name: name2id("actuator", name)

        if not hasattr(model, "camera_name2id"):
            model.camera_name2id = lambda name: name2id("camera", name)

        if not hasattr(model, "sensor_name2id"):
            model.sensor_name2id = lambda name: name2id("sensor", name)

        if not hasattr(model, "get_joint_qpos_addr"):

            def get_joint_qpos_addr(name):
                joint_id = model.joint_name2id(name)
                joint_type = model.jnt_type[joint_id]
                joint_addr = model.jnt_qposadr[joint_id]
                # TODO: remove hardcoded joint ids (find where mjtJoint is)
                if joint_type == 0:
                    ndim = 7
                elif joint_type == 1:
                    ndim = 4
                else:
                    assert joint_type in (2, 3)
                    ndim = 1

                if ndim == 1:
                    return joint_addr
                else:
                    return (joint_addr, joint_addr + ndim)

            model.get_joint_qpos_addr = lambda name: get_joint_qpos_addr(name)

        if not hasattr(model, "get_joint_qvel_addr"):

            def get_joint_qvel_addr(name):
                joint_id = model.joint_name2id(name)
                joint_type = model.jnt_type[joint_id]
                joint_addr = model.jnt_dofadr[joint_id]
                if joint_type == 0:
                    ndim = 6
                elif joint_type == 1:
                    ndim = 3
                else:
                    assert joint_type in (3, 2)
                    ndim = 1

                if ndim == 1:
                    return joint_addr
                else:
                    return (joint_addr, joint_addr + ndim)

            model.get_joint_qvel_addr = lambda name: get_joint_qvel_addr(name)
        
        def get_xml():
            temp_file = "/tmp/temp_mujoco_model.xml"
            mjlib.mj_saveLastXML(temp_file, model.ptr)
            with open(temp_file, "r") as f:
                xml = f.read()
            os.remove(temp_file)
            xml_root = ET.fromstring(xml)
            # convert all ../ to absolute path
            for elem in xml_root.iter():
                for attr in elem.attrib:
                    # if "../" in elem.attrib[attr]:
                    #     elem.attrib[attr] = elem.attrib[attr].replace("../", "")
                    if "../" in elem.attrib[attr]:
                        orig = copy(elem.attrib[attr].split("/")[0])
                        while "../" in elem.attrib[attr]:
                            elem.attrib[attr] = elem.attrib[attr].replace("../", "")
                        # extract first part of the path
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == "adept_models":
                            import d4rl.kitchen as kitchen
                            # get the absolute path
                            abs_path = os.path.join(kitchen.__path__[0], elem.attrib[attr])
                        elif first_part == 'scenes':
                            import d4rl.kitchen.adept_models as adept_models
                            abs_path = os.path.join(adept_models.__path__[0], elem.attrib[attr])
                        elif first_part == 'kitchen':
                            import d4rl.kitchen.adept_models as adept_models
                            abs_path = os.path.join(adept_models.__path__[0], elem.attrib[attr])
                        elif first_part == 'third_party':
                            import d4rl.kitchen as kitchen
                            abs_path = os.path.join(kitchen.__path__[0], elem.attrib[attr])
                        else:
                            print(first_part, orig)
                        elem.attrib[attr] = abs_path
            xml = ET.tostring(xml_root, encoding='unicode')   
            # save to file
            abs_path_temp_file = "/home/mdalal/temp_mujoco_model_abs_path.xml"
            with open(abs_path_temp_file, "w") as f:
                f.write(xml)
            return xml
        # load xml file
        model.get_xml = lambda: get_xml()
            
        if not hasattr(data, "body_xpos"):
            data.body_xpos = data.xpos

        if not hasattr(data, "body_xquat"):
            data.body_xquat = data.xquat

        if not hasattr(data, "body_xmat"):
            data.body_xmat = data.xmat

        if not hasattr(data, "get_body_xpos"):
            data.get_body_xpos = lambda name: data.body_xpos[model.body_name2id(name)]

        if not hasattr(data, "get_body_xquat"):
            data.get_body_xquat = lambda name: data.body_xquat[model.body_name2id(name)]

        if not hasattr(data, "get_body_xmat"):
            data.get_body_xmat = lambda name: data.xmat[model.body_name2id(name)].reshape(
                (3, 3)
            )

        if not hasattr(data, "get_geom_xpos"):
            data.get_geom_xpos = lambda name: data.geom_xpos[model.geom_name2id(name)]

        if not hasattr(data, "get_geom_xquat"):
            data.get_geom_xquat = lambda name: data.geom_xquat[model.geom_name2id(name)]

        if not hasattr(data, "get_joint_qpos"):

            def get_joint_qpos(name):
                addr = model.get_joint_qpos_addr(name)
                if isinstance(addr, (int, np.int32, np.int64)):
                    return data.qpos[addr]
                else:
                    start_i, end_i = addr
                    return data.qpos[start_i:end_i]

            data.get_joint_qpos = lambda name: get_joint_qpos(name)

        if not hasattr(data, "set_joint_qpos"):

            def set_joint_qpos(name, value):
                addr = model.get_joint_qpos_addr(name)
                if isinstance(addr, (int, np.int32, np.int64)):
                    data.qpos[addr] = value
                else:
                    start_i, end_i = addr
                    value = np.array(value)
                    assert value.shape == (
                        end_i - start_i,
                    ), "Value has incorrect shape %s: %s" % (name, value)
                    data.qpos[start_i:end_i] = value

            data.set_joint_qpos = lambda name, value: set_joint_qpos(name, value)

        if not hasattr(data, "get_site_xmat"):
            data.get_site_xmat = lambda name: data.site_xmat[
                model.site_name2id(name)
            ].reshape((3, 3))

        if not hasattr(data, "get_geom_xmat"):
            data.get_geom_xmat = lambda name: data.geom_xmat[
                model.geom_name2id(name)
            ].reshape((3, 3))

        if not hasattr(data, "get_mocap_pos"):
            data.get_mocap_pos = lambda name: data.mocap_pos[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "get_mocap_quat"):
            data.get_mocap_quat = lambda name: data.mocap_quat[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "set_mocap_pos"):

            def set_mocap_pos(name, value):
                data.mocap_pos[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_pos = lambda name, value: set_mocap_pos(name, value)

        if not hasattr(data, "set_mocap_quat"):

            def set_mocap_quat(name, value):
                data.mocap_quat[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_quat = lambda name, value: set_mocap_quat(name, value)

        def site_jacp():
            jacps = np.zeros((model.nsite, 3 * model.nv))
            for i, jacp in enumerate(jacps):
                jacp_view = jacp.reshape((3, -1))
                mjlib.mj_jacSite(model.ptr, data.ptr, jacp_view, None, i)
            return jacps

        def site_xvelp():
            jacp = site_jacp().reshape((model.nsite, 3, model.nv))
            xvelp = np.dot(jacp, data.qvel)
            return xvelp

        def site_jacr():
            jacrs = np.zeros((model.nsite, 3 * model.nv))
            for i, jacr in enumerate(jacrs):
                jacr_view = jacr.reshape((3, -1))
                mjlib.mj_jacSite(model.ptr, data.ptr, None, jacr_view, i)
            return jacrs

        def site_xvelr():
            jacr = site_jacr().reshape((model.nsite, 3, model.nv))
            xvelr = np.dot(jacr, data.qvel)
            return xvelr

        if not hasattr(data, "site_xvelp"):
            data.site_xvelp = site_xvelp()

        if not hasattr(data, "site_xvelr"):
            data.site_xvelr = site_xvelr()

        if not hasattr(data, "get_site_jacp"):
            data.get_site_jacp = lambda name: site_jacp()[model.site_name2id(name)].reshape(
                (3, model.nv)
            )

        if not hasattr(data, "get_site_jacr"):
            data.get_site_jacr = lambda name: site_jacr()[model.site_name2id(name)].reshape(
                (3, model.nv)
            )

        def body_jacp():
            jacps = np.zeros((model.nbody, 3 * model.nv))
            for i, jacp in enumerate(jacps):
                jacp_view = jacp.reshape((3, -1))
                mjlib.mj_jacBody(model.ptr, data.ptr, jacp_view, None, i)
            return jacps

        def body_xvelp():
            jacp = body_jacp().reshape((model.nbody, 3, model.nv))
            xvelp = np.dot(jacp, data.qvel)
            return xvelp

        def body_jacr():
            jacrs = np.zeros((model.nbody, 3 * model.nv))
            for i, jacr in enumerate(jacrs):
                jacr_view = jacr.reshape((3, -1))
                mjlib.mj_jacBody(model.ptr, data.ptr, None, jacr_view, i)
            return jacrs

        def body_xvelr():
            jacp = body_jacr().reshape((model.nbody, 3, model.nv))
            xvelp = np.dot(jacp, data.qvel)
            return xvelp

        if not hasattr(data, "body_xvelp"):
            data.body_xvelp = body_xvelp()

        if not hasattr(data, "body_xvelr"):
            data.body_xvelr = body_xvelr()

        if not hasattr(data, "get_body_jacp"):
            data.get_body_jacp = lambda name: body_jacp()[model.body_name2id(name)].reshape(
                (3, model.nv)
            )

        if not hasattr(data, "get_body_jacr"):
            data.get_body_jacr = lambda name: body_jacr()[model.body_name2id(name)].reshape(
                (3, model.nv)
            )
