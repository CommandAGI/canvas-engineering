"""Multi-robot MuJoCo scene for the canvas-attention animation.

Three mobile manipulators share a walled arena with coloured cubes and goal
pads. Each robot is a holonomic base (slide x, slide y, hinge yaw) carrying a
two-joint arm (extend, lift) that ends in a suction cup (MuJoCo adhesion
actuator). Per-robot modalities:

    video    64x64 RGB from a forward-facing onboard camera
    lidar    8 planar rangefinder beams
    proprio  base x, y, yaw, vx, vy, yaw rate, arm extension, arm lift
    touch    suction-cup contact force, adhesion command

Actions are split into a gross movement command (vx, vy, yaw rate) and a fine
manipulator command (extension, lift, suction). A scripted controller drives
each robot through pick -> carry -> place.

Run with MUJOCO_GL=glfw (EGL is unavailable on this machine).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

N_ROBOTS = 3
N_LIDAR = 8
CTRL_DT = 0.1
PHYS_DT = 0.004
ARENA = (2.0, 1.5)  # half extents, metres

ROBOT_RGBA = ["0.93 0.55 0.25 1", "0.35 0.55 0.85 1", "0.45 0.72 0.42 1"]
ROBOT_NAMES = ["A", "B", "C"]
CUBES = {  # name: (rgba, start xy)
    "red": ("0.85 0.2 0.2 1", (-0.9, 0.9)),
    "blue": ("0.2 0.35 0.9 1", (1.3, 0.8)),
    "green": ("0.2 0.7 0.3 1", (0.2, -1.05)),
    "yellow": ("0.95 0.8 0.15 1", (-1.35, -0.6)),
}
PADS = {"red": (1.45, -1.05), "blue": (-1.5, 0.95), "green": (-0.3, 1.1), "yellow": (1.0, 0.1)}
ROBOT_START = [(-1.5, 0.1, 0.4), (1.5, -0.2, 2.6), (0.2, 0.2, -1.5)]
# (cube, destination) per robot, in order; second rounds re-shuffle cubes another robot delivered
TASKS = [
    [("red", PADS["red"]), ("yellow", PADS["yellow"])],
    [("blue", PADS["blue"]), ("green", (1.45, 1.05))],
    [("green", PADS["green"]), ("red", (-1.45, -1.05))],
]
STANDOFF = 0.30  # base-centre to cup-centre distance at the pick pose


def build_xml() -> str:
    hx, hy = ARENA
    walls = "".join(
        f'<geom type="box" pos="{x} {y} 0.15" size="{sx} {sy} 0.15" rgba="0.82 0.82 0.8 1" contype="5" conaffinity="5"/>'
        for x, y, sx, sy in [(0, hy + 0.05, hx + 0.1, 0.05), (0, -hy - 0.05, hx + 0.1, 0.05),
                             (hx + 0.05, 0, 0.05, hy), (-hx - 0.05, 0, 0.05, hy)]
    )
    shelves = "".join(
        f'<geom type="box" pos="{x} {y} 0.2" size="{sx} {sy} 0.2" euler="0 0 {a}" rgba="0.55 0.5 0.45 1" contype="5" conaffinity="5"/>'
        for x, y, sx, sy, a in [(-0.75, -1.3, 0.3, 0.06, 0), (1.72, 1.15, 0.06, 0.25, 0)]
    )
    pads = "".join(
        f'<geom type="cylinder" pos="{x} {y} 0.001" size="0.14 0.001" rgba="{CUBES[c][0][:-2]} 0.35" contype="0" conaffinity="0"/>'
        for c, (x, y) in PADS.items()
    )
    cubes = "".join(
        f'<body name="cube_{c}" pos="{x} {y} 0.036"><freejoint/>'
        f'<geom type="box" size="0.035 0.035 0.035" rgba="{rgba}" mass="0.08" contype="7" conaffinity="7" friction="1 0.01 0.001"/></body>'
        for c, (rgba, (x, y)) in CUBES.items()
    )
    robots = []
    for i in range(N_ROBOTS):
        x, y, yaw = ROBOT_START[i]
        lidar_sites = "".join(
            f'<site name="r{i}_lidar{k}" pos="{0.17 * math.cos(a):.4f} {0.17 * math.sin(a):.4f} 0.08" '
            f'zaxis="{math.cos(a):.4f} {math.sin(a):.4f} 0" size="0.005" rgba="0 0 0 0"/>'
            for k, a in enumerate(np.linspace(0, 2 * math.pi, N_LIDAR, endpoint=False))
        )
        robots.append(f"""
    <body name="r{i}" pos="{x} {y} 0">
      <joint name="r{i}_x" type="slide" axis="1 0 0" damping="4"/>
      <joint name="r{i}_y" type="slide" axis="0 1 0" damping="4"/>
      <joint name="r{i}_yaw" type="hinge" axis="0 0 1" damping="1"/>
      <geom name="r{i}_base" type="cylinder" size="0.16 0.05" pos="0 0 0.06" rgba="{ROBOT_RGBA[i]}" mass="3" contype="4" conaffinity="4"/>
      <geom type="box" size="0.03 0.03 0.08" pos="-0.07 0 0.18" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0" mass="0.1"/>
      <camera name="cam{i}" pos="-0.04 0 0.27" xyaxes="0 -1 0 0.42 0 0.91" fovy="80"/>
      {lidar_sites}
      <body name="r{i}_arm" pos="0.12 0 0.14">
        <joint name="r{i}_ext" type="slide" axis="1 0 0" range="0 0.16" damping="15"/>
        <joint name="r{i}_lift" type="slide" axis="0 0 1" range="-0.1 0.06" damping="15"/>
        <geom type="box" size="0.08 0.015 0.012" pos="0 0 0" rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0" mass="0.2"/>
        <body name="r{i}_cup" pos="0.08 0 -0.025">
          <geom name="r{i}_cupg" type="cylinder" size="0.03 0.008" rgba="0.1 0.1 0.1 1" mass="0.05" contype="2" conaffinity="2" margin="0.006" gap="0.006"/>
          <site name="r{i}_touch" type="cylinder" size="0.034 0.014" rgba="0 0 0 0"/>
        </body>
      </body>
    </body>""")
    acts, sens = [], []
    for i in range(N_ROBOTS):
        acts += [
            f'<velocity joint="r{i}_x" kv="60" ctrlrange="-0.6 0.6"/>',
            f'<velocity joint="r{i}_y" kv="60" ctrlrange="-0.6 0.6"/>',
            f'<velocity joint="r{i}_yaw" kv="6" ctrlrange="-2.5 2.5"/>',
            f'<position joint="r{i}_ext" kp="300" ctrlrange="0 0.16"/>',
            f'<position joint="r{i}_lift" kp="400" ctrlrange="-0.1 0.06"/>',
            f'<adhesion body="r{i}_cup" ctrlrange="0 1" gain="12"/>',
        ]
        sens += [f'<touch name="r{i}_touch" site="r{i}_touch"/>']
        sens += [f'<rangefinder name="r{i}_lidar{k}" site="r{i}_lidar{k}" cutoff="3"/>' for k in range(N_LIDAR)]
    return f"""
<mujoco model="canvas_robots">
  <option timestep="{PHYS_DT}" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.93 0.91" rgb2="0.86 0.86 0.84" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="8 6"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="2 -2 3" dir="-0.5 0.5 -1" diffuse="0.3 0.3 0.3"/>
    <geom type="plane" size="{hx} {hy} 0.1" material="floor" contype="1" conaffinity="1"/>
    <camera name="top" pos="0 0 5.2" xyaxes="1 0 0 0 1 0" fovy="42"/>
    {walls}{shelves}{pads}{cubes}{''.join(robots)}
  </worldbody>
  <actuator>{''.join(acts)}</actuator>
  <sensor>{''.join(sens)}</sensor>
</mujoco>"""


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Controller:
    """Scripted pick -> carry -> place state machine for one robot."""
    idx: int
    tasks: list
    phase: str = "goto_cube"
    timer: int = 0
    lift: float = 0.04
    ext: float = 0.02
    suction: float = 0.0

    def target(self, sim: "Sim"):
        if not self.tasks:
            return None
        c, dest = self.tasks[0]
        if self.phase in ("goto_cube", "lower", "grab", "raise"):
            return sim.cube_xy(c)
        return np.array(dest)

    def act(self, sim: "Sim"):
        x, y, yaw = sim.pose(self.idx)
        tgt = self.target(sim)
        move = np.zeros(3)
        if tgt is None:  # idle: slow patrol turn
            move[2] = 0.4
            return move, np.array([self.ext, self.lift, self.suction])
        d = tgt - np.array([x, y])
        dist = float(np.linalg.norm(d))
        head = math.atan2(d[1], d[0])
        err = wrap(head - yaw)
        if self.phase == "carry" and sim.data.sensor(f"r{self.idx}_touch").data[0] < 0.5:
            self.lost = getattr(self, "lost", 0) + 1
            if self.lost > 3:  # cube slipped off the cup: go back for it
                self.phase, self.lost, self.suction = "goto_cube", 0, 0.0
        else:
            self.lost = 0
        if self.phase in ("goto_cube", "carry"):
            move[2] = float(np.clip(3.0 * err, -2.0, 2.0))
            if abs(err) < 0.6:
                speed = float(np.clip(1.6 * (dist - STANDOFF), -0.3, 0.5))
                v = speed * np.array([math.cos(yaw), math.sin(yaw)])
                # keep clear of the other robots
                for j in range(N_ROBOTS):
                    if j == self.idx:
                        continue
                    o = np.array([x, y]) - sim.pose(j)[:2]
                    r = float(np.linalg.norm(o))
                    if r < 0.65:
                        v += 0.45 * (0.65 - r) / 0.65 * o / max(r, 1e-3)
                move[:2] = v
            self.ext, self.lift = 0.10, 0.04
            if abs(dist - STANDOFF) < 0.025 and abs(err) < 0.06:
                self.phase = "lower" if self.phase == "goto_cube" else "place"
                self.timer = 0
        elif self.phase == "lower":
            move[2] = float(np.clip(3.0 * err, -1, 1))
            self.lift, self.suction = -0.095, 1.0
            self.timer += 1
            if self.timer > 6:
                self.phase, self.timer = "raise", 0
        elif self.phase == "raise":
            self.lift = 0.05
            self.timer += 1
            if self.timer > 4:
                self.phase = "carry"
        elif self.phase == "place":
            self.lift = -0.07
            self.timer += 1
            if self.timer > 4:
                self.suction = 0.0
            if self.timer > 7:
                self.lift = 0.05
            if self.timer > 9:
                self.tasks.pop(0)
                self.phase, self.timer = "goto_cube", 0
        return move, np.array([self.ext, self.lift, self.suction])


class Sim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        m = self.model
        self.jx = [m.joint(f"r{i}_x").qposadr[0] for i in range(N_ROBOTS)]
        self.jy = [m.joint(f"r{i}_y").qposadr[0] for i in range(N_ROBOTS)]
        self.jyaw = [m.joint(f"r{i}_yaw").qposadr[0] for i in range(N_ROBOTS)]
        self.jext = [m.joint(f"r{i}_ext").qposadr[0] for i in range(N_ROBOTS)]
        self.jlift = [m.joint(f"r{i}_lift").qposadr[0] for i in range(N_ROBOTS)]
        self.vx = [m.joint(f"r{i}_x").dofadr[0] for i in range(N_ROBOTS)]
        self.vy = [m.joint(f"r{i}_y").dofadr[0] for i in range(N_ROBOTS)]
        self.vyaw = [m.joint(f"r{i}_yaw").dofadr[0] for i in range(N_ROBOTS)]
        self.start = [np.array(s) for s in ROBOT_START]
        for i in range(N_ROBOTS):  # start yaw lives in the joint so the x/y slides stay world-aligned
            self.data.qpos[self.jyaw[i]] = ROBOT_START[i][2]
            self.data.qpos[self.jlift[i]] = 0.04
        mujoco.mj_forward(m, self.data)
        self.ctrls = [Controller(i, list(TASKS[i])) for i in range(N_ROBOTS)]
        self.ego = mujoco.Renderer(m, 64, 64)
        self.top = mujoco.Renderer(m, 360, 270)
        self.vopt = mujoco.MjvOption()
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    # base qpos is relative to the body's start pose (x, y slides are world-aligned)
    def pose(self, i):
        q = self.data.qpos
        s = self.start[i]
        return np.array([s[0] + q[self.jx[i]], s[1] + q[self.jy[i]], wrap(q[self.jyaw[i]])])

    def cube_xy(self, c):
        return self.data.body(f"cube_{c}").xpos[:2].copy()

    def observe(self, i):
        d = self.data
        self.ego.update_scene(d, camera=f"cam{i}", scene_option=self.vopt)
        img = self.ego.render().copy()
        lidar = np.array([d.sensor(f"r{i}_lidar{k}").data[0] for k in range(N_LIDAR)])
        lidar = np.where(lidar < 0, 3.0, lidar)
        qv = d.qvel
        p = self.pose(i)
        proprio = np.array([p[0], p[1], math.cos(p[2]), math.sin(p[2]),
                            qv[self.vx[i]], qv[self.vy[i]], qv[self.vyaw[i]],
                            d.qpos[self.jext[i]], d.qpos[self.jlift[i]]])
        touch = np.array([d.sensor(f"r{i}_touch").data[0], self.ctrls[i].suction])
        return dict(video=img, lidar=lidar, proprio=proprio, touch=touch)

    def render_top(self):
        self.top.update_scene(self.data, camera="top", scene_option=self.vopt)
        return self.top.render().copy()

    def step(self):
        """One control tick. Returns per-robot (move, manip) commands."""
        cmds = []
        for i, c in enumerate(self.ctrls):
            move, manip = c.act(self)
            yaw = self.pose(i)[2]
            # world-frame base velocity -> the world-aligned slide joints
            self.data.ctrl[6 * i:6 * i + 3] = move
            self.data.ctrl[6 * i + 3:6 * i + 6] = manip
            cmds.append((move.copy(), manip.copy(), c.phase))
        for _ in range(int(round(CTRL_DT / PHYS_DT))):
            mujoco.mj_step(self.model, self.data)
        return cmds


if __name__ == "__main__":
    sim = Sim()
    for t in range(380):
        cmds = sim.step()
        if t % 10 == 0:
            print(t, [c[2] for c in cmds], [np.round(sim.observe(i)["touch"], 2).tolist() for i in range(N_ROBOTS)],
                  {c: np.round(sim.data.body(f"cube_{c}").xpos, 2).tolist() for c in CUBES})
