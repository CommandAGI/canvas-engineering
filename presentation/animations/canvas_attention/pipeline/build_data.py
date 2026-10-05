"""Run the multi-robot sim, push it through a canvas DiT, and record real QK^T.

Pipeline (inference only, nothing is trained):

1. sim.py rolls out three scripted mobile manipulators in MuJoCo for N ticks.
2. Every tick becomes one 20 x 48 canvas slice (three 16-column robot strips):
     video  16x16   onboard camera, 4x4-pixel patches of a 64x64 frame
     lidar   2x4    8 rangefinder beams
     proprio 3x3    base pose/velocity + arm joints
     touch   1x2    suction contact force + suction command
     think   4x4    latent "thinking" tokens (central)
     move    1x1    gross base-movement readout (vx, vy, yaw rate)
     manip   1x1    fine manipulator readout (extend, lift, suction)
   declared with canvas_engineering's CanvasLayout and CanvasTopology.
3. A small, randomly initialised canvas DiT runs one reverse-diffusion step
   over an 8-tick window ending at each tick. Each block is depthwise-separable
   in spirit: an intra-region AttentionDispatcher (self edges), then an
   inter-region AttentionDispatcher (cross edges), then an MLP, all adaLN-
   conditioned on the diffusion timestep.
4. For the last frame of each window we recompute the dispatcher's own
   softmax(QK^T / sqrt(d)) (same projections, same hidden states, heads
   averaged) for the movement, manipulator and thinking queries of every robot,
   and keep the top keys of every edge.

The weights are random, so *which* blocks match is not learned behaviour; the
scores are the real QK^T of this network on this rollout.

Usage:  MUJOCO_GL=glfw python build_data.py   (writes ../assets/)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))  # repo root -> canvas_engineering
sys.path.insert(0, str(HERE))

from canvas_engineering import (  # noqa: E402
    AttentionDispatcher, CanvasLayout, CanvasTopology, Connection, RegionSpec,
)
from canvas_engineering.canvas import SinusoidalPositionalEncodingND  # noqa: E402

import sim as S  # noqa: E402

N_TICKS = int(os.environ.get("N_TICKS", 160))
WIN = 8                 # ticks denoised jointly
D, HEADS, DEPTH = 96, 4, 4
ALPHA_BAR = 0.5         # noise level of the recorded denoising step
TAU = 500               # matching diffusion timestep index (of 1000)
TOPK = 8
SEED = 7
OUT = HERE.parent / "assets"

H, W = 20, 48
# local (row0, row1, col0, col1) inside a 20 x 16 robot strip
LOCAL = {
    "video": (0, 16, 0, 16),
    "lidar": (16, 18, 0, 4),
    "touch": (18, 19, 0, 2),
    "think": (16, 20, 6, 10),
    "proprio": (16, 19, 11, 14),
    "move": (19, 20, 11, 12),
    "manip": (19, 20, 13, 14),
}
SEMANTIC = {
    "video": "64x64 RGB from the robot's onboard camera, 4x4-pixel patches",
    "lidar": "8-beam planar rangefinder, metres",
    "touch": "suction-cup contact force and suction command",
    "think": "latent thinking tokens",
    "proprio": "base pose, base velocity, arm extension and lift",
    "move": "gross base movement command: vx, vy, yaw rate",
    "manip": "fine manipulator command: extend, lift, suction",
}
OUTPUTS = {"video", "move", "manip"}  # noised at the last frame of each window
ROBOTS = S.ROBOT_NAMES


def rname(r, kind):
    return f"{ROBOTS[r]}.{kind}"


def build_layout(T: int) -> CanvasLayout:
    regions = {}
    for r in range(S.N_ROBOTS):
        o = 16 * r
        for kind, (r0, r1, c0, c1) in LOCAL.items():
            regions[rname(r, kind)] = RegionSpec(
                bounds=(0, T, r0, r1, o + c0, o + c1),
                is_output=kind in OUTPUTS, semantic_type=SEMANTIC[kind],
            )
    return CanvasLayout(T=T, H=H, W=W, d_model=D, regions=regions)


def build_topologies():
    intra, inter = [], []
    for r in range(S.N_ROBOTS):
        n = lambda k: rname(r, k)  # noqa: E731
        # depthwise: each region attends within itself
        for k in ("video", "lidar", "proprio", "touch"):
            intra.append(Connection(src=n(k), dst=n(k), t_src=0, t_dst=0))
        intra.append(Connection(src=n("think"), dst=n("think"), t_src=0))  # own trajectory, whole window
        # pointwise: declared cross-region edges
        for k in ("video", "lidar", "proprio", "touch"):
            inter.append(Connection(src=n("think"), dst=n(k), t_src=0, t_dst=0))
        for s in range(S.N_ROBOTS):
            if s != r:
                inter.append(Connection(src=n("think"), dst=rname(s, "think"), t_src=0, t_dst=-1))
        for q, keys in (("move", ("video", "lidar", "proprio")), ("manip", ("video", "touch", "proprio"))):
            inter.append(Connection(src=n(q), dst=n("think"), t_src=0, t_dst=0))
            for k in keys:
                inter.append(Connection(src=n(q), dst=n(k), t_src=0, t_dst=-1))
        inter.append(Connection(src=n("video"), dst=n("think"), t_src=0, t_dst=-1))  # prediction reads belief
    return CanvasTopology(connections=intra), CanvasTopology(connections=inter)


class Block(nn.Module):
    def __init__(self, layout, intra, inter):
        super().__init__()
        self.ada = nn.Linear(D, 6 * D)
        self.n1 = nn.LayerNorm(D, elementwise_affine=False)
        self.n2 = nn.LayerNorm(D, elementwise_affine=False)
        self.n3 = nn.LayerNorm(D, elementwise_affine=False)
        self.intra = AttentionDispatcher(intra, layout, D, HEADS)
        self.inter = AttentionDispatcher(inter, layout, D, HEADS)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        # positions that are queries of some edge; the dispatcher passes every other position through
        for name, topo in (("intra_q", intra), ("inter_q", inter)):
            m = torch.zeros(layout.num_positions, 1)
            for cn in topo.connections:
                m[layout.region_indices(cn.src)] = 1.0
            self.register_buffer(name, m)

    def forward(self, x, c):
        b1, s1, b2, s2, b3, s3 = self.ada(c).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + self.intra(h) * self.intra_q
        h = self.n2(x) * (1 + s2) + b2
        x = x + self.inter(h) * self.inter_q
        return x + self.mlp(self.n3(x) * (1 + s3) + b3)


class CanvasDiT(nn.Module):
    def __init__(self, layout, intra, inter):
        super().__init__()
        self.layout = layout
        self.enc = nn.ModuleDict({
            "video": nn.Linear(48, D), "lidar": nn.Linear(1, D), "proprio": nn.Linear(1, D),
            "touch": nn.Linear(1, D), "move": nn.Linear(3, D), "manip": nn.Linear(3, D),
        })
        self.think0 = nn.Parameter(torch.randn(16, D) * 0.5)
        self.kind_emb = nn.ParameterDict({k: nn.Parameter(torch.randn(D) * 0.3) for k in LOCAL})
        self.robot_emb = nn.Parameter(torch.randn(S.N_ROBOTS, D) * 0.3)
        self.pos = SinusoidalPositionalEncodingND(D, max_T=64, max_spatial=(H, W))
        self.temb = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.blocks = nn.ModuleList([Block(layout, intra, inter) for _ in range(DEPTH)])

    def timestep_embedding(self, tau):
        half = D // 2
        f = torch.exp(-math.log(10000) * torch.arange(half) / half)
        a = tau * f
        return self.temb(torch.cat([a.sin(), a.cos()])[None])

    def embed(self, feats):
        """feats[r][kind] -> (T, n_tokens, in_dim) tensors. Returns (1, N, D) clean canvas."""
        L = self.layout
        x = torch.zeros(L.num_positions, D)
        for r in range(S.N_ROBOTS):
            for kind in LOCAL:
                name = rname(r, kind)
                if kind == "think":
                    tok = self.think0[None].expand(L.T, -1, -1)
                else:
                    tok = self.enc[kind](feats[r][kind])
                tok = tok + self.kind_emb[kind] + self.robot_emb[r]
                for t in range(L.T):
                    idx = L.region_indices_at_t(name, t)
                    x[idx] = tok[t]
        pe = self.pos(L.T, (H, W)).reshape(-1, D)
        return (x + pe)[None]

    def forward(self, x, tau):
        c = self.timestep_embedding(torch.tensor(float(tau)))
        for b in self.blocks:
            x = b(x, c)
        return x


def edge_attention(disp: AttentionDispatcher, h: torch.Tensor, conn: Connection, layout: CanvasLayout,
                   q_local: list[int] | None):
    """softmax(QK^T/sqrt(d)) the dispatcher computes for one edge at the window's last frame.

    Returns (query positions, key positions, weights[n_q, n_k]) with heads averaged.
    """
    ref = layout.T - 1
    s_idx = layout.region_indices_at_t(conn.src, ref + (conn.t_src or 0))
    if conn.t_dst is None:
        d_idx = layout.region_indices(conn.dst)
    else:
        d_idx = layout.region_indices_at_t(conn.dst, ref + conn.t_dst)
    if not s_idx or not d_idx:
        return None
    if q_local is not None:
        s_idx = [s_idx[i] for i in q_local]
    fn = disp.fn_modules[disp.topology.resolve_fn(conn, layout)]
    q = fn.q_proj(h[0, s_idx]).view(len(s_idx), HEADS, -1).transpose(0, 1)
    k = fn.k_proj(h[0, d_idx]).view(len(d_idx), HEADS, -1).transpose(0, 1)
    a = F.softmax((q @ k.transpose(-1, -2)) * fn.scale, dim=-1).mean(0)
    return s_idx, d_idx, a


def pos_to_cell(layout, flat):
    t, rem = divmod(flat, H * W)
    return t, rem // W, rem % W


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "top").mkdir(exist_ok=True)

    # ---- 1. roll out the sim -------------------------------------------------------
    sim = S.Sim()
    rec = {r: {k: [] for k in ("video", "lidar", "proprio", "touch", "move", "manip")} for r in range(S.N_ROBOTS)}
    world = []
    for t in range(N_TICKS):
        cmds = sim.step()
        for r in range(S.N_ROBOTS):
            ob = sim.observe(r)
            for k in ("video", "lidar", "proprio", "touch"):
                rec[r][k].append(ob[k])
            rec[r]["move"].append(cmds[r][0])
            rec[r]["manip"].append(cmds[r][1])
        Image.fromarray(sim.render_top()).save(OUT / "top" / f"{t:03d}.jpg", quality=72)
        world.append({
            "phase": [c[2] for c in cmds],
            "touch": [round(float(rec[r]["touch"][-1][0]), 2) for r in range(S.N_ROBOTS)],
            "pose": [np.round(sim.pose(r), 3).tolist() for r in range(S.N_ROBOTS)],
        })
    # top-down sprite sheet: 16 columns of 300x225 tiles, one per tick
    cols, tw, th = 16, 300, 225
    sheet = Image.new("RGB", (cols * tw, math.ceil(N_TICKS / cols) * th), "white")
    for t in range(N_TICKS):
        im = Image.open(OUT / "top" / f"{t:03d}.jpg").resize((tw, th), Image.LANCZOS)
        sheet.paste(im, ((t % cols) * tw, (t // cols) * th))
    sheet.save(OUT / "top_sprite.jpg", quality=70, optimize=True)
    # onboard camera sprite: one row per robot, one 64px tile per tick
    sprite = np.concatenate([np.concatenate(rec[r]["video"], axis=1) for r in range(S.N_ROBOTS)], axis=0)
    Image.fromarray(sprite).save(OUT / "ego_sprite.png", optimize=True)
    print("sim done", sprite.shape)

    # ---- 2. features (standardised per channel over the episode) -------------------
    def std(a):
        a = np.asarray(a, dtype=np.float32)
        return (a - a.mean(0)) / (a.std(0) + 1e-6)

    feats_all = {}
    for r in range(S.N_ROBOTS):
        v = np.asarray(rec[r]["video"], dtype=np.float32) / 255.0                      # (N, 64, 64, 3)
        patches = v.reshape(N_TICKS, 16, 4, 16, 4, 3).transpose(0, 1, 3, 2, 4, 5).reshape(N_TICKS, 256, 48)
        patches = (patches - patches.mean()) / (patches.std() + 1e-6)
        feats_all[r] = {
            "video": torch.tensor(patches),
            "lidar": torch.tensor(std(rec[r]["lidar"]))[..., None],
            "proprio": torch.tensor(std(rec[r]["proprio"]))[..., None],
            "touch": torch.tensor(std(rec[r]["touch"]))[..., None],
            "move": torch.tensor(std(rec[r]["move"]))[:, None, :],
            "manip": torch.tensor(std(rec[r]["manip"]))[:, None, :],
        }

    # ---- 3. canvas DiT -------------------------------------------------------------
    layout = build_layout(WIN)
    intra, inter = build_topologies()
    model = CanvasDiT(layout, intra, inter).eval()
    # LeCun-normal init (sigma = 1/sqrt(fan_in)) for every linear layer. PyTorch's default
    # uniform init is ~3x smaller and leaves random-weight attention almost exactly uniform.
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=mod.in_features ** -0.5)
            nn.init.zeros_(mod.bias)
    last = model.blocks[-1]
    captured = {}
    last.intra.register_forward_pre_hook(lambda m, a: captured.__setitem__("intra", a[0].detach()))
    last.inter.register_forward_pre_hook(lambda m, a: captured.__setitem__("inter", a[0].detach()))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"canvas DiT: {n_params:,} params, {len(intra.connections)} intra + {len(inter.connections)} inter edges, "
          f"{layout.num_positions} positions per window")

    out_mask = torch.zeros(layout.num_positions, dtype=torch.bool)
    for r in range(S.N_ROBOTS):
        for kind in OUTPUTS:
            out_mask[layout.region_indices_at_t(rname(r, kind), WIN - 1)] = True
    gen = torch.Generator().manual_seed(SEED + 1)

    recorded_edges = []
    for r in range(S.N_ROBOTS):
        for c in inter.connections:
            if c.src in (rname(r, "move"), rname(r, "manip"), rname(r, "think")):
                recorded_edges.append(("inter", c))
        recorded_edges.append(("intra", next(c for c in intra.connections if c.src == c.dst == rname(r, "think"))))

    frames = []
    with torch.no_grad():
        for t in range(N_TICKS):
            ticks = [max(0, t - WIN + 1 + i) for i in range(WIN)]
            feats = {r: {k: v[ticks] for k, v in feats_all[r].items()} for r in range(S.N_ROBOTS)}
            x0 = model.embed(feats)
            noise = torch.randn(x0.shape, generator=gen)
            x = torch.where(out_mask[None, :, None], math.sqrt(ALPHA_BAR) * x0 + math.sqrt(1 - ALPHA_BAR) * noise, x0)
            model(x, TAU)

            frame = {}
            # thinking queries: report the token with the sharpest match on each robot
            for r in range(S.N_ROBOTS):
                for qkind in ("move", "manip", "think"):
                    src = rname(r, qkind)
                    edges = [(w, c) for w, c in recorded_edges if c.src == src]
                    results = []
                    for which, c in edges:
                        disp = last.intra if which == "intra" else last.inter
                        res = edge_attention(disp, captured[which], c, layout, None)
                        if res is not None:
                            results.append((c, res))
                    if not results:
                        continue
                    # pick the query token whose best key is hottest (always 0 for 1x1 readouts)
                    peak = torch.stack([a.max(-1).values for _, (_, _, a) in results]).max(0).values
                    qi = int(peak.argmax())
                    qflat = results[0][1][0][qi]
                    _, qr, qc = pos_to_cell(layout, qflat)
                    entry = {"q": [qr, qc], "edges": []}
                    for c, (s_idx, d_idx, a) in results:
                        w = a[qi]
                        top = torch.topk(w, min(TOPK, len(d_idx)))
                        keys = []
                        for val, j in zip(top.values.tolist(), top.indices.tolist()):
                            kt, kr, kc = pos_to_cell(layout, d_idx[j])
                            keys.append([kt - (WIN - 1), kr, kc, round(val, 4)])
                        ent = float(-(w * (w + 1e-12).log()).sum() / math.log(len(d_idx))) if len(d_idx) > 1 else 0.0
                        entry["edges"].append({"dst": c.dst, "n": len(d_idx), "ent": round(ent, 3), "keys": keys})
                    frame[src] = entry
            frames.append(frame)
            if t % 20 == 0:
                print("tick", t)

    regions = []
    for r in range(S.N_ROBOTS):
        for kind, (r0, r1, c0, c1) in LOCAL.items():
            regions.append({"name": rname(r, kind), "robot": ROBOTS[r], "kind": kind,
                            "r0": r0, "r1": r1, "c0": 16 * r + c0, "c1": 16 * r + c1})
    meta = {
        "H": H, "W": W, "ticks": N_TICKS, "win": WIN, "ctrl_dt": S.CTRL_DT, "robots": ROBOTS,
        "model": {"params": n_params, "d_model": D, "heads": HEADS, "depth": DEPTH, "alpha_bar": ALPHA_BAR,
                  "tau": TAU, "seed": SEED, "recorded_block": DEPTH - 1, "trained": False, "init": "lecun_normal"},
        "topology": {"intra": [[c.src, c.dst, c.t_src, c.t_dst] for c in intra.connections],
                     "inter": [[c.src, c.dst, c.t_src, c.t_dst] for c in inter.connections]},
        "regions": regions, "world": world, "frames": frames,
    }
    (OUT / "attention.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("wrote", OUT / "attention.json", os.path.getsize(OUT / "attention.json") // 1024, "KB")


if __name__ == "__main__":
    main()
