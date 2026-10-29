"""The learned scene behind the scene graph's interface, for the live page.

The live page (`scene/live.py`) speaks to a scene graph: it observes frames, lists
nodes, asks for their masks, edits and renders. This adapter answers with the
learned scene: objects found by learned motion (slots), sharpened by object files,
over a remembered background. It replaces the engineered template search; the
page does not change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..scene.world import SPRITE
from .learned_scene import LearnedScene, shifted
from .slots import MotionSlots


@dataclass
class _Node:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    mask: np.ndarray
    belief: float
    depth: float = 0.0
    state: str = "seen"

    @property
    def mass(self) -> float:
        return float(self.mask.sum())


class LearnedGraph:
    def __init__(
        self, cortex, field=None, width: int = 256, height: int = 192, files=True, silenced=False
    ):
        self.cortex, self.field, self.files, self.silenced = cortex, field, files, silenced
        self.width, self.height = width, height
        self.reset()

    def reset(self) -> None:
        self.slots = MotionSlots(field=self.field, device=self.cortex.device)
        self.scene = LearnedScene(
            self.cortex,
            self.slots,
            width=self.width,
            height=self.height,
            files=self.files,
            silenced=self.silenced,
        )
        self.nodes: list[_Node] = []
        self.frame_count = 0

    # Seeing ------------------------------------------------------------------

    def observe(self, image: np.ndarray) -> None:
        self.scene.observe(image)
        self.frame_count += 1
        self._refresh()

    def _refresh(self) -> None:
        """The nodes: one per object slot in view, with its mask and motion."""

        scale = self.width / 128  # MT's velocity is in px of the 128-wide view
        self.nodes = []
        for k, mask in sorted(self.scene.masks(self.scene.labels).items()):
            ys, xs = np.nonzero(mask)
            u = self.slots.u[k].tolist() if k < len(self.slots.u) else [0.0, 0.0]
            file = self.scene.files.get(k)
            self.nodes.append(
                _Node(
                    id=k,
                    x=float(xs.mean()),
                    y=float(ys.mean()),
                    vx=u[0] * scale,
                    vy=u[1] * scale,
                    mask=mask,
                    belief=1.0 if file is not None and file.confident else 0.5,
                )
            )

    def _alpha(self, node: _Node) -> np.ndarray:
        return node.mask.astype(np.float32)

    # Pictures ------------------------------------------------------------------

    def render(self, shade: bool = False) -> np.ndarray:
        """What the scene explains: the remembered background, with its objects."""

        frame = self.scene.frame
        if frame is None:
            return np.zeros((self.height, self.width), np.float32)
        out = np.where(self.scene.seen > 0, self.scene.memory, frame).astype(np.float32)
        for node in self.nodes:
            out[node.mask] = frame[node.mask]
        return out

    def edit(self, move=None, delete=None, depth=None, duplicate=None, shade=False):
        """The scene as it would look after the edits (``depth`` is not modelled)."""

        picture = self.render()
        if self.scene.labels is None:
            return picture
        ids = {n.id for n in self.nodes}
        move = {k: (int(round(v[0])), int(round(v[1]))) for k, v in (move or {}).items()}
        out = self.scene.edit(
            picture,
            self.scene.labels,
            delete={k for k in (delete or set()) if k in ids},
            move={k: v for k, v in move.items() if k in ids},
        )
        masks = {n.id: n.mask for n in self.nodes}
        for k, offset in (duplicate or {}).items():
            if k in masks:
                dx, dy = int(round(offset[0])), int(round(offset[1]))
                target = shifted(masks[k], dx, dy, False)
                out[target] = shifted(picture, dx, dy)[target]
        return out

    def expect(self) -> np.ndarray:
        """The next frame: the last one, with every object moved on by its velocity.

        Where an object could be, the remembered background shows; what surely belongs
        to it is moved by its velocity, to the sub-pixel. With no motion this is
        exactly the last frame."""

        frame = self.scene.frame
        if frame is None:
            return np.zeros((self.height, self.width), np.float32)
        out = frame.astype(np.float32).copy()
        coarse = self.scene._grid_masks(self.scene.labels) if self.scene.labels is not None else {}
        moving = [n for n in self.nodes if abs(n.vx) + abs(n.vy) > 1e-6]
        removed = np.zeros(out.shape, bool)
        for node in moving:
            removed |= node.mask | coarse.get(node.id, np.zeros_like(removed))
        known = removed & (self.scene.seen > 0)
        out[known] = self.scene.memory[known]
        for node in moving:
            warp = np.float32([[1, 0, node.vx], [0, 1, node.vy]])
            size = (self.width, self.height)
            alpha = cv2.warpAffine(node.mask.astype(np.float32), warp, size, flags=cv2.INTER_LINEAR)
            layer = cv2.warpAffine(
                np.where(node.mask, frame, 0).astype(np.float32), warp, size, flags=cv2.INTER_LINEAR
            )
            inside = alpha > 1e-3
            out[inside] = out[inside] * (1 - alpha[inside]) + layer[inside]
        return out

    # The page's scene ------------------------------------------------------------

    def to_json(self) -> dict:
        nodes = [
            {
                "id": n.id,
                "kind": "object",
                "x": round(n.x, 1),
                "y": round(n.y, 1),
                "vx": round(n.vx, 2),
                "vy": round(n.vy, 2),
                "depth": n.depth,
                "state": n.state,
                "belief": n.belief,
                "size": round(n.mass, 1),
            }
            for n in self.nodes
        ]
        edges = []
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1 :]:
                if math.hypot(a.x - b.x, a.y - b.y) < SPRITE:
                    edges.append({"relation": "near", "from": a.id, "to": b.id})
                speed = math.hypot(a.vx, a.vy) + math.hypot(b.vx, b.vy)
                if speed > 0.5 and math.hypot(a.vx - b.vx, a.vy - b.vy) < 0.2 * speed:
                    edges.append({"relation": "moves_with", "from": a.id, "to": b.id})
        return {
            "frame": self.frame_count,
            "nodes": nodes,
            "edges": edges,
            "groups": [],
            "self": {"kind": "eye", "gaze": None},
            "objects": "learned",
        }
