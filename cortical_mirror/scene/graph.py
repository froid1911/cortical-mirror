"""The world explained as a scene graph: a background, objects in depth order, an eye.

Each frame, every object node predicts where it will be from its velocity and
finds itself by local search around that prediction (inference, like motion
energy: the template is compared with the input only in a small window). The
graph then renders itself back to front, and every learned quantity changes by
a local rule:

* a node's appearance and mask, by a delta rule weighted by how well the node,
  rather than the background, explains each of its pixels (the online form of
  layered-sprite EM, Jojic & Frey 2001);
* the background, by a delta rule wherever no node claims a pixel, fast where
  the input is still;
* surfaces in front of objects (occluders), wherever a node is expected but the
  remembered background explains the pixel instead;
* depth order, by a vote wherever two nodes overlap.

That an object exists (an object file with position, velocity and depth) is
innate. What it looks like, which pixels belong to it, and what stands in front
of it are learned. New nodes form only from residual that moves (common fate).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SPRITE = 48
SEARCH = 8
SLOTS = 8
# What an unknown background costs to explain a pixel (a squared luminance error):
# where the background has not been seen still, it explains nothing well.
UNEXPLAINED = 0.004
# Frames back that "still" is judged against: slow, smooth objects barely change
# from one frame to the next.
STILL_LAG = 8
# The most one pixel can count against a node's placement.
AGAINST = 0.004
# Visibility above which a view of a node counts as complete.
COMPLETE = 0.7
# What a pixel of deviation from the predicted position costs, per unit of mass.
CONTINUITY = 0.0001
# Frames a forgotten node is remembered for recognition; one that has not come
# back by then is taken to be gone for good.
MEMORY = 150
# A newborn forgets what it cannot see within a few frames.
NEWBORN_PERSISTENCE = 1.0


@dataclass
class Node:
    id: int
    template: np.ndarray
    mask: np.ndarray
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    depth: float = 0.0
    belief: float = 1.0
    state: str = "seen"
    age: int = 0
    unseen: int = 0
    # Where it was born: a node that never moves away from there is not an object.
    origin: tuple = (0.0, 0.0)
    # Recent positions, newest last, and frames in a row it has been seen still.
    trail: list | None = None
    still: int = 0
    # Frames it has spent in the fovea.
    looked: int = 0
    # How much of it was seen in the last frame.
    visibility: float = 1.0
    # Its brightness, for the picture, beside the code it is matched in.
    shade: np.ndarray | None = None

    def displacement(self, frames: int = 5) -> tuple[float, float] | None:
        if not self.trail or len(self.trail) <= frames:
            return None
        (x0, y0), (x1, y1) = self.trail[-frames - 1], self.trail[-1]
        return x1 - x0, y1 - y0

    @property
    def mass(self) -> float:
        return float(self.mask.sum())


def _place(width: int, height: int, x: float, y: float):
    """World and sprite slices of a sprite centred at (x, y), clipped to the world."""

    left, top = int(round(x)) - SPRITE // 2, int(round(y)) - SPRITE // 2
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + SPRITE, width), min(top + SPRITE, height)
    if x0 >= x1 or y0 >= y1:
        return None
    world = (slice(y0, y1), slice(x0, x1))
    sprite = (slice(y0 - top, y1 - top), slice(x0 - left, x1 - left))
    return world, sprite


class SceneGraph:
    def __init__(
        self,
        width: int = 256,
        height: int = 192,
        slots: int = SLOTS,
        persistence: float = 30.0,
        learn: bool = True,
        learn_persistence: bool = False,
    ):
        self.width, self.height, self.slots = width, height, slots
        # Frames an unseen node stays believed (the time constant of its belief).
        # Unlike everything else in the graph, it carries over from scene to scene:
        # it is what development changes.
        self.persistence = persistence
        self.learn = learn
        self.learn_persistence = learn_persistence
        self.reset()

    def save(self, path) -> None:
        import torch

        torch.save(
            {
                "size": (self.width, self.height),
                "slots": self.slots,
                "persistence": float(self.persistence),
                "learn_persistence": self.learn_persistence,
            },
            path,
        )

    @classmethod
    def load(cls, path, learn: bool = True) -> SceneGraph:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            *state["size"],
            slots=state["slots"],
            persistence=state["persistence"],
            learn=learn,
            learn_persistence=state["learn_persistence"],
        )

    def _adjust_persistence(self, target: float, rate: float) -> None:
        """Move the persistence toward a target, in log space (a global, three-factor
        signal: the outcome of a belief, applied to the one shared time constant)."""

        if self.learn_persistence:
            log = math.log(self.persistence)
            self.persistence = math.exp(log + rate * (math.log(max(target, 1.0)) - log))

    def reset(self) -> None:
        self.nodes: list[Node] = []
        self.background: np.ndarray | None = None
        self.background_shade: np.ndarray | None = None
        # How sure the graph is of each background pixel: it grows where the input stays still.
        self.confidence = np.zeros((self.height, self.width), np.float32)
        self.front = np.zeros((self.height, self.width), np.float32)
        self.previous: np.ndarray | None = None
        self._recent: list[np.ndarray] = []
        self.frame_count = 0
        self._next_id = 0
        self.events: list[dict] = []
        # Recently forgotten nodes, with the frame they were forgotten: a thing that
        # reappears is recognised, not born again.
        self.memory: list[tuple[Node, int]] = []
        # Frames each pair of nodes has touched while moving together.
        self._together: dict[tuple[int, int], int] = {}

    # Rendering -----------------------------------------------------------------

    def _alpha(self, node: Node) -> np.ndarray:
        out = np.zeros((self.height, self.width), np.float32)
        placed = _place(self.width, self.height, node.x, node.y)
        if placed:
            out[placed[0]] = node.mask[placed[1]]
        return out

    def render(self, nodes: list[Node] | None = None, shade: bool = False) -> np.ndarray:
        """The graph's picture of the world, ``[H, W]`` float32 in [0, 1]: in the code
        it matches in, or with ``shade`` in brightness."""

        if self.background is None:
            return np.zeros((self.height, self.width), np.float32)
        background = self.background_shade if shade else self.background
        image = background.copy()
        for node in sorted(self.nodes if nodes is None else nodes, key=lambda n: n.depth):
            placed = _place(self.width, self.height, node.x, node.y)
            if not placed:
                continue
            world, sprite = placed
            alpha = node.mask[sprite]
            surface = node.shade if shade else node.template
            image[world] = alpha * surface[sprite] + (1 - alpha) * image[world]
        # Surfaces known to stand in front of objects hide them.
        return self.front * background + (1 - self.front) * image

    def edit(
        self,
        move: dict | None = None,
        delete: set | None = None,
        depth: dict | None = None,
        duplicate: dict | None = None,
        shade: bool = False,
    ) -> np.ndarray:
        """Render the world as it would be if changed; the graph itself is untouched."""

        nodes = [copy.deepcopy(n) for n in self.nodes if n.id not in (delete or set())]
        for node in nodes:
            dx, dy = (move or {}).get(node.id, (0, 0))
            node.x, node.y = node.x + dx, node.y + dy
            if depth and node.id in depth:
                node.depth = depth[node.id]
        for source, (dx, dy) in (duplicate or {}).items():
            if source in (delete or set()):
                continue
            original = next(n for n in self.nodes if n.id == source)
            twin = copy.deepcopy(original)
            # A duplicate goes where its original was moved to, plus its own offset.
            mx, my = (move or {}).get(source, (0, 0))
            twin.x, twin.y = twin.x + mx + dx, twin.y + my + dy
            twin.depth = original.depth + 0.5
            nodes.append(twin)
        return self.render(nodes, shade=shade)

    def expect(self, shade: bool = False) -> np.ndarray:
        """The graph's picture of the next frame: every node moved on by its velocity."""

        return self.edit(move={n.id: (n.vx, n.vy) for n in self.nodes}, shade=shade)

    # Inference -----------------------------------------------------------------

    def _cover(self, nodes: list[Node]) -> np.ndarray:
        cover = self.front.copy()
        for node in nodes:
            cover = 1 - (1 - cover) * (1 - self._alpha(node))
        return cover

    def _search(self, node: Node, image: np.ndarray, cover: np.ndarray, radius: int = SEARCH):
        """Best placement near the prediction: where the node explains its pixels
        better than the background does. Returns (x, y, gain, visibility)."""

        pad = SPRITE // 2 + radius
        cx, cy = int(round(node.x)), int(round(node.y))

        def window(a, fill):
            padded = np.pad(a, pad, constant_values=fill)
            y0, x0 = cy + pad - SPRITE // 2 - radius, cx + pad - SPRITE // 2 - radius
            if not (0 <= y0 and 0 <= x0):
                return None
            out = padded[y0 : y0 + SPRITE + 2 * radius, x0 : x0 + SPRITE + 2 * radius]
            if out.shape != (SPRITE + 2 * radius,) * 2:
                return None
            return sliding_window_view(out, (SPRITE, SPRITE))

        views = [window(image, 0.0), window(self._background_cost(image), 0.0), window(cover, 1.0)]
        if any(v is None for v in views):
            return node.x, node.y, 0.0, 0.0
        seen, behind, covered = views
        weight = node.mask * (1 - covered)
        # Evidence against a placement is capped: a pixel the node cannot explain may
        # be something known standing in front of it.
        evidence = behind - (seen - node.template) ** 2
        # Where the known background explains the pixel, it may stand in front of the
        # node: no evidence against it there at all.
        evidence = np.where(
            behind < 0.002, np.maximum(evidence, 0.0), np.maximum(evidence, -AGAINST)
        )
        gain = (weight * evidence).sum((-1, -2))
        mass = max(node.mass, 1e-6)
        # Continuity: things keep moving as they did; each pixel off the prediction
        # costs a little, so a partial view does not drag a node off its path.
        offsets = np.arange(-radius, radius + 1)
        # Only once its motion is established.
        established = min(max((node.age - 15) / 15, 0.0), 1.0)
        # The cost saturates: it resists small drifts, not a clear jump (a bounce).
        deviation = np.minimum(offsets[:, None] ** 2 + offsets[None, :] ** 2, 4)
        gain = gain - established * CONTINUITY * mass * deviation
        best = np.unravel_index(int(np.argmax(gain)), gain.shape)
        # How much of the node is seen: not hidden by anything known, and explained by
        # its template where it is placed (an unknown occluder hides the rest).
        explained = (seen[best] - node.template) ** 2 < 0.003
        visibility = float((weight[best] * explained).sum() / mass)
        return (
            node.x + best[1] - radius,
            node.y + best[0] - radius,
            float(gain[best] / mass),
            visibility,
        )

    def _background_cost(self, image: np.ndarray) -> np.ndarray:
        """How badly the background explains each pixel of the input."""

        known = self.confidence * (image - self.background) ** 2
        return known + (1 - self.confidence) * UNEXPLAINED

    def observe(self, image: np.ndarray, shade: np.ndarray | None = None) -> dict:
        """One frame, ``[H, W]`` in [0, 1] (or uint8 BGR): the code objects are matched
        in, and its brightness for the picture (the same image if not given)."""

        image = _as_gray(image, self.width, self.height)
        self.shade = image if shade is None else _as_gray(shade, self.width, self.height)
        self.frame_count += 1
        self.events = []
        if self.background is None:
            self.background = image.copy()
            self.background_shade = self.shade.copy()
            self.previous = image
            self._recent = [image]
            return {"nodes": 0}
        # Change over the last few frames: a slowly moving object is not still,
        # even where it changes little from one frame to the next.
        motion = np.abs(image - self._recent[0])
        for node in self.nodes:
            node.x, node.y = node.x + node.vx, node.y + node.vy
            node.age += 1
        # Nearest first: a node's visibility depends only on what is in front of it.
        ordered = sorted(self.nodes, key=lambda n: -n.depth)
        for index, node in enumerate(ordered):
            cover = self._cover(ordered[:index])
            predicted = (node.x, node.y)
            x, y, gain, visibility = self._search(node, image, cover)
            node.visibility = visibility
            if visibility > 0.25 and gain > 0.002:
                vx, vy = x - predicted[0] + node.vx, y - predicted[1] + node.vy
                # A partly hidden node's measured motion is trusted less.
                rate = 0.5 * min(visibility, 1.0) ** 2
                node.vx += rate * (vx - node.vx)
                node.vy += rate * (vy - node.vy)
                # Mostly hidden, it is where it was expected to be, more than where
                # its few visible pixels fit best.
                trust = min(max((visibility - 0.25) / 0.65, 0.0), 1.0)
                node.x = predicted[0] + trust * (x - predicted[0])
                node.y = predicted[1] + trust * (y - predicted[1])
                node.state, node.unseen, node.belief = "seen", 0, 1.0
            else:
                self._unseen(node, visibility)
        if self.learn:
            self._learn(image, motion)
        self._spawn(image, motion)
        self._retire()
        for node in self.nodes:
            if node.age in (10, 20, 30):
                self._identify(node)
        for node in self.nodes:
            # Recentring moves the frame, not the object: the trail follows the object.
            shift = self._recentre(node) if node.visibility >= COMPLETE else (0, 0)
            node.trail = [(x + shift[0], y + shift[1]) for x, y in node.trail or []]
            node.trail = [*node.trail, (node.x, node.y)][-12:]
            if node.state == "seen" and node.visibility >= COMPLETE and len(node.trail) >= 6:
                # Velocity as the slope of the recent path: single-frame steps are whole
                # pixels, and averaging them is slow and biased.
                path = np.array(node.trail[-10:])
                steps = np.arange(len(path))
                node.vx = float(np.polyfit(steps, path[:, 0], 1)[0])
                node.vy = float(np.polyfit(steps, path[:, 1], 1)[0])
            moved = node.displacement()
            moving = moved is None or math.hypot(*moved) > 1.5
            node.still = 0 if moving or node.state != "seen" else node.still + 1
        self.previous = image
        self._recent = (self._recent + [image])[-STILL_LAG:]
        return {"nodes": len(self.nodes)}

    def _unseen(self, node: Node, visibility: float) -> None:
        inside = 0 <= node.x < self.width and 0 <= node.y < self.height
        node.state = "occluded" if inside else "out of view"
        node.unseen += 1
        node.belief *= math.exp(-1.0 / max(self.persistence, 1e-6))

    # Learning ------------------------------------------------------------------

    def _learn(self, image: np.ndarray, motion: np.ndarray) -> None:
        ordered = sorted(self.nodes, key=lambda n: -n.depth)
        claimed = np.zeros_like(image)
        for index, node in enumerate(ordered):
            placed = _place(self.width, self.height, node.x, node.y)
            if not placed:
                continue
            world, sprite = placed
            nearer = self._cover(ordered[:index])[world]
            seen, behind = image[world], self.background[world]
            template, mask = node.template[sprite], node.mask[sprite]
            by_node = (seen - template) ** 2
            by_background = self._background_cost(image)[world]
            remembered = self.confidence[world] > 0.5
            if node.state == "seen":
                free = 1 - nearer
                # Common fate, pixel by pixel. For: the template predicted the pixel
                # better than "nothing changed since the last frame" did (it moved with
                # the node). Against: the template failed, or the pixel stayed still and
                # the known background explains it.
                unchanged = (seen - self.previous[world]) ** 2
                moved = motion[world] > 0.02
                background_here = remembered & ~moved & ((seen - behind) ** 2 < 0.002)
                belongs = (by_node < 0.002) & (by_node + 0.0002 < unchanged)
                foreign = (by_node > 0.01) | background_here
                evidence = free * (belongs | foreign)
                # Shape is learned only from complete views: in a partial one, what is
                # missing may be hidden, not absent.
                whole = float(node.visibility >= COMPLETE)
                mask += 0.2 * whole * evidence * (belongs - mask)
                # Cohesion: where the evidence is silent, a pixel goes with its neighbours.
                mask += 0.1 * whole * free * (cv2.GaussianBlur(mask, (0, 0), 2.0) - mask)
                # The whole frame learns, slowly where the mask is low: only pixels
                # that keep matching while the node moves (its own) grow into the mask.
                # Inside the mask the template only refines what it already roughly
                # matches, so a moment out of step does not overwrite the object with
                # whatever lies there. Never where the known background is seen.
                rate = np.where(mask < 0.3, 0.1, 0.3 * (by_node < 0.01))
                step = free * ~background_here * rate
                template += step * (seen - template)
                surface = node.shade[sprite]
                surface += step * (self.shade[world] - surface)
            # A node still partly in view, whose expected pixels the background explains
            # instead: something remembered stands in front of it. Not once it is out
            # of sight: a hidden thing that fails to reappear is not explained away.
            in_view = node.state == "seen"
            hidden = (
                in_view
                & remembered
                & (mask > 0.7)
                & ((seen - behind) ** 2 < 0.0025)
                & (by_node > 0.02)
            )
            shown = (mask > 0.7) & (by_node < by_background) & (nearer < 0.5)
            front = self.front[world]
            front += 0.5 * hidden * (1 - front) - 0.2 * shown * front
            claimed[world] = 1 - (1 - claimed[world]) * (1 - mask * (1 - nearer))
        self._vote_depth(image)
        # What stays still where no node claims it is background.
        still = (motion < 0.02).astype(np.float32)
        free = 1 - claimed
        self.confidence += (
            0.1 * free * (still * (1 - self.confidence) - (1 - still) * self.confidence)
        )
        rate = free * (0.01 + 0.25 * still)
        self.background += rate * (image - self.background)
        self.background_shade += rate * (self.shade - self.background_shade)

    def _vote_depth(self, image: np.ndarray) -> None:
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1 :]:
                both = (self._alpha(a) > 0.5) & (self._alpha(b) > 0.5)
                if both.sum() < 20:
                    continue
                ea = self._pixel_error(a, image)[both]
                eb = self._pixel_error(b, image)[both]
                vote = float(np.mean(ea < eb) - np.mean(eb < ea))
                # A clear vote puts the node that explains the overlap in front at once.
                if abs(vote) > 0.2:
                    near, far = (a, b) if vote > 0 else (b, a)
                    if near.depth <= far.depth:
                        # Just above the far one, below whatever was already above it,
                        # so the order of every other pair is kept.
                        above = [
                            n.depth for n in self.nodes if n is not near and n.depth > far.depth
                        ]
                        near.depth = (far.depth + min(above)) / 2 if above else far.depth + 1.0

    def _pixel_error(self, node: Node, image: np.ndarray) -> np.ndarray:
        out = np.full_like(image, np.inf)
        placed = _place(self.width, self.height, node.x, node.y)
        if placed:
            world, sprite = placed
            out[world] = (image[world] - node.template[sprite]) ** 2
        return out

    # Birth and death -------------------------------------------------------------

    def _spawn(self, image: np.ndarray, motion: np.ndarray) -> None:
        # Smoothed, so that a faint object adds up over its area.
        residual = cv2.GaussianBlur(np.abs(image - self.render()), (0, 0), 2.0)
        claimed = self._cover(self.nodes)
        # What lies right next to a visible node is probably its own edge, and more
        # so while a newborn node's mask is still growing.
        for node in self.nodes:
            if node.state != "seen":
                continue
            reach = np.ones((21, 21) if node.age < 10 else (13, 13), np.uint8)
            near = cv2.dilate((self._alpha(node) > 0.3).astype(np.uint8), reach)
            claimed = np.maximum(claimed, near.astype(np.float32))
        moving = cv2.dilate((motion > 0.04).astype(np.uint8), np.ones((7, 7), np.uint8))
        candidate = (residual > 0.05) & (moving > 0) & (claimed < 0.3) & (self.front < 0.5)
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        count, labels, stats, centres = cv2.connectedComponentsWithStats(candidate)
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
        for label in order:
            if len(self.nodes) >= self.slots or stats[label, cv2.CC_STAT_AREA] < 80:
                break
            x, y = (float(v) for v in centres[label])
            if self._recognise(image, x, y):
                continue
            template = np.zeros((SPRITE, SPRITE), np.float32)
            surface = np.zeros((SPRITE, SPRITE), np.float32)
            mask = np.zeros((SPRITE, SPRITE), np.float32)
            placed = _place(self.width, self.height, x, y)
            world, sprite = placed
            template[sprite] = image[world]
            surface[sprite] = self.shade[world]
            mask[sprite] = 0.6 * (labels[world] == label)
            # Something appearing over a known node came out from behind it.
            overlapping = [
                n.depth for n in self.nodes if self._alpha(n)[labels == label].mean() > 0.2
            ]
            if overlapping:
                depth = min(overlapping) - 1.0
            else:
                depth = max((n.depth for n in self.nodes), default=0.0) + 1.0
            self.nodes.append(
                Node(self._next_id, template, mask, x, y, depth=depth, origin=(x, y), shade=surface)
            )
            self.events.append({"born": self._next_id})
            self._next_id += 1

    def _recognise(self, image: np.ndarray, x: float, y: float) -> bool:
        """Something new near (x, y) may be a hidden node coming back, away from where
        it was expected, or a recently forgotten one: revive it instead of a birth."""

        for node in self.nodes:
            if node.state == "seen":
                continue
            cover = self._cover([n for n in self.nodes if n is not node])
            probe = copy.deepcopy(node)
            probe.x, probe.y = x, y
            found_x, found_y, gain, visibility = self._search(probe, image, cover, radius=24)
            if visibility > 0.25 and gain > 0.004:
                error = math.hypot(found_x - node.x, found_y - node.y)
                self.events.append({"returned": node.id, "after": node.unseen, "error": error})
                node.x, node.y, node.trail = found_x, found_y, None
                node.state, node.unseen, node.belief = "seen", 0, 1.0
                return True
        cover = self._cover(self.nodes)
        for index, (node, forgotten) in enumerate(self.memory):
            probe = copy.deepcopy(node)
            probe.x, probe.y = x, y
            found_x, found_y, gain, visibility = self._search(probe, image, cover, radius=16)
            if visibility > 0.25 and gain > 0.004:
                node.x, node.y, node.vx, node.vy = found_x, found_y, 0.0, 0.0
                node.state, node.belief, node.trail = "seen", 1.0, None
                self.nodes.append(node)
                del self.memory[index]
                self._recognised(node, forgotten)
                return True
        return False

    def _recognised(self, node: Node, forgotten: int) -> None:
        after = self.frame_count - forgotten
        self.events.append({"recognised": node.id, "after": after})
        # It was forgotten too early: unseen for this long, it still existed.
        unseen = node.unseen + after
        node.unseen = 0
        if unseen > self.persistence:
            self._adjust_persistence(unseen, 0.2)

    def _identify(self, node: Node) -> None:
        """A node just formed may be a forgotten one seen again: same look, same shape.
        Then it is that one, with its identity and depth."""

        for index, (old, forgotten) in enumerate(self.memory):
            if _likeness(node, old) > 0.003:
                continue
            del self.memory[index]
            self.events.append({"identified": node.id, "as": old.id})
            node.id, node.depth, node.unseen = old.id, old.depth, old.unseen
            self._recognised(node, forgotten)
            return

    def _retire(self) -> None:
        keep = []
        for node in self.nodes:
            travelled = math.hypot(node.x - node.origin[0], node.y - node.origin[1])
            if node.still >= 12 and travelled < 15:
                # Never moved: not an object, just background seen anew.
                self._dissolve(node)
                continue
            if node.belief < 0.1 or (node.age > 5 and node.mass < 30):
                self.events.append({"forgotten": node.id})
                if node.age > 10 and node.mass > 100:
                    self.memory = [(node, self.frame_count), *self.memory][:16]
                continue
            keep.append(node)
        # Common fate: nodes that touch and move together for a while are one object.
        keep.sort(key=lambda n: (-n.age, n.id))
        survivors: list[Node] = []
        together = {}
        for node in keep:
            touching = cv2.dilate((self._alpha(node) > 0.3).astype(np.uint8), np.ones((5, 5)))
            partner = None
            for other in survivors:
                pair = (other.id, node.id)
                a, b = node.displacement(), other.displacement()
                if (
                    a is not None
                    and b is not None
                    and (touching & (self._alpha(other) > 0.3)).any()
                    and math.hypot(a[0] - b[0], a[1] - b[1]) < 2.5
                ):
                    together[pair] = self._together.get(pair, 0) + 1
                    if together[pair] >= 5:
                        partner = other
                        break
            if partner is None:
                survivors.append(node)
                continue
            self._absorb(partner, node)
            self.events.append({"merged": node.id, "into": partner.id})
        self._together = together
        kept = [(n, f) for n, f in self.memory if self.frame_count - f < MEMORY]
        for _ in range(len(self.memory) - len(kept)):
            # Forgotten and never seen again: believing in it longer would have been wasted.
            self._adjust_persistence(NEWBORN_PERSISTENCE, 0.25)
        self.memory = kept
        self.nodes = sorted(survivors, key=lambda n: n.id)

    def _dissolve(self, node: Node) -> None:
        placed = _place(self.width, self.height, node.x, node.y)
        if placed:
            world, sprite = placed
            alpha = node.mask[sprite]
            self.background[world] = (
                alpha * node.template[sprite] + (1 - alpha) * self.background[world]
            )
            self.background_shade[world] = (
                alpha * node.shade[sprite] + (1 - alpha) * self.background_shade[world]
            )
        self.events.append({"dissolved": node.id})

    def _absorb(self, keep: Node, gone: Node) -> None:
        dx, dy = int(round(gone.x - keep.x)), int(round(gone.y - keep.y))
        template, mask = _shift(gone.template, dx, dy), _shift(gone.mask, dx, dy)
        take = mask > keep.mask
        keep.template = np.where(take, template, keep.template)
        keep.shade = np.where(take, _shift(gone.shade, dx, dy), keep.shade)
        keep.mask = np.maximum(keep.mask, mask)
        self._recentre(keep)

    def _recentre(self, node: Node) -> tuple[int, int]:
        """Keep the object in the middle of its own frame: position means its centre.
        Returns how far the frame moved."""

        if node.mass < 1e-6:
            return 0, 0
        yy, xx = np.mgrid[:SPRITE, :SPRITE]
        dx = int(round(float((node.mask * xx).sum() / node.mass) - (SPRITE - 1) / 2))
        dy = int(round(float((node.mask * yy).sum() / node.mass) - (SPRITE - 1) / 2))
        if dx == 0 and dy == 0:
            return 0, 0
        node.template = _shift(node.template, -dx, -dy)
        node.shade = _shift(node.shade, -dx, -dy)
        node.mask = _shift(node.mask, -dx, -dy)
        node.x, node.y = node.x + dx, node.y + dy
        return dx, dy

    # The graph -----------------------------------------------------------------

    def to_json(self) -> dict:
        nodes = [
            {
                "id": n.id,
                "kind": "object",
                "x": round(n.x, 1),
                "y": round(n.y, 1),
                "vx": round(n.vx, 2),
                "vy": round(n.vy, 2),
                "depth": round(n.depth, 2),
                "state": n.state,
                "belief": round(n.belief, 3),
                "size": round(n.mass, 1),
            }
            for n in self.nodes
        ]
        edges = []
        alphas = {n.id: self._alpha(n) > 0.5 for n in self.nodes}
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1 :]:
                overlap = int((alphas[a.id] & alphas[b.id]).sum())
                if overlap:
                    near, far = (a, b) if a.depth > b.depth else (b, a)
                    edges.append({"relation": "occludes", "from": near.id, "to": far.id})
                distance = math.hypot(a.x - b.x, a.y - b.y)
                if distance < SPRITE:
                    edges.append({"relation": "near", "from": a.id, "to": b.id})
                speed = math.hypot(a.vx, a.vy) + math.hypot(b.vx, b.vy)
                if speed > 0.5 and math.hypot(a.vx - b.vx, a.vy - b.vy) < 0.2 * speed:
                    edges.append({"relation": "moves_with", "from": a.id, "to": b.id})
        return {
            "frame": self.frame_count,
            "nodes": nodes,
            "edges": edges,
            "self": {"kind": "eye", "gaze": None},
        }


def _likeness(a: Node, b: Node, reach: int = 8) -> float:
    """How unlike two nodes look at their best alignment: the mean squared template
    difference where both masks hold, if they overlap on most of the smaller one."""

    mine = a.mask > 0.5
    best = math.inf
    for dy in range(-reach, reach + 1, 2):
        for dx in range(-reach, reach + 1, 2):
            theirs = _shift(b.mask, dx, dy) > 0.5
            both = mine & theirs
            if both.sum() < 0.4 * min(mine.sum(), theirs.sum(), 1e9) or not both.any():
                continue
            difference = a.template[both] - _shift(b.template, dx, dy)[both]
            best = min(best, float(np.mean(difference**2)))
    return best


def _shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Content moved by (dx, dy), with zeros coming in."""

    out = np.zeros_like(a)
    h, w = a.shape
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = a[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def _as_gray(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.dtype == np.uint8:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
    if image.shape != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)
