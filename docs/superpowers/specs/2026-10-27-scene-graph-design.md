# v0.4: a cortex that explains the world as a scene graph — design

## Intent

Joscha Bach's feedback on the project: "Der Output sollte ein Scenegraph sein. Es geht nicht nur darum, Bitmaps vorherzusagen, sondern die Welt zu erklären, so dass man sie manipulieren kann." (The output should be a scene graph. It is not only about predicting bitmaps, but about explaining the world so that one can manipulate it.)

v0.3's cortex predicts retinal codes. It has edge cells, but no things. v0.4 adds a level that explains the input as **objects in a world**: a scene graph whose nodes are objects (with appearance, shape, position, motion and depth), a remembered background, and the eye itself. Rendering the graph must reproduce what is seen; editing the graph must produce what the world *would* look like if it were changed. That is the test of an explanation, as opposed to a prediction.

The three directions agreed for v0.4 (user: "make 1-3") all become properties of this graph:

1. **Things, not edges:** nodes are discovered from the input; object permanence is a node that survives occlusion.
2. **Attention it learns itself:** the eye chooses which node to look at, by a learned value of looking; the hand-coded "look toward surprise" reflex is replaced.
3. **"Me" versus "the world":** the eye is a node. Its own movements explain the image shifts they cause, so the world's nodes stay put when the eye moves.

The user is at home in Berlin, not at the hackathon: no CPU or live-demo constraint. Training uses flank's GPUs 0, 1 and 3 for as long as it needs. The user asked to "see how far you can push it": the build proceeds without further questions, and every call below not made by the user is marked as Claude's.

## Goals

1. A scene level (`cortical_mirror/scene/`) whose state is an explicit, inspectable scene graph, served as JSON, and whose rendering is the mirror's picture.
2. Editing that graph (move, delete, reorder depth, duplicate) renders a counterfactual image.
3. Pre-registered protocols 8–12 (below), each with a measurement tool and a report, on untrained, newborn and adult, with failures reported.
4. Twin mirrors as before, now showing the explained world and the graph, with objects that can be dragged and deleted in the browser.

## Non-goals

- Beating any segmentation or video model. The comparison is untrained vs newborn vs adult vs simple baselines.
- Semantic labels ("dog", "ball"). Nodes are things, not words.
- 3D. Depth is an occlusion order (2.5D layers), as in early infancy.
- Fitting to neural data.

## Rules

v0.3's rules hold: no backpropagation or autograd, no weight transport, newborn learns only from its own stream, protocols written before measuring, failures reported.

**Rule 2 is extended (Claude's call, flagged to the user):** innate structure may include *object files* (Kahneman & Treisman 1992; Spelke's core knowledge): the architecture has a small number of slots, each holding one object's position, velocity, depth rank and appearance, and a renderer that composes them back to front over a background. That an object exists is innate; **what** objects look like, **which** pixels belong together, **how long** to believe in an unseen object, **where** to look, and **how** the eye's commands shift the view are all learned with local rules. Nothing about permanence, gaze value or the eye's motor mapping is built in beyond a newborn default of "none".

Inference may do local search (template matching in a window around a node's predicted position, like motion energy); that is inference, not learning, and is declared.

## Architecture (Claude's calls)

### World and scale

- The scene level works in luminance at world resolution (256×192), because a scene graph must render to a picture and be edited as one. Its error is also reported in retinal-code space, through the fixed v0.3 retina, so it is comparable with v0.3's cortex.
- v0.3's V1/V2 stay as they are and keep running for the twin cells grid. Building nodes from V1 features instead of luminance is a later step, not v0.4.

### The scene graph

- **Background node:** a world-coordinate canvas larger than the view, with a per-pixel confidence. Pixels are written where no object explains them; occluded background stays remembered.
- **Object nodes (K slots, K = 8):** appearance template and soft mask (both in the node's own frame, 48×48 max), position, velocity, depth rank, presence belief, visibility (seen / occluded / gone), uncertainty, and an id that persists.
- **Self node:** gaze position in world coordinates, last motor command, and the learned command→shift mapping.
- **Edges, derived each frame:** `occludes(a, b)` from depth rank and mask overlap, `moves_with(a, b)` from velocity similarity, `near(a, b)`.

### Inference per frame

1. Predict each node's position from its velocity; predict the view from the self node's command.
2. Render back to front: background, then nodes by depth, each through its mask.
3. For each visible node: local template search around the predicted position; update position and velocity.
4. Responsibilities: each pixel goes to the front-most node whose mask and appearance explain it, else the background.
5. Spawn a node where the residual is large, spatially coherent and moves coherently over a few frames (common fate). Retire a node after its presence belief falls to zero.

### Learning (all local)

- **Appearance and mask:** a responsibility-weighted delta rule per pixel of the node's template. This is the online form of layered-sprite EM (Jojic & Frey 2001), and each update uses only that pixel's input, prediction and responsibility.
- **Depth order:** a pairwise vote. Where two nodes overlap, the one whose appearance explains the overlap gets depth over the other.
- **Persistence (permanence emerges here):** per node, the belief time constant τ decides how long an unseen node is kept. Newborn τ ≈ 0, so it forgets anything it cannot see. τ learns by a three-factor rule:
  - a node that reappears where its extrapolation predicted strengthens τ;
  - a new node whose template matches a recently forgotten one also strengthens τ (it forgot too early);
  - a kept node that never reappears weakens τ.
  A shared τ for all nodes (Claude's call) keeps it a developmental property, not a per-object one.
- **Self (motor) mapping:** per command, the expected view shift is learned by a delta rule on the shift that best aligns the views before and after it. The newborn expects no shift, so it sees the world jump when it looks around.
- **Gaze value (learned attention):** each node's features are uncertainty, prediction error, time since fixated and motion predictability. They are weighted by w to give a value, and the eye looks at the highest (with inhibition of return kept as before). After each fixation, w changes by η · δ · feature, where δ is the error reduction the fixation produced minus what w predicted. That is a three-factor rule: a local product times a global signal like dopamine, and it learns which kind of looking teaches the model most (Oudeyer's learning progress). The newborn's w is zero, so it looks at random.
- **Fovea:** nodes inside the fovea update appearance at full resolution. Outside it they are updated only at 1/4 resolution (position and coarse appearance), so looking matters.

### Manipulation API

`graph.edit(...)` supports move, delete, set depth and duplicate, then `graph.render()` produces the image. Edits act on a copy and never change the learned state.

## Worlds

- **Sprite world (primary, synthetic, ground truth available for scoring only):**
  - a 1/f background larger than the view;
  - 1–4 textured objects with random smooth shapes, moving with constant velocity and bouncing;
  - static occluder bars;
  - optional saccading eye or camera pan.
  It is deterministic by seed, with training and held-out seeds kept apart.
- **Real video (UCF101, camera):** the same model, reported qualitatively plus reconstruction error. Real video has camera motion, deformation and lighting changes; v0.4 promises nothing there beyond an honest report.

## Phenomena and protocols (written before measuring, `docs/protocols/08-12`)

| # | Phenomenon | Test | Pass (to be fixed in each protocol before measuring) |
|---|---|---|---|
| 8 | Object discovery | Held-out sprite scenes. Match nodes to true objects (IoU of masks). | The adult's mean best-match IoU is well above untrained and above a motion-threshold baseline. |
| 9 | Object permanence | An object passes behind an occluder. (a) Does its node survive, and is its predicted reappearance point right? (b) Violation of expectation: error when it fails to reappear vs when it does. | The adult keeps the node and predicts reappearance within a few pixels. The violation error is greater than the expected error. The newborn does not (developmental contrast). |
| 10 | Explanation by manipulation | On held-out scenes, apply edits (move, delete) to the inferred graph and render. Compare with the ground-truth world rendered with the same edit. | The adult beats bitmap cut-and-paste, especially on delete, which needs the remembered background (amodal completion). |
| 11 | Learned attention | Re-run protocol 7 (panning vs unpredictable onsets) with the learned gaze value, plus model error after N fixations vs random and surprise-reflex gaze. | Dwell on onsets rises by at least 0.1 over the frozen control (v0.3: 0.0). The model learns faster than with random gaze. |
| 12 | Self vs world | The same retinal shift, either commanded (saccade) or uncommanded (the world pans). Measure how far world nodes move in world coordinates. | Commanded: world nodes stay put (small drift). Uncommanded: they move by most of the true shift. The newborn confuses the two. |

## Order of work

- **4a: sprite world and the scene graph core.** Discovery, rendering and editing; protocols 8 and 10.
- **4b: permanence.** Learned τ; protocol 9.
- **4c: the self node and learned attention.** Fovea, motor mapping and gaze value; protocols 11 and 12.
- **4d: real video and the demo.** UCF101 and camera; twin mirrors with the graph view, drag and delete; README and v0.4.0 release.

Each step is merged by its own PRs. Training runs on flank GPUs 0, 1 and 3 only.

## Constraints

- GPUs on flank: 0, 1, 3, never 2.
- No AI attribution anywhere: commits, PRs, releases, documents, comments.
- Small incremental commits on side branches, merged via PR.
- v0.2 and v0.3 code and releases stay intact; the scene level is a new package.

## Left to the builder

Exact sizes, rates and thresholds (thresholds fixed in the protocols before any measurement), the spawning heuristic's constants, the search window, and the visual details of the graph view.
