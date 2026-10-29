# Research brief: objects that grow from learned features

The last large engineered piece of Cortical Mirror is the object level: slots with pixel templates and soft masks, found by template search and merged by rules. Following the project's direction (a virtual cortex, engineered parts replaced by learned ones), this brief collects how objects could emerge from learned features under local learning. Citations were checked by web search from abstracts or records; **[unverified]** marks the rest.

## The state of the art, in one line

No existing method makes whole objects (segments with identity, shape and depth order) emerge from learned features with only local rules. The working parts exist separately.

## The parts

- **Binding by synchrony and phase.**
  - Units of one object fire in phase. In LEGION (Wang & Terman 1995, IEEE TNN 6:283), local excitation and one global inhibitor do this. The affinities are hand-set, but could be learned by a Hebbian rule.
  - Capacity is about 4–7 objects per cycle.
  - Complex-valued autoencoders (Löwe et al. 2022, arXiv 2204.02075) and Rotating Features (2023, arXiv 2306.00600) learn the phase code, but by backprop. Only the representation transfers.
- **Border ownership (figure-ground) in V2.**
  - Many V2 cells signal which side of an edge the figure is on (Zhou, Friedman & von der Heydt 2000, J Neurosci 20:6594).
  - In Craft et al. 2007 (J Neurophysiol 97:4310), grouping cells pool border-ownership cells and feed back. That gives figure-ground and local depth order.
  - Plain Hebbian learning fails to learn the feedback. A modified rule succeeds: Grant, Tanner & Itti 2017, Neural Networks 88:32, the first learned border-ownership network.
- **Surface filling-in** (Grossberg's FACADE theory **[unverified citations]**). V1 codes edges, and surface appearance is filled in by diffusion bounded by the edges. This is the clearest explanation of why learned V1 codes found only outlines in v0.5.
- **Learned V2 codes.**
  - Sparse coding stacked on pooled V1 energy learns V2-like contour, corner and texture tuning: Lee, Ekanadham & Ng 2007 (NIPS); Hosoya & Hyvärinen 2015 (J Neurosci 35:10412).
  - It uses the same local rule as our V1. It does not give object identity on its own.
- **Common fate, learned (the key precedent).** Prodöhl, Würtz & von der Malsburg 2003 (Neural Computation 15:1865) show a Hebbian rule, run on real camera video, learning grouping links: edges of one moving object move together. Our MT cells, with self-motion removed by the v0.8 efference map, supply exactly this signal.
- **Slot-based object learning** (Slot Attention, IODINE, MONet, Tagger, Neural EM). All use backprop. Their inference loop is soft competition of slots for features, then updates of each slot from what it was assigned: EM, which is local. Neural EM with learned codes in place of a trained decoder is close to our graph.
- **Predictive coding with local learning** (Brucklacher et al. 2023, Front. Comput. Neurosci. 17:1207361) learns position-invariant object identity from temporal continuity. That is recognition; no local-learning model that *segments* was found. This is an opening.

## Recommended path, ranked

1. **Grouping affinity learned from common fate, read out by competitive EM.**
   - Features at each place: the V1 code plus MT's motion with self-motion removed.
   - A lateral affinity W learns by Hebbian co-activation, gated by matching motion.
   - Objects form by EM over K slots, with W as the similarity, instead of pixel templates.
   - Pre-registerable:
     - mean object IoU on held-out sprite worlds at least 0.9 × the engineered graph's;
     - on still frames, the learned W beats a pure similarity baseline by ≥ 0.05 ARI, i.e. grouping learned from motion transfers to static features.
2. **A V2 surface and texture code.**
   - Sparse coding with our local rule on pooled V1 energy plus MT, used as each slot's appearance in place of the hand-designed normalisation.
   - Pre-registerable:
     - identity retention at least as high as with the normalisation;
     - a within-object to between-object code distance ratio below 0.5;
     - interior filled in, not only edges.
3. **Border ownership as a learned depth cue.** Craft-style grouping cells, with feedback learned by the Grant et al. rule, trained on occlusion events in video. Pre-registerable: border-ownership side accuracy above 75% (chance 50%) and pairwise depth-order accuracy above 80%.

Order: 1 first, since it reuses MT and the efference map and removes template search directly; then 2; then 3, the most speculative.
