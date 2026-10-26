# Third-party notices

## DMVFN

`cortical_mirror/dmvfn_impl.py` is derived from **A Dynamic Multi-Scale Voxel Flow Network for Video Prediction**, CVPR 2023, by Xiaotao Hu, Zhewei Huang, Ailin Huang, Jun Xu, and Shuchang Zhou.

Source: https://github.com/hzwer/CVPR2023-DMVFN

Copyright 2023 Megvii Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The local adaptation removes training-only dependencies and hard-coded CUDA allocations, caches grids by actual device and dtype, and adds deterministic single-image inference. Checkpoints retain their original upstream terms.

## Sample clip

`sample.mp4` (a v0.2.0 release asset, played by `cortical-mirror demo` when no camera is found) and the animation `docs/images/live-demo.gif` contain **"FFSb - Corona LSFB"** by the **Fédération Francophone des Sourds de Belgique (FFSB)**.

Source: https://commons.wikimedia.org/wiki/File:FFSb_-_Corona_LSFB.webm

License: Creative Commons Attribution 3.0 Unported (CC BY 3.0), https://creativecommons.org/licenses/by/3.0/

Changes: audio removed, scaled to 640×360, and re-encoded as H.264. The animation additionally shows the clip mirrored and as predicted by this software, with an interface overlay. The FFSB does not endorse this project.
