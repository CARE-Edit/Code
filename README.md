# CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing (CVPR 2026)

### 🔥 Please star CARE-Edit ⭐ and share it. Thanks! 🔥

>[Yucheng Wang*](https://wangandyyucheng.github.io/), [Zedong Wang*](https://jacky1128.github.io/), [Yuetong Wu](https://scholar.google.com/citations?user=Fjqr0fEAAAAJ&hl=zh-CN), [Yue Ma](https://mayuelala.github.io/), [Dan Xu†](https://www.danxurgb.net/)

>The Hong Kong University of Science and Technology (HKUST)

[![arXiv](https://img.shields.io/badge/arXiv-2603.08589-b31b1b.svg)](https://arxiv.org/abs/2603.08589) [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://wangandyyucheng.github.io/CARE-Edit/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yellow)](https://huggingface.co/papers/2603.08589) [![CVPR 2026](https://img.shields.io/badge/CVPR-2026-green)](#)

---

## 🚩 Updates

- ☑ Our paper is now available on [arXiv](https://arxiv.org/abs/2603.08589).
- ☑ **CARE-Edit** is accepted by **CVPR 2026**. Codes will be released soon.

## 💡 Motivation

Existing unified diffusion editors suffer from task interference and cannot dynamically handle conflicting conditions, leading to color bleeding, identity drift, and unpredictable behavior. We propose **CARE-Edit** - a unified editor which routes diffusion tokens to **four specialized experts** via a lightweight **condition-aware router**.

<p align="center">
  <img src="assets/abstract.png" alt="Motivation" width="100%">
</p>

## 🔧 Framework

<p align="center">
  <img src="assets/out.png" alt="CARE-Edit Framework" width="100%">
</p>

**CARE-Edit** introduces condition-aware specialized experts within the frozen DiT backbone. Given multimodal conditions, inputs are tokenized and projected to **heterogeneous expert branches**. The router assigns confidence scores and selects the Top-K experts to process each token. Expert outputs are normalized, modulated, and fused through the **Latent Mixture** module, yielding denoised representations refined by **Mask Repaint** module.

## 🎨 Results

### Contextual Image Editing

<p align="center">
  <img src="assets/comp1.png" alt="Comparison 1" width="100%">
</p>

### Qualitative Comparisons

<p align="center">
  <img src="assets/comp2.png" alt="Comparison 2" width="100%">
</p>

## 🛠️ Getting Started

> **Code coming soon!** Stay tuned for the full release.

<!--
### Installation

```bash
git clone https://github.com/xxx/CARE-Edit.git
cd CARE-Edit
pip install -r requirements.txt
```

### Usage

```bash
python run.py --config configs/default.yaml
```
-->

## 📜 BibTeX

If CARE-Edit is helpful for your research, please cite:

```bibtex
@inproceedings{wang2026careedit,
  title={CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing},
  author={Yucheng Wang and Zedong Wang and Yuetong Wu and Yue Ma and Dan Xu},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 📧 Contact

If you have any questions, please email **ywangls@connect.ust.hk**.

## 📜 Sincere Acknowledgement

Appreciate the following works for their great contributions:

- [**UNO**](https://bytedance.github.io/UNO/): Serves as the inspiration for our project.
- [**OmniControl**](https://github.com/): Foundational conditioning approaches that motivate our routing design.
