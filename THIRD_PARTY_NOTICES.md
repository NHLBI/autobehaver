# Third-party notices

Third-party software and resources remain under their own licenses.

## DeepLabCut

Autobehaver uses DeepLabCut for keypoint extraction. DeepLabCut is primarily
licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). Cite
DeepLabCut when using or rerunning its workflow. Trained project resources and
externally derived materials retain their applicable licenses.

## Scientific Python dependencies

Autobehaver uses NumPy, pandas, SciPy, scikit-learn, PyTorch, OpenCV, h5py, PyYAML,
XGBoost, SHAP, matplotlib and related packages. Each retains its own license.
Pinned environment specifications are provided in:

```text
environment/environment.yml
environment/environment-dlc-tf.yml
environment/requirements-core.txt
environment/requirements-local-core.txt
environment/requirements-dlc-tf.txt
```

## Models and example data

Trained models and example data are supplied separately from Git. They exercise
or support the declared scientific workflow and retain their own provenance and
licensing terms. See [resource distribution](docs/MODEL_DISTRIBUTION.md) and
[ASSET_NOTICE.md](ASSET_NOTICE.md).
