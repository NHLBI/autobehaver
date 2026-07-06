# Third-party notices

This file summarizes third-party tools and resources relevant to the Autobehaver
repository. Third-party software and resources remain under their own licenses.

## DeepLabCut

Autobehaver can run DeepLabCut for keypoint extraction and may point users to an
optional downloadable DeepLabCut project/model folder for testing the DLC stage.
DeepLabCut is primarily licensed under the GNU Lesser General Public License v3.0
(`LGPL-3.0`). Users should cite DeepLabCut when using or rerunning a DeepLabCut
workflow.

The Autobehaver repository does not claim ownership of DeepLabCut itself. Do not
copy DeepLabCut logos, artwork, or other protected DeepLabCut project materials
unless their license or maintainers permit that use.

## Python dependencies

Autobehaver depends on common scientific Python packages such as NumPy, pandas,
scikit-learn, PyTorch, OpenCV, h5py, PyYAML, XGBoost, SHAP, matplotlib, and
related packages. Each dependency remains under its own license.

For a reproducible environment, see:

```text
environment/environment.yml
environment/environment-dlc-tf.yml
environment/requirements-core.txt
environment/requirements-dlc-tf.txt
```

## Test and demo data

Smoke-test movies, keypoint files, inference CSVs, normalization parameters, and
small model checkpoints may be distributed separately through the project data
archive. Smoke-test assets are provided only to exercise the example pipeline.
They are not full-scale training data or full-scale analysis outputs.
