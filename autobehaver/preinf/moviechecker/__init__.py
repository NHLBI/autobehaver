"""Movie checker (manual QC + exclusions).

This tool provides a lightweight OpenCV-based UI for quickly scanning full-frame
movies and marking chambers to exclude during cropping/downstream processing.

The intended workflow is:

1) Put all full-frame movies into a single folder.
2) Run the movie checker locally:
     python -m autobehaver.preinf.moviechecker --input_dir /path/to/1_full_movies
3) An ``exclusions.csv`` file is written inside that folder.
4) The cropper (autocropper) can later read that file to skip excluded chambers.
"""
