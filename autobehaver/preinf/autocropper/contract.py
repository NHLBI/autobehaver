"""Identify crop requests and validate movie shape and completion receipts."""
from pathlib import Path
import os
import re
import cv2
from autobehaver.completion import request_identity, receipt_matches, file_digest
from .thread_limits import open_video_capture


def crop_request(movie, chamber_id, args, *, base_request=None):
    """Build the crop identity from source content, settings, calibration, and code.

    Reuse base_request for chambers sharing the same source and settings.
    """
    if base_request is not None:
        return {**base_request, 'settings': {**base_request['settings'], 'chamber': str(chamber_id)}}
    names = ('chamber_columns','chamber_rows','down_factor','approx_radius','crop_crf',
             'frame_avg','crop_dim_2k','crop_dim','crop_dimension','param1','param2','min_radius','max_radius',
             'calibration_tolerance')
    values = {k:getattr(args,k,None) for k in names}
    calibration = getattr(args,'chamber_calibration_json',None) or getattr(args,'calibration_file',None) or getattr(args,'calibration_json',None)
    if calibration:
        values['calibration_sha256'] = file_digest(calibration)
    values['chamber'] = str(chamber_id)
    folder = Path(__file__).parent
    return request_identity('crop-v1', [movie], values,
                            [folder/'detection.py',folder/'grid_fit.py',folder/'cropping.py',folder/'contract.py'])


def video_shape(path):
    """Return frame count, width, and height after decoding beginning, middle, and end frames."""
    capture = open_video_capture(path)
    try:
        if not capture.isOpened():
            raise ValueError('video cannot be opened')
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if min(count,width,height) <= 0:
            raise ValueError('video contains no frames')
        for index in sorted({0,count//2,count-1}):
            capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            if not capture.read()[0]:
                raise ValueError(f'cannot decode video frame {index}')
        return count,width,height
    finally:
        capture.release()


def valid_crop(movie, output, identity=None, crop_dim=None, *, source_shape=None):
    """Check full source-frame coverage, square dimensions, and the optional receipt."""
    try:
        if not Path(output).is_file():
            return False
        source_count,_,_ = source_shape if source_shape is not None else video_shape(movie)
        rows,width,height = video_shape(output)
        if rows != source_count or width != height or (crop_dim and width != crop_dim):
            return False
        if identity is None or receipt_matches(output, identity):
            return True
        # A frozen recovery bundle may pin previous producer hashes. Inputs,
        # scientific settings, output bytes and decoded shape still must match.
        # Keep the old receipt: reuse must never relabel old pixels as new code.
        for code in os.environ.get('AUTOBEHAVER_CROP_REUSE_CODES', '').split(','):
            if re.fullmatch(r'[0-9a-f]{64}', code) and receipt_matches(output, dict(identity, code=code)):
                return True
        return False
    except (OSError, ValueError, cv2.error):
        return False
