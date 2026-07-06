import math


def test_compute_output_fps_and_interval_no_cap():
    from autobehaver.postinf.infmovie.movies import _compute_output_fps_and_interval

    fps_out, interval, capped = _compute_output_fps_and_interval(40.0, 1.0, 40.0)
    assert fps_out == 40.0
    assert interval == 1.0
    assert capped is False


def test_compute_output_fps_and_interval_cap_preserves_speed_integer_drop():
    from autobehaver.postinf.infmovie.movies import _compute_output_fps_and_interval

    fps_out, interval, capped = _compute_output_fps_and_interval(40.0, 2.0, 40.0)
    assert fps_out == 40.0
    assert interval == 2.0
    assert capped is True


def test_compute_output_fps_and_interval_cap_preserves_speed_fractional_drop():
    from autobehaver.postinf.infmovie.movies import _compute_output_fps_and_interval

    fps_out, interval, capped = _compute_output_fps_and_interval(40.0, 2.0, 60.0)
    assert fps_out == 60.0
    assert math.isclose(interval, 80.0 / 60.0, rel_tol=1e-9)
    assert capped is True


def test_compute_output_fps_and_interval_downsample_to_cap_speed_1x():
    from autobehaver.postinf.infmovie.movies import _compute_output_fps_and_interval

    fps_out, interval, capped = _compute_output_fps_and_interval(60.0, 1.0, 40.0)
    assert fps_out == 40.0
    assert math.isclose(interval, 60.0 / 40.0, rel_tol=1e-9)
    assert capped is True


def test_compute_output_fps_and_interval_invalid_cap_is_no_cap():
    from autobehaver.postinf.infmovie.movies import _compute_output_fps_and_interval

    fps_out, interval, capped = _compute_output_fps_and_interval(40.0, 2.0, 0.0)
    assert fps_out == 80.0
    assert interval == 1.0
    assert capped is False


def test_compute_output_fps_and_interval_hard_clamp_240():
    from autobehaver.postinf.infmovie.movies import _compute_output_fps_and_interval

    fps_out, interval, capped = _compute_output_fps_and_interval(120.0, 3.0, 1000.0)
    # cap is clamped to 240. desired is 360.
    assert fps_out == 240.0
    assert math.isclose(interval, 360.0 / 240.0, rel_tol=1e-9)
    assert capped is True
