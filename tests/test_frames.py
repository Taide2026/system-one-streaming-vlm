"""Unit read_frames: the sequential reader returns the same pixels as one seek per frame. Synthetic MJPG video, no model."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from s1stream.input_forms import frames
from s1stream.input_forms.frames import FrameReadError, read_frames

N_FRAMES = 150


def value(k: int) -> int:
    """Grey level of frame k: neighbours differ by 7, so an off-by-one read is visible."""
    return (k * 7) % 250


def make_video(path: Path, n: int = N_FRAMES) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened()
    for k in range(n):
        writer.write(np.full((48, 64, 3), value(k), np.uint8))
    writer.release()
    return path


def seek_per_frame(path: Path, indices) -> list[np.ndarray]:
    """The reference: the old read_frames, one seek per frame."""
    cap = cv2.VideoCapture(str(path))
    out = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        assert ok
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def check(path: Path, indices) -> None:
    """read_frames equals the reference pixel for pixel, and each frame has the grey level of its index."""
    got = [np.asarray(im) for im in read_frames(path, indices)]
    assert len(got) == len(indices)
    for index, image, ref in zip(indices, got, seek_per_frame(path, indices)):
        assert np.array_equal(image, ref), f"frame {index} differs from the seek reference"
        assert abs(float(image.mean()) - value(index)) < 3, f"frame {index} has the wrong content"


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "v.avi")


@pytest.fixture(autouse=True)
def fresh_reader():
    """Each test starts with no open video in this thread."""
    frames.close_reader()
    yield
    frames.close_reader()


def test_consecutive_windows_equal_seek_per_frame(video):
    for start in range(0, 120, 30):
        check(video, [start + 4 * j for j in range(8)])


def test_returns_rgb_pil_images(video):
    images = read_frames(video, [0, 1])
    assert [im.mode for im in images] == ["RGB", "RGB"]
    assert images[0].size == (64, 48)


def test_repeated_and_backward_indices(video):
    check(video, [5, 5, 6])
    check(video, [100, 2, 50])  # backward jump inside one call
    check(video, [10, 11])  # backward jump between calls


def test_large_forward_gap_seeks(video, monkeypatch):
    monkeypatch.setattr(frames, "MAX_GRAB", 5)
    seeks = []
    real_seek = frames.SequentialReader.seek

    def counting_seek(self, index):
        seeks.append(index)
        real_seek(self, index)

    monkeypatch.setattr(frames.SequentialReader, "seek", counting_seek)
    check(video, [0, 3, 20, 140])
    assert seeks == [20, 140]  # skips of 2, 16 and 119 frames; only the last two are above MAX_GRAB


def test_two_videos_interleaved(tmp_path, video):
    other = make_video(tmp_path / "w.avi", 40)
    check(video, [10, 20])
    check(other, [1, 2])
    check(video, [30, 40])


def test_read_past_the_end_raises_and_the_next_read_works(video):
    with pytest.raises(FrameReadError, match=r"v\.avi.*frame 152"):
        read_frames(video, [140, 149, 149 + 3])
    with pytest.raises(FrameReadError):
        read_frames(video, [N_FRAMES + 500])
    check(video, [0, 1])


def test_parallel_threads_read_the_same_video(video):
    windows = [[start + 3 * j for j in range(8)] for start in range(0, 120, 12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda w: [np.asarray(im) for im in read_frames(video, w)], windows))
    for window, got in zip(windows, results):
        for image, ref in zip(got, seek_per_frame(video, window)):
            assert np.array_equal(image, ref)


def test_a_forked_child_opens_its_own_reader(video):
    import signal
    import time

    read_frames(video, [0, 1])
    pid = os.fork()
    if pid == 0:  # child: no inherited reader, and a fresh read works
        ok = getattr(frames._local, "reader", None) is None
        ok = ok and abs(float(np.asarray(read_frames(video, [9])[0]).mean()) - value(9)) < 3
        os._exit(0 if ok else 1)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pytest.fail("the forked child hung for 20 s")
    assert os.waitstatus_to_exitcode(status) == 0
    check(video, [2, 3])  # the parent's reader still works
