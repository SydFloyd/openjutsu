"""CPU checks for the temporal boundaries used by real generation jobs."""
import numpy as np
import pytest

from app import video


@pytest.mark.parametrize("capacity,overlap", [(77, 5), (81, 9)])
@pytest.mark.parametrize("total", [1, 33, 77, 81, 82, 480])
def test_windows_cover_each_source_frame_once(total, capacity, overlap):
    seen = []
    for window in video.windows(total, capacity, overlap):
        assert window["length"] >= window["count"]
        assert window["length"] <= capacity
        assert (window["length"] - 1) % 4 == 0
        seen.extend(range(window["start"] + window["overlap"], window["start"] + window["count"]))
    assert seen == list(range(total))


def test_recast_join_drops_only_duplicate_context(tmp_path):
    plan = list(video.windows(13, 9, 5))
    source = [np.full((32, 32, 3), i * 15, dtype=np.uint8) for i in range(13)]
    chunks = []
    for i, window in enumerate(plan):
        path = tmp_path / f"{i}.mkv"
        frames = source[window["start"]:window["start"] + window["count"]]
        video.write_frames(frames, path, lossless=True)
        chunks.append(path)
    output = tmp_path / "result.mp4"
    video.assemble(chunks, plan, output, 13, "recast")
    decoded = video.read_frames(output)
    assert len(decoded) == 13
    assert max(abs(float(a.mean()) - float(b.mean())) for a, b in zip(decoded, source)) < 3


def test_vace_join_blends_overlap_without_changing_duration(tmp_path):
    plan = list(video.windows(13, 9, 5))
    chunks = []
    for i, window in enumerate(plan):
        path = tmp_path / f"{i}.mkv"
        video.write_frames([np.full((32, 32, 3), i * 200, dtype=np.uint8)] * window["count"], path, lossless=True)
        chunks.append(path)
    output = tmp_path / "result.mp4"
    video.assemble(chunks, plan, output, 13, "swap")
    means = [float(frame.mean()) for frame in video.read_frames(output)]
    assert len(means) == 13
    assert means[3] < 3 and means[9] > 197
    assert all(a < b for a, b in zip(means[3:9], means[4:10]))


def test_mask_slice_remains_binary_and_pads_last_frame(tmp_path):
    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    source, output = tmp_path / "mask.mkv", tmp_path / "padded.mkv"
    video.write_frames([mask] * 9, source, lossless=True)
    video.slice_video(source, output, start=4, count=5, length=9, mask=True)
    frames = video.read_frames(output)
    assert len(frames) == 9
    assert set(np.unique(frames)) == {0, 255}
    assert all(np.array_equal(frame, mask) for frame in frames)


def test_upload_normalization_preserves_duration(tmp_path):
    source, output = tmp_path / "30fps.mp4", tmp_path / "16fps.mp4"
    frame = np.full((64, 64, 3), 100, dtype=np.uint8)
    video.write_frames([frame] * 30, source, fps=30)
    info = video.normalize(source, output)
    assert info == {"width": 832, "height": 480, "frames": 16, "fps": 16.0}


def test_assembly_restores_source_audio(tmp_path):
    import av

    source, output = tmp_path / "with-audio.mp4", tmp_path / "result.mp4"
    video.ffmpeg("-f", "lavfi", "-i", "color=size=32x32:rate=16:duration=1",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-c:v", "libx264", "-c:a", "aac", "-shortest", source)
    plan = list(video.windows(16, 81, 9))
    video.assemble([source], plan, output, 16, "restyle", audio_source=source)
    with av.open(str(output)) as container:
        assert len(container.streams.audio) == 1
        assert sum(frame.samples for frame in container.decode(audio=0)) >= 16000
    assert video.probe(output)["frames"] == 16
