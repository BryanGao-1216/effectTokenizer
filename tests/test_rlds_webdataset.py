import io
import itertools
import pickle
import sys
import tarfile
import types
from pathlib import Path

import numpy as np

from scripts.action_vqvae.rlds.webdataset import (
    iter_openx_tar_episodes,
    load_or_compute_openx_action_statistics,
    resolve_openx_tar_paths,
    stack_openx_episode_steps,
    transform_openx_tar_episode,
)


def _write_episode_tar(path, payload):
    serialized = pickle.dumps(payload)
    member = tarfile.TarInfo("sample_000000000000.data.pickle")
    member.size = len(serialized)
    with tarfile.open(path, "w") as archive:
        archive.addfile(member, io.BytesIO(serialized))


def test_openx_tar_stream_and_statistics_cache(tmp_path):
    source_dir = tmp_path / "toy"
    source_dir.mkdir()
    tar_path = source_dir / "toy_00000.tar"
    _write_episode_tar(
        tar_path,
        {
            "steps": [
                {"action": np.array([1.0, 3.0], dtype=np.float32)},
                {"action": np.array([5.0, 7.0], dtype=np.float32)},
            ]
        },
    )

    paths = resolve_openx_tar_paths(tmp_path, "toy")
    trajectory = stack_openx_episode_steps(next(iter_openx_tar_episodes(paths)))
    assert np.array_equal(trajectory["action"], [[1.0, 3.0], [5.0, 7.0]])
    assert not list(source_dir.glob("*.data.pickle"))

    class FakeTensorFlow:
        string = "string"

        @staticmethod
        def convert_to_tensor(value, dtype=None):
            return np.asarray(value, dtype=object if dtype == "string" else None)

    statistics = load_or_compute_openx_action_statistics(
        paths=paths,
        tf=FakeTensorFlow,
        standardize_fn=lambda value: value,
        hash_dependencies=("toy",),
    )
    assert statistics["num_trajectories"] == 1
    assert statistics["num_transitions"] == 2
    assert statistics["action"]["mean"] == [3.0, 5.0]
    assert len(list(source_dir.glob("dataset_statistics_*.json"))) == 1


def test_openx_tar_statistics_apply_frequency_transform_before_caching(tmp_path):
    source_dir = tmp_path / "toy_resampled"
    source_dir.mkdir()
    tar_path = source_dir / "toy_00000.tar"
    _write_episode_tar(
        tar_path,
        {
            "steps": [
                {"action": np.array([1.0, 0.0], dtype=np.float32)},
                {"action": np.array([2.0, 1.0], dtype=np.float32)},
                {"action": np.array([3.0, 1.0], dtype=np.float32)},
                {"action": np.array([4.0, 0.0], dtype=np.float32)},
            ]
        },
    )

    class FakeTensorFlow:
        string = "string"

        @staticmethod
        def convert_to_tensor(value, dtype=None):
            return np.asarray(value, dtype=object if dtype == "string" else None)

    statistics = load_or_compute_openx_action_statistics(
        paths=(tar_path,),
        tf=FakeTensorFlow,
        standardize_fn=lambda value: value,
        hash_dependencies=("toy-resampled",),
        action_transform=lambda action: action.reshape(2, 2, 2).sum(axis=1),
    )
    assert statistics["num_transitions"] == 2
    assert statistics["action"]["mean"] == [5.0, 1.0]


def test_auto_storage_mixes_tar_and_tfds_at_chunk_level(monkeypatch, tmp_path):
    try:
        import tensorflow_graphics  # noqa: F401
    except ModuleNotFoundError:
        module_names = [
            "tensorflow_graphics",
            "tensorflow_graphics.geometry",
            "tensorflow_graphics.geometry.transformation",
            "tensorflow_graphics.geometry.transformation.euler",
            "tensorflow_graphics.geometry.transformation.quaternion",
        ]
        for module_name in module_names:
            monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
        for parent, child in zip(module_names[:-1], module_names[1:], strict=True):
            setattr(
                sys.modules[parent],
                child.rsplit(".", 1)[-1],
                sys.modules[child],
            )

    from scripts.action_vqvae import oxe_dataset as module
    from scripts.action_vqvae.rlds.oxe.materialize import make_oxe_dataset_kwargs

    assert (
        make_oxe_dataset_kwargs(
            "fmb_dataset",
            tmp_path,
            load_camera_views=(),
            load_depth=False,
            load_proprio=False,
            load_language=False,
        )["name"]
        == "fmb"
    )

    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge_00000.tar").touch()
    captured_standardizer = {}

    def fake_bridge_statistics(**kwargs):
        captured_standardizer["fn"] = kwargs["standardize_fn"]
        zeros = np.zeros(7, dtype=np.float32)
        return {
            "action": {
                "mean": zeros,
                "std": zeros,
                "min": zeros,
                "max": zeros,
                "q01": zeros,
                "q99": zeros,
            },
            "num_transitions": 10,
            "num_trajectories": 2,
        }

    monkeypatch.setattr(
        module,
        "load_or_compute_openx_action_statistics",
        fake_bridge_statistics,
    )
    bridge_dataset = object.__new__(module.OXEActionDataset)
    bridge_dataset.data_root_dir = tmp_path
    bridge_dataset._webdataset_sources = []
    bridge_dataset._initialize_webdataset(
        [
            {
                "name": "bridge_orig",
                "standardize_fn": module.OXE_STANDARDIZATION_TRANSFORMS["bridge_orig"],
                "action_normalization_mask": [True] * 6 + [False],
                "absolute_action_mask": [False] * 6 + [True],
            }
        ],
        [1.0],
        False,
    )
    assert (
        captured_standardizer["fn"]
        is module.OXE_STANDARDIZATION_TRANSFORMS["bridge_oxe"]
    )

    import tensorflow as tf

    bridge_payload = {
        "steps": [
            {
                "observation": {
                    "state": np.asarray([step, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                    "natural_language_instruction": "move",
                },
                "action": {
                    "world_vector": np.zeros(3, dtype=np.float32),
                    "rotation_delta": np.zeros(3, dtype=np.float32),
                    "open_gripper": np.float32(1),
                },
            }
            for step in range(4)
        ]
    }
    bridge_trajectory = transform_openx_tar_episode(
        bridge_payload,
        tf=tf,
        transform=module.OXE_STANDARDIZATION_TRANSFORMS["bridge_oxe"],
    )
    assert tuple(bridge_trajectory["action"].shape) == (2, 7)

    mixture = [("tar_source", 1.0), ("tfds_source", 1.0)]
    monkeypatch.setattr(module, "OXE_NAMED_MIXTURES", {"toy_hybrid": mixture})
    per_dataset_kwargs = [
        {"name": "tar_source"},
        {"name": "tfds_source"},
    ]
    monkeypatch.setattr(
        module,
        "get_oxe_dataset_kwargs_and_weights",
        lambda *args, **kwargs: (per_dataset_kwargs, [1.0, 1.0]),
    )
    monkeypatch.setattr(
        module,
        "resolve_openx_tar_paths",
        lambda root, name: (
            (Path(root) / name / "source.tar",) if name == "tar_source" else ()
        ),
    )

    def statistics(num_transitions):
        return {
            "action": {"mean": np.zeros(7, dtype=np.float32)},
            "num_transitions": num_transitions,
            "num_trajectories": 2,
        }

    def fake_initialize_webdataset(self, kwargs, weights, balance_weights):
        self._webdataset_sources = [{"name": "tar_source"}]
        return {"tar_source": statistics(10)}, [10]

    monkeypatch.setattr(
        module.OXEActionDataset,
        "_initialize_webdataset",
        fake_initialize_webdataset,
    )

    class FakeTFDSDataset:
        def as_numpy_iterator(self):
            yield from itertools.repeat(
                {"action": np.full((16, 7), 2.0, dtype=np.float32)}
            )

    def fake_make_tfds(**kwargs):
        assert kwargs["apply_shuffle"] is False
        return FakeTFDSDataset(), 20, {"tfds_source": statistics(20)}

    monkeypatch.setattr(module, "make_interleaved_action_dataset", fake_make_tfds)

    dataset = module.OXEActionDataset(
        tmp_path,
        "toy_hybrid",
        horizon=16,
        action_dim=7,
        train=False,
        shuffle_buffer_size=32,
        balance_weights=False,
        storage_format="auto",
        seed=7,
    )
    dataset._iter_weighted_webdataset_chunks = lambda: iter(
        itertools.repeat(np.full((16, 7), 1.0, dtype=np.float32))
    )

    chunks = list(dataset._iter_hybrid_chunks())
    assert dataset.storage_format == "hybrid"
    assert np.array_equal(dataset._hybrid_backend_weights, [0.5, 0.5])
    assert len(chunks) == 32
    assert {float(chunk[0, 0]) for chunk in chunks} == {1.0, 2.0}
