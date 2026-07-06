"""Tests for the embedding-cache registry and its CLI."""

import pytest
import torch
from torch.utils.data import TensorDataset
from typer.testing import CliRunner

from drift_happens.cli.main import app
from drift_happens.dataset import embedding_cache
from drift_happens.dataset.dataset import TensorDatasetCache

runner = CliRunner()


def _make_cache(base, dataset_id: str, n_files: int) -> TensorDatasetCache:
    cache = TensorDatasetCache(cache_dir=base / dataset_id, dataset_id=dataset_id)
    cache.cache_dir.mkdir(parents=True)
    for i in range(n_files):
        torch.save(
            TensorDataset(torch.tensor([[i]])),
            cache.cache_dir / f"{dataset_id}_embed_k{i}.pt",
        )
    # An unrelated file that must never be touched by the cache helpers.
    (cache.cache_dir / "keep.txt").write_text("keep")
    return cache


def _make_feature_cache(root, dataset: str, producer: str, kind: str) -> None:
    cache_dir = root / dataset / f"{dataset}-variant" / producer / kind
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text("{}")
    torch.save((torch.zeros(1),), cache_dir / "chunk-00000.pt")


def _make_arxiv_text_cache(root) -> None:
    _make_feature_cache(root, "arxiv", "roberta-base", "sequence_embedding_dataset")


def _make_amazon_text_cache(root) -> None:
    _make_feature_cache(
        root, "amazon_reviews_23", "bert-base", "pooled_embedding_dataset"
    )


@pytest.fixture
def registry(tmp_path, monkeypatch):
    caches = {
        "yearbook": _make_cache(tmp_path, "yearbook", 2),
        "arxiv": _make_cache(tmp_path, "arxiv", 1),
    }
    monkeypatch.setattr(embedding_cache, "EMBEDDING_CACHES", caches)
    # Isolate the text feature-cache tree so image-only tests see no text caches.
    monkeypatch.setattr(
        embedding_cache, "FEATURE_CACHE_ROOT", tmp_path / "feature_cache"
    )
    return caches


@pytest.fixture
def feature_caches(tmp_path, monkeypatch):
    root = tmp_path / "feature_cache"
    _make_arxiv_text_cache(root)
    _make_amazon_text_cache(root)
    monkeypatch.setattr(embedding_cache, "FEATURE_CACHE_ROOT", root)
    # Isolate image caches so this fixture is fully hermetic.
    monkeypatch.setattr(embedding_cache, "EMBEDDING_CACHES", {})
    return root


def test_cached_files_lists_only_embedding_files(registry) -> None:
    files = registry["yearbook"].cached_files()
    assert len(files) == 2
    assert all(f.name.startswith("yearbook_embed_") for f in files)


def test_cached_files_empty_when_dir_missing(tmp_path) -> None:
    cache = TensorDatasetCache(cache_dir=tmp_path / "absent", dataset_id="yearbook")
    assert cache.cached_files() == ()


def test_select_caches_all_and_single(registry) -> None:
    assert set(embedding_cache.select_caches(None)) == {"yearbook", "arxiv"}
    assert set(embedding_cache.select_caches("yearbook")) == {"yearbook"}


def test_select_caches_unknown_raises(registry) -> None:
    with pytest.raises(KeyError):
        embedding_cache.select_caches("does_not_exist")


def test_delete_cache_files_removes_only_embeddings(registry) -> None:
    cache = registry["yearbook"]
    deleted = embedding_cache.delete_cache_files(cache.cached_files())

    assert len(deleted) == 2
    assert cache.cached_files() == ()
    assert (cache.cache_dir / "keep.txt").exists()


def test_delete_cache_files_refuses_foreign_path(registry, tmp_path) -> None:
    stray = tmp_path / "elsewhere.pt"
    stray.write_text("x")
    with pytest.raises(ValueError, match="refusing to delete"):
        embedding_cache.delete_cache_files((stray,))
    assert stray.exists()


def test_delete_cache_files_refuses_wrong_name_inside_cache_dir(registry) -> None:
    # A file inside a registered cache dir must still match the {dataset_id}_embed_ prefix.
    keep = registry["yearbook"].cache_dir / "keep.txt"
    assert keep.exists()
    with pytest.raises(ValueError, match="refusing to delete"):
        embedding_cache.delete_cache_files((keep,))
    assert keep.exists()


def test_cli_cache_ls_reports_counts(registry) -> None:
    result = runner.invoke(app, ["artifacts", "cache", "ls"])
    assert result.exit_code == 0
    assert "image\tyearbook\t-\t-\t2" in result.output
    assert "image\tarxiv\t-\t-\t1" in result.output


def test_cli_cache_clear_dry_run_keeps_files(registry) -> None:
    result = runner.invoke(
        app, ["artifacts", "cache", "clear", "--dataset", "yearbook"]
    )
    assert result.exit_code == 0
    assert "dry-run: 2 file(s) and 0 dir(s) would be deleted" in result.output
    assert registry["yearbook"].cached_files()  # nothing deleted


def test_cli_cache_clear_apply_deletes_selected_only(registry) -> None:
    result = runner.invoke(
        app, ["artifacts", "cache", "clear", "--dataset", "yearbook", "--apply"]
    )
    assert result.exit_code == 0
    assert "deleted 2 file(s) and 0 dir(s)" in result.output
    assert registry["yearbook"].cached_files() == ()
    assert len(registry["arxiv"].cached_files()) == 1  # untouched
    assert (registry["yearbook"].cache_dir / "keep.txt").exists()


def test_cli_cache_clear_unknown_dataset_errors(registry) -> None:
    result = runner.invoke(app, ["artifacts", "cache", "clear", "--dataset", "nope"])
    assert result.exit_code != 0
    assert "unknown dataset" in result.output


def test_cli_cache_clear_handles_text_only_dataset(registry) -> None:
    # amazon_reviews_23 has no image embedding cache, only a text feature cache.
    _make_amazon_text_cache(embedding_cache.FEATURE_CACHE_ROOT)

    result = runner.invoke(
        app,
        ["artifacts", "cache", "clear", "--dataset", "amazon_reviews_23", "--apply"],
    )

    assert result.exit_code == 0
    assert "deleted 0 file(s) and 1 dir(s)" in result.output
    assert embedding_cache.discover_feature_caches() == ()


def test_discover_feature_caches_parses_path(feature_caches) -> None:
    entries = embedding_cache.discover_feature_caches()
    by_dataset = {entry.dataset: entry for entry in entries}
    assert set(by_dataset) == {"arxiv", "amazon_reviews_23"}
    assert by_dataset["arxiv"].producer == "roberta-base"
    assert by_dataset["arxiv"].kind == "sequence_embedding_dataset"
    assert len(by_dataset["arxiv"].files) == 2  # manifest.json + chunk


def test_delete_feature_cache_dirs_removes_dir(feature_caches) -> None:
    entries = embedding_cache.discover_feature_caches()
    target = next(e for e in entries if e.dataset == "arxiv")

    deleted = embedding_cache.delete_feature_cache_dirs((target.path,))

    assert len(deleted) == 1
    assert not target.path.exists()
    remaining = {e.dataset for e in embedding_cache.discover_feature_caches()}
    assert remaining == {"amazon_reviews_23"}


def test_delete_feature_cache_dirs_refuses_non_cache(feature_caches, tmp_path) -> None:
    # A directory inside the root but with no manifest.json is refused.
    stray = tmp_path / "feature_cache" / "stray"
    stray.mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing to delete"):
        embedding_cache.delete_feature_cache_dirs((stray,))
    assert stray.exists()


def test_delete_feature_cache_dirs_refuses_path_outside_root(
    feature_caches, tmp_path
) -> None:
    # A directory outside the root is refused even when it contains a manifest.json.
    outside = tmp_path / "outside_cache"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="refusing to delete"):
        embedding_cache.delete_feature_cache_dirs((outside,))
    assert outside.exists()


def test_cli_cache_clear_covers_text_caches(registry) -> None:
    _make_arxiv_text_cache(embedding_cache.FEATURE_CACHE_ROOT)
    _make_amazon_text_cache(embedding_cache.FEATURE_CACHE_ROOT)

    result = runner.invoke(
        app, ["artifacts", "cache", "clear", "--dataset", "arxiv", "--apply"]
    )

    assert result.exit_code == 0
    assert "deleted 1 file(s) and 1 dir(s)" in result.output
    remaining = {e.dataset for e in embedding_cache.discover_feature_caches()}
    assert remaining == {"amazon_reviews_23"}  # other dataset's text cache untouched


def test_discover_ignores_manifest_at_wrong_depth(feature_caches) -> None:
    stray = feature_caches / "arxiv" / "manifest.json"  # depth 2, not a leaf cache
    stray.write_text("{}")

    datasets = sorted(e.dataset for e in embedding_cache.discover_feature_caches())

    assert datasets == ["amazon_reviews_23", "arxiv"]  # the stray is not listed
    assert stray.exists()  # and certainly not deleted


def test_cli_cache_ls_shows_text_rows(feature_caches) -> None:
    result = runner.invoke(app, ["artifacts", "cache", "ls"])
    assert result.exit_code == 0
    assert "text\tarxiv\troberta-base\tsequence_embedding_dataset" in result.output
    assert (
        "text\tamazon_reviews_23\tbert-base\tpooled_embedding_dataset" in result.output
    )


def test_cli_cache_clear_all_removes_image_and_text(registry) -> None:
    _make_arxiv_text_cache(embedding_cache.FEATURE_CACHE_ROOT)

    result = runner.invoke(app, ["artifacts", "cache", "clear", "--apply"])

    assert result.exit_code == 0
    assert "deleted 3 file(s) and 1 dir(s)" in result.output
    assert all(not cache.cached_files() for cache in registry.values())
    assert embedding_cache.discover_feature_caches() == ()
