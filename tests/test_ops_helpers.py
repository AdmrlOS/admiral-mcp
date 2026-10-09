"""Pure helpers behind the operations tools: spec normalisation, edits and diffs."""

from __future__ import annotations

import pytest

from admrl_mcp import ops


def base_spec(**over):
    spec = {
        "image": "registry.example.com/kiosk:1.0",
        "desiredState": "RUNNING",
        "command": ["run"],
        "environment": {"MODE": "prod", "DB_PASSWORD": "hunter2"},
        "mounts": [{"type": "bind", "source": "/data", "destination": "/data"}],
        "options": {"privileged": False, "tty": False},
        "volumes": {"cache": {"source": "/var/cache", "destination": "/cache"}},
        "ports": [{"port": 80, "protocol": "tcp"}],
    }
    spec.update(over)
    return spec


def test_norm_spec_fills_nulls_and_canonicalises_like_the_backend():
    out = ops.norm_spec(
        {"image": "  img:1 ", "desiredState": "stopped", "command": None, "environment": None, "mounts": None,
         "options": None, "volumes": None, "ports": [{"port": 53, "protocol": "UDP"}, {"port": 80}]}
    )
    assert out["image"] == "img:1" and out["desiredState"] == "STOPPED"
    assert out["command"] == [] and out["environment"] == {} and out["volumes"] == {}
    assert out["ports"] == [{"port": 53, "protocol": "udp"}, {"port": 80, "protocol": "tcp"}]
    assert ops.norm_spec(None)["desiredState"] == "RUNNING"


def test_norm_spec_does_not_mutate_input():
    src = {"image": "a", "ports": [{"port": 1, "protocol": "TCP"}]}
    ops.norm_spec(src)
    assert src["ports"][0]["protocol"] == "TCP"


@pytest.mark.parametrize(
    "image,tag,expected",
    [
        ("registry.example.com/app:1.0", "2.0", "registry.example.com/app:2.0"),
        ("registry.example.com:5000/team/app:1.0", "2.0", "registry.example.com:5000/team/app:2.0"),
        ("registry.example.com:5000/team/app", "2.0", "registry.example.com:5000/team/app:2.0"),
        ("app@sha256:abc", "3", "app:3"),
        ("app:1@sha256:abc", "3", "app:3"),
    ],
)
def test_retag_image(image, tag, expected):
    assert ops.retag_image(image, tag) == expected


@pytest.mark.parametrize("image,tag", [("", "1"), ("app:1", ""), ("app:1", "a b"), ("app:1", "a/b")])
def test_retag_image_rejects_bad_input(image, tag):
    with pytest.raises(ValueError):
        ops.retag_image(image, tag)


def test_merge_patch_is_rfc7386():
    target = {"a": {"b": 1, "c": 2}, "list": [1, 2], "gone": 1}
    out = ops.merge_patch(target, {"a": {"b": None, "d": 4}, "list": [9], "gone": None, "new": {"x": 1}})
    assert out == {"a": {"c": 2, "d": 4}, "list": [9], "new": {"x": 1}}
    assert target["a"]["b"] == 1  # input untouched
    assert ops.merge_patch({"a": 1}, "scalar") == "scalar"


def test_apply_edits_image_tag_env_and_patch_order():
    spec = base_spec()
    new, notes = ops.apply_spec_edits(
        spec,
        image_tag="2.0",
        env_set={"MODE": "debug", "PORT": 8080},
        env_unset=["DB_PASSWORD", "NOPE"],
        patch={"environment": {"MODE": None}, "desiredState": "stopped"},
    )
    assert new["image"] == "registry.example.com/kiosk:2.0"
    assert new["environment"] == {"PORT": "8080"}  # patch had the last word on MODE, values are strings
    assert new["desiredState"] == "STOPPED"
    assert notes == ["env_unset: NOPE was not set (ignored)"]
    assert spec["environment"]["MODE"] == "prod"


def test_apply_edits_full_replacement_is_exclusive():
    new, _ = ops.apply_spec_edits(base_spec(), replace={"image": "other:1"})
    assert new["image"] == "other:1" and new["environment"] == {}
    with pytest.raises(ValueError, match="cannot be combined"):
        ops.apply_spec_edits(base_spec(), replace={"image": "x"}, image_tag="2")
    with pytest.raises(ValueError, match="JSON object"):
        ops.apply_spec_edits(base_spec(), replace=["nope"])  # type: ignore[arg-type]


def test_apply_edits_validates_arguments():
    with pytest.raises(ValueError, match="not both"):
        ops.apply_spec_edits(base_spec(), image="a:1", image_tag="2")
    with pytest.raises(ValueError, match="empty"):
        ops.apply_spec_edits(base_spec(), env_set={" ": "x"})
    with pytest.raises(ValueError, match="NAME: value"):
        ops.apply_spec_edits(base_spec(), env_set=["A"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="merge patch"):
        ops.apply_spec_edits(base_spec(), patch=["A"])  # type: ignore[arg-type]
    new, _ = ops.apply_spec_edits(base_spec(), image="full/ref:9")
    assert new["image"] == "full/ref:9"


def test_spec_diff_rows_and_null_equals_empty():
    old = base_spec(command=None, mounts=None)
    new = base_spec(
        image="registry.example.com/kiosk:2.0",
        environment={"MODE": "prod", "NEW": "1"},
        ports=[{"port": 80, "protocol": "tcp"}, {"port": 443, "protocol": "tcp"}],
        volumes={},
        command=[],
    )
    rows = {r["path"]: r for r in ops.spec_diff(old, new)}
    assert rows["image"] == {"path": "image", "change": "changed", "old": "registry.example.com/kiosk:1.0", "new": "registry.example.com/kiosk:2.0"}
    assert rows["environment.NEW"]["change"] == "added" and rows["environment.NEW"]["new"] == "1"
    assert rows["environment.DB_PASSWORD"]["change"] == "removed"
    assert rows["ports[tcp/443]"]["change"] == "added"
    assert rows["volumes.cache.source"]["change"] == "removed"
    assert "command" not in rows and "ports[tcp/80]" not in rows
    assert "mounts[/data]" in rows  # removed in the second spec? mounts present in new via base_spec(mounts default)


def test_spec_diff_masks_credential_values_but_not_when_asked():
    old, new = base_spec(), base_spec(environment={"MODE": "prod", "DB_PASSWORD": "rotated"})
    masked = ops.spec_diff(old, new)
    assert masked == [{"path": "environment.DB_PASSWORD", "change": "changed", "old": "***", "new": "***"}]
    clear = ops.spec_diff(old, new, mask=False)
    assert clear[0]["old"] == "hunter2" and clear[0]["new"] == "rotated"


def test_spec_diff_identical_and_mount_changes():
    assert ops.spec_diff(base_spec(), base_spec()) == []
    moved = base_spec(mounts=[{"type": "bind", "source": "/other", "destination": "/data"}])
    rows = ops.spec_diff(base_spec(), moved)
    assert rows[0]["path"] == "mounts[/data]" and rows[0]["change"] == "changed"


def test_diff_summary_and_spec_summary():
    assert ops.diff_summary([]) == "no changes"
    rows = [{"path": f"environment.V{i}", "change": "added", "new": "x"} for i in range(8)]
    text = ops.diff_summary(rows)
    assert text.endswith("+2 more") and "environment.V0 added" in text
    long = [{"path": "image", "change": "changed", "old": "a" * 100, "new": "b"}]
    assert "..." in ops.diff_summary(long)
    s = ops.spec_summary(base_spec(signaturePolicy={"required": True}))
    assert s["env_vars"] == 2 and s["ports"] == 1 and s["signature_policy"] is True and s["desiredState"] == "RUNNING"


def test_mask_spec_only_masks_credential_names():
    masked = ops.mask_spec(base_spec())
    assert masked["environment"] == {"MODE": "prod", "DB_PASSWORD": "***"}
    assert ops.is_secret_key("API_TOKEN") and ops.is_secret_key("apiKey") and not ops.is_secret_key("MODE")
