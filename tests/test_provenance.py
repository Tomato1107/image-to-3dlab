import pytest

from image_to_3dlab.provenance import LICENSES, validate_run_policy


def test_sf3d_is_allowed_when_conditionals_are_enabled():
    profile = validate_run_policy("sf3d", "game", "worldwide", True)
    assert profile.classification == "commercial-conditional"


def test_hunyuan_is_blocked_for_worldwide_game():
    with pytest.raises(ValueError, match="worldwide"):
        validate_run_policy("hunyuan-comfyui", "game", "worldwide", True)


def test_trellis_inherits_dinov3_conditional_classification():
    profile = validate_run_policy("trellis2", "game", "worldwide", True)
    assert profile.classification == "commercial-conditional"


def test_pixal3d_records_split_encoder_license():
    profile = validate_run_policy("pixal3d", "game", "worldwide", True)
    assert profile.classification == "commercial-conditional"
    assert "DINOv3" in " ".join(profile.conditions)


def test_trellis_is_blocked_when_manifest_disallows_conditionals():
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy("trellis2", "game", "worldwide", False)


def test_qwen_image_is_classified_research_only():
    """Qwen-Image-2.1 is under the Qwen Research License: non-commercial only. It is the
    first non-commercial model in this repo and needed a class of its own, because
    `commercial-conditional` and `territory-restricted` both permit commercial use under
    conditions and this permits none."""
    profile = LICENSES["qwen-image-2.1"]
    assert profile.classification == "research-only"
    assert profile.folder == "research_only"
    assert any("NON-COMMERCIAL" in c for c in profile.conditions)
    assert any("Built with Qwen" in c for c in profile.conditions)


def test_qwen_outputs_are_yours_but_running_it_commercially_is_not():
    """Mirrors Qwen's 2026-09-21 statement without going past the licence text."""
    conditions = " ".join(LICENSES["qwen-image-2.1"].conditions)
    assert "https://x.com/QwenDevs/status/2101917379785838660" in conditions
    assert "commercial licence" in conditions
    assert "inherits the restriction" not in conditions


def test_research_only_is_refused_even_when_conditional_is_allowed():
    """allow_conditional is not consent to a non-commercial model. A manifest written for
    the Hunyuan or SF3D cases must not silently pick up a stricter licence."""
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy(
            "qwen-image-2.1", use_case="game", distribution="private",
            allow_conditional=True,
        )


def test_research_only_is_refused_for_public_distribution():
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy(
            "qwen-image-2.1", use_case="showcase", distribution="public",
            allow_conditional=True,
        )


def test_research_only_is_allowed_for_a_private_showcase():
    profile = validate_run_policy(
        "qwen-image-2.1", use_case="showcase", distribution="private",
        allow_conditional=True,
    )
    assert profile.classification == "research-only"


def test_research_only_still_needs_allow_conditional():
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy(
            "qwen-image-2.1", use_case="showcase", distribution="private",
            allow_conditional=False,
        )


def test_qwen_wording_is_candid_about_the_ambiguity():
    """We do not overrule Qwen either way: we give our reading and send people to the
    licence, so nobody is scared off generating and nobody is told to go ham."""
    from image_to_3dlab.provenance import QWEN_OUTPUT_RIGHTS

    assert "ambiguous" in QWEN_OUTPUT_RIGHTS
    assert "our reading" in QWEN_OUTPUT_RIGHTS
    assert "check Qwen's licence yourself" in QWEN_OUTPUT_RIGHTS
