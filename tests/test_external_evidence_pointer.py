import pytest
from hephaestus.storage.external_evidence_pointer import ExternalEvidencePointer

SHA="sha256:" + "a"*64
REVIEW="sha256:" + "b"*64

def pointer(provider, kind, revision, visibility="private", kaggle_version=None):
    return ExternalEvidencePointer(provider,"kys928/hephaestus-evidence",kind,SHA,REVIEW,visibility,revision,kaggle_version)


def test_huggingface_model_and_dataset_commit_pins():
    assert pointer("huggingface_model","private_checkpoint","a"*40).validate().provider=="huggingface_model"
    assert pointer("huggingface_dataset","internal_training_data","b"*40).validate().provider=="huggingface_dataset"
    with pytest.raises(ValueError,match="40-hex"):
        pointer("huggingface_model","private_checkpoint","main").validate()


def test_kaggle_specific_positive_version_required():
    assert pointer("kaggle_dataset","internal_training_data","12",kaggle_version=12).validate().kaggle_version==12
    for version in (0,-1,True):
        with pytest.raises(ValueError):
            pointer("kaggle_dataset","internal_training_data",str(version),kaggle_version=version).validate()


def test_github_requires_release_tag_and_restricts_asset_type():
    assert pointer("github_release","research_summary","evidence-20261028-v1",visibility="public").validate()
    with pytest.raises(ValueError):
        pointer("github_release","research_summary","latest",visibility="public").validate()
    with pytest.raises(ValueError):
        pointer("github_release","private_checkpoint","frozen-v1").validate()


@pytest.mark.parametrize("kind",["protected_holdout","private_reference","internal_training_data","original_cpu_evidence","private_checkpoint"])
def test_public_exposure_of_internal_material_is_denied(kind):
    with pytest.raises(PermissionError):
        pointer("huggingface_dataset",kind,"a"*40,visibility="public").validate()


def test_pointer_cannot_claim_review_by_omitting_or_forging_shape():
    good=pointer("huggingface_dataset","public_dataset","a"*40,visibility="public")
    d=good.to_dict()
    assert ExternalEvidencePointer.from_dict(d)==good
    with pytest.raises(ValueError):
        ExternalEvidencePointer.from_dict({**d,"github_access_token":"secret"})
    d["review_sha256"]="unreviewed"
    with pytest.raises(ValueError):
        ExternalEvidencePointer.from_dict(d)


def test_kaggle_models_require_exact_variation_version_and_nonpublic_internal_weights():
    assert pointer("kaggle_model","private_checkpoint","pytorch/default/3",kaggle_version=3).validate()
    assert pointer("kaggle_model","public_model","pytorch/default/12",visibility="public",kaggle_version=12).validate()
    for revision, version in (("latest",3),("pytorch/default/4",3),("pytorch/default/0",0),
                              ("pytorch/default/3",True)):
        with pytest.raises(ValueError):
            pointer("kaggle_model","private_checkpoint",revision,kaggle_version=version).validate()
    with pytest.raises(PermissionError):
        pointer("kaggle_model","private_checkpoint","pytorch/default/3",visibility="public",kaggle_version=3).validate()
    with pytest.raises(ValueError):
        pointer("kaggle_model","internal_training_data","pytorch/default/3",kaggle_version=3).validate()
