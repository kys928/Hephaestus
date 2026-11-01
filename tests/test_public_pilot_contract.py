"""Public-only partition and raw termination measurements; no final batch."""
from collections import Counter
import json
from types import SimpleNamespace
import pytest

from hephaestus.schemas.public_pilot import PublicPilotPlan, PublicGenerationContract
from hephaestus.evaluation.public_pilot import sample_public_pilot, public_surfaces, paired_public_summary
from hephaestus.evaluation.prospective_design import arithmetic_capacity
from hephaestus.data.generated_factory import generate_original_splits
from hephaestus.schemas.generated_data import GeneratedDataSpec


def test_nonconfirmatory_pilot_is_development_partition_excludes_original_exposures_and_clusters_templates():
    plan=PublicPilotPlan()
    original,_=generate_original_splits(GeneratedDataSpec(20261028))
    excluded={r['sample_id'] for split in original.values() for r in split}
    rows=sample_public_pilot(plan,excluded)
    assert len(rows)==64 and len({r['sample_id'] for r in rows})==64
    assert Counter(r['family'] for r in rows)==Counter(dict.fromkeys(('arithmetic','json','symbolic','reasoning'),16))
    assert all(int(r['sample_id'][7:],16)%10==8 and r['sample_id'] not in excluded for r in rows)
    assert rows==sample_public_pilot(plan,excluded)
    surfaces=public_surfaces(rows,plan)
    assert len(surfaces)==96 and len({r['surface_id'] for r in surfaces})==96
    assert len({r['sample_id'] for r in surfaces})==64  # variants are NOT new sampling units
    assert all('nonconfirmatory' in r['purpose'] for r in rows)
    assert arithmetic_capacity(20)['evaluation']==79
    assert arithmetic_capacity(100)['evaluation']==2021


@pytest.mark.parametrize('change', [{'max_new_tokens':16}, {'eos_token_id':2.}, {'purpose':'confirmatory'}])
def test_decoding_change_requires_prospective_contract_identity(change):
    with pytest.raises(ValueError):PublicGenerationContract(**change)


def test_no_eos_is_incorrect_even_when_typed_json_is_exact(tmp_path):
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from hephaestus.development.prospective_rehearsal import create_tiny_model
    from hephaestus.evaluation.public_pilot import generate_public_sample
    tokenizer,_=create_tiny_model(tmp_path/'fixture-model')
    # Explicit SOFTWARE fixture; this is not a pretrained Smol result.
    answer=tokenizer.encode('{"result":3}',add_special_tokens=False)
    class Model:
        config=SimpleNamespace(max_position_embeddings=1024)
        def generate(self,*,input_ids,attention_mask,generation_config):
            assert generation_config.forced_eos_token_id is None
            assert generation_config.eos_token_id==2 and generation_config.do_sample is False
            return torch.cat((input_ids,torch.tensor([answer],dtype=torch.long)),dim=1)
        def __call__(self,**kwargs):return SimpleNamespace(loss=torch.tensor(1.))
    row={'prompt':'Compute 1 plus 2. Return JSON with key result.','target':'{"result":3}',
         'family':'arithmetic','sample_id':'public-software-fixture'}
    result=generate_public_sample(Model(),tokenizer,row,PublicGenerationContract())
    assert result['typed_exact'] and result['valid_json']
    assert not result['eos'] and not result['terminated_exact']
    assert result['answer_text']=='{"result":3}'  # no injected EOS or repaired text


def test_pairing_cannot_compare_different_surfaces():
    with pytest.raises(ValueError,match='identities'):
        paired_public_summary([{'surface_id':'one'}],[{'surface_id':'two'}])
