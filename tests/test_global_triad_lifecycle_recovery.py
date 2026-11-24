"""Regression coverage for Global triad lifecycle recovery, no paid compute."""
from pathlib import Path
import base64,sys,subprocess
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from ministral_global_cleanup import terminate_campaign_pods
from launch_approved_ministral_triad_cuda import command_for,GLOBAL_ID
from hephaestus.providers.runpod.execution import RunPodApiError

TAG="45372244159"
NAME="hephaestus-ministral-triad-"+TAG+"-global-1"

class Provider:
    def __init__(self,rows):
        self.rows=list(rows)
        self.deleted=[]
    def list_pods(self):return list(self.rows)
    def get_pod(self,id):return next(x for x in self.rows if x["id"]==id)
    def _request(self,verb,path):
        assert verb=="DELETE"
        id=path.split("/")[-1]
        self.deleted.append(id)
        self.rows=[x for x in self.rows if x["id"]!=id]
        return 204,None
    def delete_pod(self,id):
        raise AssertionError("SQLite-dependent legacy deletion must never run")

def test_direct_cleanup_preserves_unrelated_pod():
    p=Provider([{"id":"own123","name":NAME},{"id":"other123","name":"compulsory_white_centipede"}])
    terminate_campaign_pods(p,TAG,sleep=lambda _:None)
    assert p.deleted==["own123"]
    assert p.rows==[{"id":"other123","name":"compulsory_white_centipede"}]

def test_cleanup_refuses_unrecognized_campaign_resource():
    p=Provider([{"id":"own123","name":NAME+"-other"}])
    with pytest.raises(PermissionError):
        terminate_campaign_pods(p,TAG,sleep=lambda _:None)
    assert not p.deleted

def test_bootstrap_compiles_and_terminal_container_holds():
    cmd,env=command_for("a"*40,"sha256:"+"b"*64,"sha256:"+"c"*64,TAG,{})
    assert cmd[:2]==["bash","-lc"]
    assert cmd[2].endswith("exec sleep infinity")
    encoded=cmd[2].split("base64.b64decode('",1)[1].split("'",1)[0]
    code=base64.b64decode(encoded).decode()
    compile(code,"<pod-bootstrap>","exec")
    assert "TRIAD_RESTART_REFUSED_EXISTING_WORKSPACE" in code
    assert "sys.exit(73)" in code
    assert env["RUNPOD_GLOBAL_VOLUME_ID"]==GLOBAL_ID
    assert subprocess.run(["bash","-n","-c",cmd[2]]).returncode==0
