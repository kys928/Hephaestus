"""Recover only the immutable public #152 baseline; no inference or training."""
from hashlib import sha256
import json
from pathlib import Path
import urllib.error
import urllib.request

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION="12fd25f77366fa6b3b4b768ec3050bf629380bac"
EXPECTED={"model.safetensors":"5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
          "tokenizer.json":"9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"}


def recover(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    snapshot=root/"snapshot"; snapshot.mkdir()
    hashes={}; sizes={}
    for name in ("config.json","generation_config.json","tokenizer_config.json","special_tokens_map.json",
                 "tokenizer.json","model.safetensors","README.md","LICENSE"):
        url=f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        try:
            response=urllib.request.urlopen(url,timeout=180)
        except urllib.error.HTTPError as error:
            if error.code==404 and name in {"LICENSE","special_tokens_map.json","generation_config.json"}:
                continue
            raise
        digest=sha256(); count=0
        with response, (snapshot/name).open("xb") as stream:
            while chunk:=response.read(1024*1024):
                count+=len(chunk)
                if count>600*1024*1024:
                    raise ValueError("public source exceeds fixed per-file bound")
                digest.update(chunk); stream.write(chunk)
        hashes[name]=digest.hexdigest(); sizes[name]=count
        if name in EXPECTED and hashes[name]!=EXPECTED[name]:
            raise ValueError("original public source hash differs: "+name)
    if any(hashes.get(name)!=expected for name,expected in EXPECTED.items()):
        raise ValueError("original base/tokenizer not recovered")
    card=(snapshot/"README.md").read_text()
    if "license: apache-2.0" not in card.lower():
        raise ValueError("pinned model card does not retain expected source license")
    tokenizer=json.loads((snapshot/"tokenizer_config.json").read_text())
    template=tokenizer.get("chat_template")
    if not isinstance(template,str) or not template:
        raise ValueError("original tokenizer has no actual chat template")
    report={"schema":"original-public-model-byte-recovery.v1","model_id":MODEL,"revision":REVISION,
        "file_sha256":hashes,"file_bytes":sizes,"chat_template_sha256":sha256(template.encode()).hexdigest(),
        "model_card_license":"apache-2.0","generation_executed":False,"training_executed":False,
        "external_certification":False,"separate_private_r10_archive_recovered":False}
    (root/"recovery.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return report


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("root",type=Path)
    recover(parser.parse_args().root)
