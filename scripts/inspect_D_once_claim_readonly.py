#!/usr/bin/env python3
"""Read-only check for historical S3 once-claim overwritten between GPU variants."""
import json
import run_interface_repair_v1 as s3
from evaluator_contrast_inference_protocol import PREFIX
ID="heph-D-contrasts-BLACKWELL-EURIS2-20261023-SECOND-USER-AUTH"
KEY=f"{PREFIX}/approvals/{ID}/once.json"
def main():
 c=s3.s3_client(); b=s3.bucket()
 h=c.head_object(Bucket=b,Key=KEY)
 o=c.get_object(Bucket=b,Key=KEY)
 try: j=json.loads(o["Body"].read())
 finally:o["Body"].close()
 out={"exists":True,"last_modified":str(h.get("LastModified")),
     "run_id":j.get("runid"),"gpu_type":j.get("gpu_type_id"),
     "historical_once_claim_name":ID,
     "read_only":True}
 print("D_HISTORICAL_ONCE_CLAIM_S3 "+json.dumps(out,sort_keys=True),flush=True)
if __name__=="__main__":main()
