"""Measure the new option against the parent default on identical-vector controls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

evidence_dir = Path(__file__).resolve().parent
repo = evidence_dir.parents[1]
with tempfile.TemporaryDirectory(prefix=".tmp-sandbox-test-", dir=repo) as folder:
    sandbox = Path(folder)
    shutil.copy2(evidence_dir / "minimal_bootstrap.py", sandbox)
    source = '''
import json, math, runpy
import torch
Loss = runpy.run_path(SOURCE)["ContrastiveLoss"]
generator = torch.Generator().manual_seed(1023)
rows = []
for features in (3, 5, 17):
    for index in range(8):
        vector = torch.randn((1, features), generator=generator)
        for compute_dtype in (None, torch.float64):
            x = vector.repeat(2, 1).requires_grad_()
            y = vector.repeat(2, 1).requires_grad_()
            loss = Loss(temperature=1e-8, compute_dtype=compute_dtype)(x, y)
            loss.backward()
            rows.append({"features":features,"index":index,"compute_dtype":str(compute_dtype),
                         "loss":loss.item(),"loss_dtype":str(loss.dtype),"gradient_dtype":str(x.grad.dtype),
                         "max_gradient":max(x.grad.abs().max().item(),y.grad.abs().max().item())})
            assert math.isclose(loss.item(), math.log(3), rel_tol=1e-6)
            assert x.grad.dtype == torch.float32 and y.grad.dtype == torch.float32
            if compute_dtype is not None:
                assert loss.dtype == torch.float64
                assert rows[-1]["max_gradient"] <= 1e-7
summary = {"temperature":1e-8,"cases_per_mode":24,"expected_gradient":0.0,
           "default_max_error":max(r["max_gradient"] for r in rows if r["compute_dtype"] == "None"),
           "float64_max_error":max(r["max_gradient"] for r in rows if r["compute_dtype"] != "None"),
           "torch":torch.__version__,"rows":rows}
assert summary["default_max_error"] >= 0.5
print(json.dumps(summary,indent=2))
'''.replace("SOURCE", repr(str(repo / "monai/losses/contrastive.py")))
    (sandbox / "probe.py").write_text(source, encoding="utf-8")
    env = os.environ.copy()
    env.update({"TEMP":folder,"TMP":folder,"TMPDIR":folder,"XDG_CACHE_HOME":folder,
                "PYTHONDONTWRITEBYTECODE":"1", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
                "OPENBLAS_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1", "CUDA_VISIBLE_DEVICES":"",
                "MONAI_QA_REQUIRED_SITE":sysconfig.get_path("purelib"), "PYTHONPATH":str(repo)})
    result = subprocess.run([sys.executable, "-B", "-S", "minimal_bootstrap.py", "probe.py"], cwd=sandbox,
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=180)
    output = json.loads(result.stdout) if result.returncode == 0 else {"stderr":result.stderr}
output.update({"returncode":result.returncode,"sandbox_removed":not sandbox.exists()})
(evidence_dir / "precision_observation.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in output.items() if k != "rows"},indent=2))
raise SystemExit(result.returncode)
