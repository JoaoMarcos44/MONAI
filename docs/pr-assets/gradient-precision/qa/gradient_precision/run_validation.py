"""Validate ContrastiveLoss in a disposable sandbox and retain final evidence only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("red", "green"), required=True)
    options = parser.parse_args()
    evidence_dir = Path(__file__).resolve().parent
    repo = evidence_dir.parents[1]
    runs = []
    with tempfile.TemporaryDirectory(prefix=".tmp-sandbox-test-", dir=repo) as folder:
        sandbox = Path(folder)
        for name in ("deps", "temp", "cache"):
            (sandbox / name).mkdir()
        env = os.environ.copy()
        env.update({
            "TEMP": str(sandbox / "temp"), "TMP": str(sandbox / "temp"), "TMPDIR": str(sandbox / "temp"),
            "XDG_CACHE_HOME": str(sandbox / "cache"), "MPLCONFIGDIR": str(sandbox / "cache"),
            "BLACK_CACHE_DIR": str(sandbox / "cache"), "TORCH_HOME": str(sandbox / "cache"),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "", "MONAI_QA_REQUIRED_SITE": sysconfig.get_path("purelib"),
            "PYTHONPATH": os.pathsep.join([str(sandbox / "deps"), str(repo)]),
        })
        shutil.copy2(evidence_dir / "minimal_bootstrap.py", sandbox)
        shutil.copy2(repo / "tests/losses/test_contrastive_loss.py", sandbox)
        (sandbox / ".coveragerc").write_text("[run]\ncore = pytrace\n", encoding="utf-8")

        def run(label, args, bootstrap=True):
            command = [sys.executable, "-B"]
            if bootstrap:
                command += ["-S", "minimal_bootstrap.py"]
            completed = subprocess.run(command + args, cwd=sandbox, env=env, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", timeout=240)
            result = {"label": label, "exit_code": completed.returncode,
                      "stdout": completed.stdout, "stderr": completed.stderr}
            runs.append(result)
            print(f"{label}: exit {completed.returncode}", flush=True)
            return result

        packages = ["parameterized==0.9.0"]
        if options.phase == "green":
            packages += ["ruff==0.16.10", "black==26.10.0", "isort==9.0.2", "coverage"]
        setup = run("dependencies", ["-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                                     "--target", str(sandbox / "deps"), *packages], bootstrap=False)
        if setup["exit_code"]:
            raise RuntimeError(setup["stderr"])
        test_args = ["-m", "unittest", "test_contrastive_loss", "-v"]
        if options.phase == "green":
            test_args = ["-m", "coverage", "run", "--rcfile=.coveragerc",
                         f"--include={repo.as_posix()}/monai/losses/contrastive.py", *test_args]
        tests = run("tests", test_args)
        if options.phase == "green":
            run("coverage", ["-m", "coverage", "report", "--rcfile=.coveragerc", "-m", "--fail-under=80"])
        run("observation", ["-c", "import torch, numpy; from monai.losses import ContrastiveLoss; "
            "print('torch',torch.__version__,'numpy',numpy.__version__); "
            "x=torch.ones(2,4,requires_grad=True); loss=ContrastiveLoss(temperature=0.01)(x,x); "
            "print('loss',loss.item()); loss.backward(); print('finite gradients',torch.isfinite(x.grad).all().item())"])
        if options.phase == "green":
            files = [str(repo / "monai/losses/contrastive.py"), str(repo / "tests/losses/test_contrastive_loss.py")]
            run("ruff", ["-m", "ruff", "check", "--no-cache", *files])
            run("black", ["-m", "black", "--check", "--diff", "--workers", "1", *files])
            run("isort", ["-m", "isort", "--check-only", "--diff", *files])
    confirmed = (tests["exit_code"] == 1 and "unexpected keyword argument 'compute_dtype'" in tests["stderr"]
                 and "FAILED (errors=49)" in tests["stderr"]
                 if options.phase == "red" else all(r["exit_code"] == 0 for r in runs))
    evidence = {"phase": options.phase, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "python": sys.version, "sandbox_removed": not sandbox.exists(), "confirmed": confirmed, "runs": runs}
    evidence["source_sha256"] = hashlib.sha256((repo / "monai/losses/contrastive.py").read_bytes()).hexdigest()
    evidence["tests_sha256"] = hashlib.sha256((repo / "tests/losses/test_contrastive_loss.py").read_bytes()).hexdigest()
    (evidence_dir / f"{options.phase}_results.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    for result in runs:
        if result["label"] != "dependencies":
            (evidence_dir / f"{options.phase}_{result['label']}.txt").write_text(
                f"Exit code: {result['exit_code']}\n{result['stdout']}{result['stderr']}", encoding="utf-8")
    print(f"Confirmed: {confirmed}; sandbox removed: {not sandbox.exists()}")
    return 0 if confirmed and not sandbox.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
