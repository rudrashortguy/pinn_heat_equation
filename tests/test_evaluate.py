import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import evaluate


def test_evaluate_missing_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs("checkpoints", exist_ok=True)
        try:
            evaluate()
            assert False, "Should have raised FileNotFoundError"
        except (FileNotFoundError, RuntimeError):
            pass
        finally:
            os.chdir(old_cwd)
