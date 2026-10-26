import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/sglang"))
import download_model


class DownloadTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux deployment uses symlinks")
    def test_pinned_download_checksum_overlay_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "upstream" / download_model.SUBFOLDER
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"test weight bytes")
            (checkpoint / "config.json").write_text("{}")
            processor = b'{"test": true}'
            def fetch(repo, name, **kwargs):
                self.assertEqual(repo, download_model.PROCESSOR_REPOSITORY)
                self.assertEqual(kwargs["revision"], download_model.PROCESSOR_REVISION)
                target = Path(kwargs["local_dir"]) / name
                target.write_bytes(processor)
                return str(target)
            with patch.object(download_model, "ROOT", root), \
                 patch.object(download_model, "WEIGHT_SHA256", hashlib.sha256(b"test weight bytes").hexdigest()), \
                 patch.object(download_model, "PROCESSOR_HASHES", {"preprocessor_config.json": hashlib.sha256(processor).hexdigest()}), \
                 patch.object(download_model, "snapshot_download") as snapshot, \
                 patch.object(download_model, "hf_hub_download", side_effect=fetch):
                download_model.main()
                download_model.main()
                self.assertEqual(snapshot.call_args.kwargs["revision"], download_model.REVISION)
                self.assertEqual(snapshot.call_args.kwargs["allow_patterns"], [download_model.SUBFOLDER + "/*"])
                self.assertEqual((root / "model-overlay/model.safetensors").resolve(), checkpoint / "model.safetensors")
                self.assertFalse(os.readlink(root / "model-overlay/model.safetensors").startswith("/"))
                (checkpoint / "model.safetensors").write_bytes(b"damaged")
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    download_model.main()


@unittest.skipUnless(sys.platform == "linux", "Linux process management")
class SGLangStopTests(unittest.TestCase):
    def test_stop_only_owned_group_even_if_worker_frozen(self):
        import service_sglang as service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root / "server.json"
            fake_python = root / ".venv-sglang/bin/python"
            sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
            code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"
            child = subprocess.Popen([sys.executable, "-c", code, "sglang.launch_server", str(fake_python)],
                                     cwd=root, start_new_session=True, stdout=subprocess.PIPE, text=True)
            worker = int(child.stdout.readline())
            try:
                os.kill(worker, signal.SIGSTOP)
                pidfile.write_text(json.dumps({"pid": child.pid, "started": service.process_info(child.pid)["started"]}))
                with patch.object(service, "ROOT", root), patch.object(service, "PYTHON", fake_python), \
                     patch.object(service, "PIDFILE", pidfile), patch.object(service, "STOP_TIMEOUT", .2):
                    service.stop()
                    self.assertFalse(service.group_members(child.pid))
                    self.assertFalse(pidfile.exists())
                self.assertIsNone(sentinel.poll())
            finally:
                for process in (child, sentinel):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                child.stdout.close()

    def test_wrong_identity_is_rejected(self):
        import service_sglang as service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], cwd=root, start_new_session=True)
            try:
                wrong = {"pid": child.pid, "started": "wrong-start-time"}
                with patch.object(service, "ROOT", root), self.assertRaises(SystemExit):
                    service.owned_members(wrong)
                self.assertIsNone(child.poll())
            finally:
                child.terminate()
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
