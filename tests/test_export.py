import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.export import _normalize_tokenizer_config_for_conversion, quantize_to_gguf
from pipeline.logger import PipelineLogger


class ExportHelpersTest(unittest.TestCase):
    def _logger(self, root: Path) -> PipelineLogger:
        return PipelineLogger("export_test", root / "manifests")

    def test_quantize_to_gguf_builds_expected_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported_dir = Path(tmp)
            logger = self._logger(exported_dir)
            export_cfg = {"quantization": {"gguf": ["Q4_K_M", "Q8_0"]}}

            with patch("pipeline.export._resolve_tool_path") as resolve_tool:
                resolve_tool.side_effect = ["/tools/convert_hf_to_gguf.py", "/tools/llama-quantize"]
                with patch("pipeline.export.subprocess.run") as run_cmd:
                    run_cmd.return_value.returncode = 0
                    run_cmd.return_value.stdout = ""
                    run_cmd.return_value.stderr = ""

                    quantize_to_gguf(exported_dir, export_cfg, logger)

            self.assertEqual(run_cmd.call_count, 3)
            commands = [call.args[0] for call in run_cmd.call_args_list]

            self.assertEqual(
                commands[0],
                [
                    sys.executable,
                    "/tools/convert_hf_to_gguf.py",
                    str(exported_dir),
                    "--outfile",
                    str(exported_dir / "model-f16.gguf"),
                    "--outtype",
                    "f16",
                ],
            )
            self.assertEqual(
                commands[1],
                [
                    "/tools/llama-quantize",
                    str(exported_dir / "model-f16.gguf"),
                    str(exported_dir / "model-q4_k_m.gguf"),
                    "Q4_K_M",
                ],
            )
            self.assertEqual(
                commands[2],
                [
                    "/tools/llama-quantize",
                    str(exported_dir / "model-f16.gguf"),
                    str(exported_dir / "model-q8_0.gguf"),
                    "Q8_0",
                ],
            )

    def test_quantize_to_gguf_requires_convert_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported_dir = Path(tmp)
            logger = self._logger(exported_dir)
            export_cfg = {"quantization": {"gguf": ["Q4_K_M"]}}

            with patch("pipeline.export._resolve_tool_path", side_effect=[None]):
                with self.assertRaises(RuntimeError):
                    quantize_to_gguf(exported_dir, export_cfg, logger)

    def test_normalize_tokenizer_config_converts_extra_special_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported_dir = Path(tmp)
            logger = self._logger(exported_dir)
            tokenizer_config = exported_dir / "tokenizer_config.json"
            tokenizer_config.write_text(
                '{"tokenizer_class":"Qwen2Tokenizer","extra_special_tokens":["<a>","<b>"]}',
                encoding="utf-8",
            )

            _normalize_tokenizer_config_for_conversion(exported_dir, logger)

            updated = tokenizer_config.read_text(encoding="utf-8")
            self.assertIn('"additional_special_tokens": [', updated)
            self.assertNotIn('"extra_special_tokens"', updated)


if __name__ == "__main__":
    unittest.main()
