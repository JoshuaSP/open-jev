import hashlib
import json
from pathlib import Path
import unittest
from inference import prepare

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_request_compiles_literal_json(self):
        request = json.loads((ROOT / "examples/routing.json").read_text())
        prompt, alternatives = prepare(request)
        self.assertEqual(len(alternatives), 6)
        self.assertIn("<user_text>", prompt)
        self.assertLess(prompt.index("</user_text>"), prompt.index("<instructions>"))
        self.assertEqual(
            {json.loads(x)["department"] for x in alternatives},
            {"billing", "technical", "sales"},
        )

    def test_rejects_duplicate_and_exploding_languages(self):
        with self.assertRaises(ValueError):
            prepare(
                {
                    "document": "x",
                    "questions": {"x": {"question": "x", "options": [True, True]}},
                }
            )
        with self.assertRaises(ValueError):
            prepare(
                {
                    "document": "x",
                    "questions": {
                        str(i): {"question": "x", "options": [False, True]}
                        for i in range(13)
                    },
                }
            )

    def test_large_schema_compiles_without_cartesian_enumeration(self):
        request = {
            "document": "x",
            "questions": {
                str(i): {"question": "x", "options": [False, True]} for i in range(30)
            },
        }
        _, fields = prepare(request, enumerate_candidates=False)
        self.assertEqual(len(fields), 30)

    def test_public_source_checksums(self):
        manifest = json.loads(
            (ROOT / "typesafe-public-evals/manifest.json").read_text()
        )
        for source in manifest["workflows"]:
            self.assertEqual(
                hashlib.sha256((ROOT / source["file"]).read_bytes()).hexdigest(),
                source["sha256"],
            )

    def test_public_receipt_scores_and_token_accounting(self):
        d = json.loads(
            (ROOT / "results/public_evals_20260916T193219Z.json").read_text()
        )
        rows = d["predictions"]
        scored = [
            r
            for r in rows
            if r["reference_label"] is not None and r["jev_label"] is not None
        ]
        self.assertEqual(len(rows), 408)
        self.assertEqual(len(scored), 337)
        canon = lambda x: str(x).lower() if isinstance(x, bool) else str(x)
        self.assertEqual(
            sum(canon(r["value"]) == r["reference_label"] for r in scored), 298
        )
        self.assertEqual(
            sum(r["jev_label"] == r["reference_label"] for r in scored), 306
        )
        self.assertEqual(sum(r["input_tokens"] for r in rows), 2214128)
        self.assertEqual(sum(b["input_tokens"] for b in d["records"]), 2214128)
        self.assertEqual(
            [key for b in d["records"] for key in b["question_ids"]],
            [r["id"] for r in rows],
        )
        self.assertTrue(all(r["valid"] for r in rows))

    def test_parallel_receipts_preserve_questions_and_node_outputs(self):
        original = json.loads(
            (ROOT / "results/public_evals_20260916T193219Z.json").read_text()
        )
        expected = {r["id"] for r in original["predictions"]}
        for path in (ROOT / "results").glob("parallel_evals_*.json"):
            result = json.loads(path.read_text())
            rows = result["predictions"]
            self.assertEqual(len(rows), 408)
            self.assertEqual({r["id"] for r in rows}, expected)
            self.assertEqual(sum(len(o) for o in result["node_outputs"].values()), 408)
            for row in rows:
                node, key = row["id"].rsplit("/", 1)
                self.assertEqual(row["value"], result["node_outputs"][node][key])
            self.assertEqual(
                sum(r["input_tokens"] for r in result["outputs"]),
                result["inference_totals"]["input_tokens"],
            )
            scored = [
                r
                for r in rows
                if r["reference_label"] is not None and r["jev_label"] is not None
            ]
            self.assertEqual(len(scored), 337)
            canon = lambda x: str(x).lower() if isinstance(x, bool) else str(x)
            self.assertEqual(
                sum(canon(r["value"]) == r["reference_label"] for r in scored),
                result["summary"]["ALL"]["ours_correct"],
            )

    def test_every_receipt_accuracy(self):
        d = json.loads((ROOT / "results/every_lab_20260916T145029Z.json").read_text())
        for steps, expected in [(1, 138), (4, 139)]:
            rows = [r for r in d["records"] if r["steps"] == steps]
            code = [r for r in rows if r["experiment"] == "code-rag"]
            support = [r for r in rows if r["experiment"] == "customer-voice"]
            self.assertEqual(
                sum(v == r["labels"][k] for r in code for k, v in r["values"].items()),
                48,
            )
            self.assertEqual(
                sum(
                    v == (r["reference_p_true"][k] >= 0.5)
                    for r in support
                    for k, v in r["values"].items()
                ),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
