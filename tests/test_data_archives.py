import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("extract_data", ROOT / "scripts/extract_data.py")
EXTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACT)


class DataArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.original_root = EXTRACT.ROOT
        self.addCleanup(setattr, EXTRACT, "ROOT", self.original_root)
        EXTRACT.ROOT = Path(self.temporary.name)
        (EXTRACT.ROOT / "data/archives").mkdir(parents=True)
        self.payload = b"dataset fixture"
        self.expected = hashlib.sha256(self.payload).hexdigest()

    def bundle(self, member="dataset.pkl"):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(member, self.payload)
        contents = output.getvalue()
        midpoint = len(contents) // 2
        parts = []
        for index, block in enumerate((contents[:midpoint], contents[midpoint:]), 1):
            name = "fixture.zip.part%02d" % index
            (EXTRACT.ROOT / "data/archives" / name).write_bytes(block)
            parts.append({"file": name, "size_bytes": len(block),
                          "sha256": hashlib.sha256(block).hexdigest()})
        return {"parts": parts, "archive_sha256": hashlib.sha256(contents).hexdigest()}

    def test_extract_parts_and_repeat(self):
        archives = self.bundle()
        EXTRACT.extract_dataset("yelp", archives, self.expected)
        EXTRACT.extract_dataset("yelp", archives, self.expected)
        self.assertEqual((EXTRACT.ROOT / "data/canonical/yelp/dataset.pkl").read_bytes(), self.payload)

    def test_corrupt_part_rejected(self):
        archives = self.bundle()
        (EXTRACT.ROOT / "data/archives" / archives["parts"][0]["file"]).write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            EXTRACT.extract_dataset("yelp", archives, self.expected)
        self.assertFalse((EXTRACT.ROOT / "data/canonical/yelp/dataset.pkl").exists())

    def test_unexpected_member_rejected(self):
        archives = self.bundle("../dataset.pkl")
        with self.assertRaises(ValueError):
            EXTRACT.extract_dataset("yelp", archives, self.expected)

    def test_existing_mismatch_not_overwritten(self):
        target = EXTRACT.ROOT / "data/canonical/yelp/dataset.pkl"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing")
        with self.assertRaises(ValueError):
            EXTRACT.extract_dataset("yelp", self.bundle(), self.expected)
        self.assertEqual(target.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
