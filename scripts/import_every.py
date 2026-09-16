"""Import a user-supplied Every lab ZIP without overwriting existing data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('zip', type=Path)
args = parser.parse_args()
if (root/'typesafe-lab').exists() or (root/'typesafe-lab-source.zip').exists():
    raise SystemExit('Bundle already exists; refusing to overwrite it.')
with tempfile.TemporaryDirectory() as tmp:
    target = Path(tmp)
    with zipfile.ZipFile(args.zip) as archive:
        for entry in archive.infolist():
            dest = (target/entry.filename).resolve()
            if not dest.is_relative_to(target.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe ZIP entry')
        archive.extractall(target)
    matches = list(target.rglob('experiments/RUN_MANIFEST.json'))
    if len(matches) != 1:
        raise ValueError('Expected exactly one Every lab manifest')
    source = matches[0].parent.parent
    manifest = json.loads(matches[0].read_text())
    for name, expected in manifest['artifact_sha256'].items():
        artifact = (source/name).resolve()
        if not artifact.is_relative_to(source.resolve()):
            raise ValueError('Unsafe manifest path')
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Hash mismatch: {name}')
    shutil.copytree(source, root/'typesafe-lab')
    shutil.copyfile(args.zip, root/'typesafe-lab-source.zip')
print('Imported Every bundle and verified all manifest hashes.')
