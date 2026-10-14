"""Inspect the manifest of the documented public MSIX without installing it."""
import io
import base64
import json
import pathlib
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

URL = "https://persistent.oaistatic.com/codex-app-prod/ChatGPT-x64.msix"
TRANSPORT = pathlib.Path(__file__).with_name("read_official_msix_range.ps1")


def public_metadata_request(*arguments):
    # Use Windows' normal HTTPS certificate validation; do not disable TLS checks.
    command = [shutil.which("pwsh") or "powershell.exe", "-NoProfile", "-File", str(TRANSPORT), *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


class PublicRangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        self.transferred = 0
        metadata = public_metadata_request()
        self.length = int(metadata["length"])
        self.modified = metadata["modified"]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.length}[whence]
        position = base + offset
        if position < 0:
            raise ValueError("Negative position")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        if size <= 0:
            return b""
        if size > 8 * 1024 * 1024 or self.transferred + size > 16 * 1024 * 1024:
            raise RuntimeError("Metadata-only transfer limit exceeded")
        start = self.position
        end = start + size - 1
        data = base64.b64decode(public_metadata_request("-Start", start, "-Length", size, "-ArchiveLength", self.length)["body"], validate=True)
        if len(data) != size:
            raise RuntimeError("Incomplete or oversized range response")
        self.transferred += len(data)
        self.position += len(data)
        return data


reader = PublicRangeReader(URL)
with zipfile.ZipFile(reader) as package:
    manifest = ET.fromstring(package.read("AppxManifest.xml"))
identity = next(element for element in manifest if element.tag.rsplit("}", 1)[-1] == "Identity")
print(json.dumps({"url": URL, "archive_bytes": reader.length, "last_modified": reader.modified,
                  "identity": identity.attrib, "metadata_bytes_read": reader.transferred,
                  "installed": False, "signature_validated": False}, indent=2))
