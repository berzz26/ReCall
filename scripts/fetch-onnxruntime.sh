#!/bin/sh
# Local development helper: download the pinned ONNX Runtime distribution into
# third_party/. The native (in-process) embedding and rerank backends load
# this shared library at API startup. Docker users do not need this; the API
# image downloads and verifies the same archive during the image build.
set -eu
cd "$(dirname "$0")/.."

VERSION="${ORT_VERSION:-1.31.1}"
SHA256="${ORT_SHA256:-c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9}"
DEST="$PWD/third_party/onnxruntime/onnxruntime-linux-x64-${VERSION}"
ARCHIVE="/tmp/onnxruntime-linux-x64-${VERSION}.tgz"

if [ -f "$DEST/lib/libonnxruntime.so" ]; then
  echo "ONNX Runtime already present at $DEST"
  exit 0
fi

curl --fail --location --silent --show-error \
  --output "$ARCHIVE" \
  "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-x64-${VERSION}.tgz"
echo "${SHA256}  ${ARCHIVE}" | sha256sum -c -

mkdir -p "$DEST"
tar -xzf "$ARCHIVE" --strip-components=1 -C "$DEST"
rm -f "$ARCHIVE"
echo "ONNX Runtime ${VERSION} installed at $DEST"
