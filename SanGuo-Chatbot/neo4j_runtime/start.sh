#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "$source_dir/../.." && pwd)"
runtime="$workspace/.runtime/neo4j"
cp -- "$source_dir/neo4j.conf" "$runtime/instance/conf/neo4j.conf"
exec java -Xms256m -Xmx512m -Dfile.encoding=UTF-8 -cp "$runtime/lib/*" \
    org.neo4j.server.Neo4jCommunity --home-dir="$runtime/instance" \
    --config-dir="$runtime/instance/conf" --console-mode
