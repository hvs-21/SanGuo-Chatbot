#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "$source_dir/../.." && pwd)"
runtime="$workspace/.runtime/neo4j"
maven_version=3.9.11
archive="apache-maven-${maven_version}-bin.tar.gz"
url="https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/$maven_version/$archive"
mkdir -p -- "$runtime/instance/conf"
cd -- "$runtime"
if [[ ! -x "apache-maven-$maven_version/bin/mvn" ]]; then
    curl -fLsS --retry 2 --max-time 90 "$url" -o "$archive.part"
    curl -fLsS --max-time 20 "$url.sha512" -o "$archive.sha512"
    checksum="$(tr -d '\r\n' < "$archive.sha512")"
    [[ "$checksum" =~ ^[0-9A-Fa-f]{128}$ ]]
    printf '%s  %s\n' "$checksum" "$archive.part" | sha512sum -c -
    mv -- "$archive.part" "$archive"
    tar -xzf "$archive"
fi
"$runtime/apache-maven-$maven_version/bin/mvn" -B -ntp -f "$source_dir/pom.xml" \
    "-Dmaven.repo.local=$runtime/maven-cache" \
    -Daether.connector.connectTimeout=10000 -Daether.connector.requestTimeout=25000 \
    org.apache.maven.plugins:maven-dependency-plugin:3.9.0:copy-dependencies \
    -DincludeScope=runtime "-DoutputDirectory=$runtime/lib"
cp -- "$source_dir/neo4j.conf" "$runtime/instance/conf/neo4j.conf"
printf 'Neo4j runtime ready at %s\n' "$runtime"
