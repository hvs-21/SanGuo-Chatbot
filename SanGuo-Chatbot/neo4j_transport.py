"""Local HTTP transport, with an optional stdin bridge into WSL loopback."""

from contextlib import closing
from http.client import HTTPConnection
import json
import subprocess
import sys

from config import NEO4J_MAX_RESPONSE_BYTES, NEO4J_TIMEOUT_SECONDS, PROJECT_ROOT


def post_http(host, port, path, headers, body, timeout=NEO4J_TIMEOUT_SECONDS):
    with closing(HTTPConnection(host, port, timeout=timeout)) as client:
        client.request("POST", path, body=body, headers=headers)
        response = client.getresponse()
        return response.status, response.read(NEO4J_MAX_RESPONSE_BYTES + 1)


def post_wsl(distro, port, path, headers, body, timeout=NEO4J_TIMEOUT_SECONDS):
    request = {"port": port, "path": path, "headers": headers, "body": body.decode("utf-8"), "timeout": timeout}
    result = subprocess.run(
        ["wsl.exe", "-d", distro, "--cd", str(PROJECT_ROOT), "--", "python3", "-B", "neo4j_transport.py"],
        input=json.dumps(request, ensure_ascii=False).encode("utf-8"), capture_output=True,
        check=True, timeout=timeout + 8, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    status, _, response = result.stdout.partition(b"\n")
    return int(status), response


def main():
    request = json.load(sys.stdin.buffer)
    status, body = post_http("127.0.0.1", request["port"], request["path"], request["headers"],
                             request["body"].encode("utf-8"), request.get("timeout", NEO4J_TIMEOUT_SECONDS))
    sys.stdout.buffer.write(str(status).encode("ascii") + b"\n" + body)


if __name__ == "__main__":
    main()
