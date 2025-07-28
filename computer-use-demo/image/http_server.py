import os
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler


# class HTTPServerV6(HTTPServer):
#     address_family = socket.AF_INET6


def run_server():
    os.chdir(os.path.dirname(__file__) + "/static_content")
    server_address = ("0.0.0.0", 8080)
    httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)
    print("Starting HTTP server on port 8080...")  # noqa: T201
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
