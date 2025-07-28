import os
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


# class HTTPServerV6(HTTPServer):
#     address_family = socket.AF_INET6


def run_server():
    os.chdir(os.path.dirname(__file__) + "/static_content")
    server_address = ("0.0.0.0", 8080)
    httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)
    print("Starting HTTP server on port 8080...")  # noqa: T201
    httpd.serve_forever()


if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True  # Allows the main program to exit even if the thread is running
    server_thread.start()
