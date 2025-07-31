import os
import signal
import sys
import logging
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


class RobustHTTPServer(ThreadingHTTPServer):
    """HTTP server with improved resource management"""
    daemon_threads = True
    allow_reuse_address = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = 30  # 30-second socket timeout
        
    def server_bind(self):
        """Bind socket with SO_REUSEADDR"""
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


class HealthCheckHandler(SimpleHTTPRequestHandler):
    """Enhanced handler with health check endpoint"""
    
    def log_message(self, fmt, *args):
        """Override to use proper logging"""
        logging.info(f"{self.client_address[0]} - {fmt % args}")


def setup_logging():
    """Configure logging for the server"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def signal_handler(signum, _):
    """Handle shutdown signals gracefully"""
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def run_server():
    """Run the HTTP server with proper error handling and resource management"""
    setup_logging()
    
    # Set up signal handling
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Change to static content directory
        static_dir = os.path.join(os.path.dirname(__file__), "static_content")
        os.chdir(static_dir)

        # Create and configure server
        server_address = ("0.0.0.0", 8080)
        httpd = RobustHTTPServer(server_address, HealthCheckHandler)

        # Run server with proper exception handling
        httpd.serve_forever()
        
    except OSError as e:
        logging.error(f"Failed to start server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Server interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logging.info("Server shutdown complete")


if __name__ == "__main__":
    run_server()
