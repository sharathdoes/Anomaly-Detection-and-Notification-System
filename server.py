import http.server
import socketserver
import os

# Set the port number for the server
PORT = 8000

# Directory where your anomaly frames are saved
ANOMALY_FRAMES_DIR = "anomaly_frames"

# Define the handler to serve files
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ANOMALY_FRAMES_DIR, **kwargs)
    
    def list_files(self):
        # List image files in the directory
        files = os.listdir(ANOMALY_FRAMES_DIR)
        image_files = [f for f in files if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))]
        return image_files

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # HTML and CSS content
            html_content = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Anomaly Frames</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f0f0f0;
                        margin: 0;
                        padding: 20px;
                    }
                    h1 {
                        color: #333;
                    }
                    .frame {
                        border: 2px solid #ccc;
                        padding: 10px;
                        margin: 10px;
                        background-color: #fff;
                        display: inline-block;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                    }
                </style>
            </head>
            <body>
                <h1>Anomaly Frames</h1>
                <div id="frames-container">
            """
            
            # Add image frames to the HTML
            image_files = self.list_files()
            for image in image_files:
                html_content += f"""
                <div class="frame">
                    <img src="{image}" alt="{image}">
                </div>
                """
            
            # Close the HTML content
            html_content += """
                </div>
            </body>
            </html>
            """
            
            self.wfile.write(html_content.encode('utf-8'))
        else:
            super().do_GET()

# Create a TCP server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving anomaly frames at port {PORT}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Server stopped.")
