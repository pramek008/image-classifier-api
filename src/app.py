import logging
import sys
from flask import Flask
from routes import configure_routes
from config import logging as config_logging

app = Flask(__name__)

# Set up logging to both file and stdout
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

configure_routes(app)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')