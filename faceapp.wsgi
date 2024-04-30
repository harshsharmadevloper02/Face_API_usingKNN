import sys
import logging
 
sys.path.insert(0, '/var/www/face_app/')
 
# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
 
# Import and run the Flask app
from app import app as application
