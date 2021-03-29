from flask import Flask
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename


application = Flask(__name__)

application.secret_key = b'Ovarian'
from application import routes







