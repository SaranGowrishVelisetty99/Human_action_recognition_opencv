from flask import Flask, render_template, request
import os
from detection_and_marking import process_video
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static\output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', output_video=False)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No video file uploaded', 400

    file = request.files['video']
    input_path = os.path.join(UPLOAD_FOLDER, 'input.mp4')
    output_path = os.path.join(OUTPUT_FOLDER, 'output.mp4')

    file.save(input_path)
    process_video(input_path, output_path)

    # Cache-busting query string
    cache_buster = int(time.time())

    return render_template(
        'index.html',
        output_video=True,
        cache_buster=cache_buster
    )

if __name__ == '__main__':
    app.run(debug=True)
