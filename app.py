from flask import Flask, render_template, request, redirect, url_for, jsonify
from get_images import get_images, get_path, get_directory
from get_prediction import get_prediction, get_caps_from_upload
from generate_html import generate_html
from model import EncoderDecoder
import torch
from base64 import b64encode
from get_loader import dataset
from gtts import gTTS



app = Flask(__name__)

def textTospeech(text):
    audio = gTTS(text)
    audio_res = b""
    for i in audio.stream():
        audio_res = audio_res + i
    return("data:audio/mpeg;base64," + b64encode(audio_res).decode('utf-8'))


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

# mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the model
final_model = EncoderDecoder(
    embed_size=300,
    vocab_size=len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

PATH = './models/model.pt'
final_model.load_state_dict(torch.load(PATH, map_location=device))

# use the pre-trained model

final_model.eval()

# define the function to get the images from the url and predicted the class
def get_image_class(dir):
    # get images from the URL and store it in a given path
    get_images(dir)
    # predict the image class of the images with provided directory
    path = get_path(dir)
    images_with_tags = get_prediction(final_model, path)
    # generate html file to render once we predict the classes
    generate_html(images_with_tags)
    
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def request_operations(request):
    f = request.files['file']
    if not (f and allowed_file(f.filename)):
        return jsonify({"error": 1001, "msg": "Only png、PNG、jpg、JPG、bmp"})
    cap, img_base64 = get_caps_from_upload(f, final_model)
    text = cap[0].upper() + cap[1:]
    # audio = text2speech(text)
    audio_base64 = textTospeech(text)
    # audio_base64 = "data:audio/mpeg;base64," + b64encode(audio).decode('utf-8')
    return text, img_base64, audio_base64


@app.route('/', methods=['POST', 'GET'])
def upload():
    global text
    global img_base64
    if request.method == 'POST':
        text, img_base64, audio_base64 = request_operations(request)
        return render_template('preview_ok.html', text=text, uri=img_base64, audio=audio_base64)
    return render_template('preview.html')


@app.route('/show', methods=['POST', 'GET'])
def show():
    global text
    global img_base64
    if request.method == 'POST':
        text, img_base64, audio_base64 = request_operations(request)
    return render_template('preview_ok.html', text=text, uri=img_base64, audio=audio_base64)


@app.route('/url')
def home():
    return render_template('home.html')

@app.route('/url', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        # if search button hit, call the function get_image_class
        get_image_class(user)
        #render the image_class.html
        return redirect(url_for('success', name=get_directory(user)))


@app.route('/success/<name>')
def success(name):
    return render_template('image_class.html')


if __name__ == '__main__' :
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=8080)
