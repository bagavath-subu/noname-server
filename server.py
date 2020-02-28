from flask import Flask, request, jsonify
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS, cross_origin
from pathlib import Path
from IPython.display import Audio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils import read_file, transform_path
import fastai
import torch

app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
# learn = load_learner(path='./models', file='trained_model.pkl')
# classes = learn.data.classes


def log_mel_spec_tfm(fname, src_path, dst_path):
    x, sample_rate = read_file(fname, src_path)

    n_fft = 1024
    hop_length = 256
    n_mels = 40
    fmin = 20
    fmax = sample_rate / 2

    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    n_mels=n_mels, power=2.0,
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    dst_fname = dst_path / (fname[:-4] + '.png')
    plt.imsave(dst_fname, mel_spec_db)


def predict_single():
    learn = load_learner('./models', 'staged.pkl')
    log_mel_spec_tfm('output.wav', 'out', Path('./out'))
    img = plt.imread('out/output.png')
    plt.imshow(img, origin='lower')
    prediction = learn.predict(open_image('./out/output.png'))
    probs_list = prediction[2].numpy()

    print(prediction[0])
    return {
        'res': str(prediction[0]),
        'result': str(probs_list)
    }

# route for prediction


@app.route('/predict', methods=['POST'])
def predict():
    print(request.data)
    f = open('./out/output.wav', 'wb')
    f.write(request.data)
    f.close()
    return jsonify(predict_single())


@app.route('/check', methods=['GET'])
def check():
    return "Working"


if __name__ == '__main__':
    app.run()
