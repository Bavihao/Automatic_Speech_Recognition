from flask import Flask, render_template, request, jsonify, send_file
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import sounddevice as sd
from werkzeug.utils import secure_filename
import soundfile as sf
from pydub import AudioSegment
import msvcrt
from datetime import datetime
import nltk

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'ogg', 'mp3', 'm4a'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = r"automatic speech recognition processing model"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, local_files_only=False,use_auth_token=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def map_to_array(input_string):
    try:
        sound = AudioSegment.from_file(input_string)
        temp_wav_path = "temp.wav"
        sound.export(temp_wav_path, format="wav")
        speech, _ = sf.read(temp_wav_path)

        if len(speech.shape) > 1:
            speech = speech[:, 0]
        return {"speech": speech}
    except Exception as e:
        print(f"Lỗi khi đọc file âm thanh: {e}")
        return None
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def process_audio_input(file_path):
    try:
        temp_wav_path = "temp.wav"
        sound = AudioSegment.from_file(file_path)
        sound.export(temp_wav_path, format="wav")

        time_start = datetime.now()
        result = pipe(temp_wav_path, generate_kwargs={"language": "vietnamese", "task": "translate"})
        time_end = datetime.now()
        time_run = time_end - time_start
        txt_file_path = os.path.join("files", "txt", "result.txt")
        with open(txt_file_path, "a", encoding="utf-8") as txt_file:
            txt_file.write("=" * 25 + "\n")
            txt_file.write(result["text"] + "\n")
        return {'text': result["text"], 'time_run': str(time_run)}

    except Exception as e:
        return {'error': f'Lỗi xử lý âm thanh: {str(e)}'}

    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/microauto/page')
def microauto():
    return render_template('index.html')

@app.route('/samsum/page')
def samsum():
    return render_template('index2.html')

@app.route('/process_micro', methods=['POST'])
def process_micro():
    try:
        file = request.files['audio_blob']
        file_path = 'uploads/continuous_audio.ogg'
        file.save(file_path)
        result = pipe(file_path, generate_kwargs={"language": "vietnamese", "task": "translate"})
        txt_file_path = os.path.join("files", "txt", "micro_result.txt")
        with open(txt_file_path, "a", encoding="utf-8") as txt_file:
            txt_file.write(result["text"] + "\n")

        return jsonify(result["text"])

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý âm thanh: {str(e)}'}), 500

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        file = request.files['audio_blob']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = process_audio_input(file_path)
            return jsonify(result)
        return jsonify({'error': 'File âm thanh không hợp lệ.'}), 400
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý âm thanh: {str(e)}'}), 500
    
@app.route('/download')
def download_file():
    file_path = 'files/txt/micro_result.txt'
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, port=2468, host='0.0.0.0', ssl_context=('pem/cert.pem', 'pem/key.pem'))
