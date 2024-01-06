import io
import os
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, jsonify, request, render_template
from PIL import Image

app = Flask(__name__)

# Modelling Task
model = models.resnet18()
num_inftr = model.fc.in_features
model.fc = nn.Linear(num_inftr, 4)
model.load_state_dict(torch.load('./fix_resnet18.pth'))
model.eval()

class_names = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']

def transform_image(image_bytes):
	my_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = model.forward(tensor)
	_, prediction = torch.max(outputs, 1)
	return class_names[prediction]

diseases = {
	"Healthy": "Kamu tidak perlu khawatir karena tanamanmu sehat, terus pertahankan yaa!",
	"Apple Scab": "Kudis apel mengganggu bagian tanaman yang terdapat di atas tanah. Gejala awal terdapat pada daun, \
            bunga, dan buah. Bercak berkembang pada daun hingga daun menjadi berubah bentuk. Tepi pada daun umumnya \
            tampak melepuh dan bersisik dengan batas tepi antara jaringan daun yang sehat dan sakit terlihat jelas. \
            Beberapa buah terinfeksi rusak hingga jatuh dari rantingnya sebelum buah matang.",
	"Cedar Apple Rust": "Cedar apple rust, atau CAR, adalah penyakit jamur aneh yang mempengaruhi pohon apel dan cedar merah. Penyakit ini dapat dengan cepat merusak pohon apel dan \
                menyebabkan noda pada buah.Pertama kali muncul di dedaunan sebagai bintik-bintik kuning kehijauan kecil yang secara bertahap membesar, menjadi oranye-kuning menjadi berwarna karat \
                dengan pita merah. Bagian bawah daun mulai membentuk lesi yang menghasilkan spora ",
	"Black Rot": "Penyebaran penyakit busuk hitam  disebabkan jamur Gloeosporium sp dan bisa melalui udara, percikan air dan alat pertanian yang terkontaminasi. \
                 Faktor kelembaban dan kurangnya intensitas matahari yang masuk ke tanaman berakibat jamur cepat untuk berkembang terlebih disaat musim penghujan seperti ini.Seiring waktu,\
				 bintik-bintik melebar dan daun-daun yang terinfeksi berat jatuh dari pohon. cabang yang mengalami penyakit busuk hitam mungkin dapat menginfeksi bagian tumbuhan lain."
}

# Treat the web process
@app.route('/deteksi', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		
		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files.get('file')
		if not file:
			return
		img_bytes = file.read()
		prediction_name = get_prediction(img_bytes)
		return render_template('result.html', name=prediction_name.lower(), description=diseases[prediction_name])

	return render_template('index.html')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/about')
def about():
	return render_template('about.html')

if __name__ == '__main__':
	app.run(debug=True)