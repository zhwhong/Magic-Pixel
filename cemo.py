import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template,jsonify
from flask.ext.uploads import (UploadSet, configure_uploads, IMAGES,
                              UploadNotAllowed)
from werkzeug import secure_filename
from magic import deal_image

BASE_URL= "http://localhost:5000/"

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        url = photos.url(filename)
        return render_template('main.html', image=url)
    return render_template('main.html',image= url_for('static', filename='img/lanshou.jpg') )

@app.route('/photo/<filename>')
def show(filename):
    url = photos.url(filename)
    urlout = photos.url('zjm.jpg')
    return render_template('main.html', image=url, imageout=urlout)


@app.route('/magic', methods=['GET', 'POST'])
def magic():
        img = request.args.get('img')
        a = "uploads/" + img.split('/')[-2]
        x1 = request.args.get('x1')
        y1 = request.args.get('y1')
        x2 = request.args.get('x2')
        y2 = request.args.get('y2')
        size = request.args.get('size')

        print "/magic:size- ==",size
        outpath = deal_image(a, x1, y1, x2, y2, size)
        print img, x1, y1, x2, y2, outpath
        outpath = photos.url(outpath)
        return jsonify(result= outpath)


if __name__ == '__main__':
    app.run(debug=True) 
