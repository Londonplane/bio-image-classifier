import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from model import ImageClassifier

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化图像分类器
classifier = ImageClassifier()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        flash('没有选择文件')
        return redirect(url_for('upload_file'))
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('没有选择文件')
        return redirect(url_for('upload_file'))
    
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 使用分类器处理图片
            try:
                category, confidence = classifier.predict(filepath)
                results.append({
                    'filename': filename,
                    'category': category,
                    'confidence': confidence
                })
            except Exception as e:
                flash(f'处理文件时出错: {filename} - {str(e)}')
        else:
            flash(f'不支持的文件类型: {file.filename}')
    
    if not results:
        flash('没有成功处理任何文件')
        return redirect(url_for('upload_file'))
    
    return render_template('result.html', results=results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 