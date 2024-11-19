from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Flask应用初始化
app = Flask(__name__)

# 配置文件夹路径
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
HTML_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '前端页面版本/前端页面正式版01'))  # 上层目录中的HTML文件夹
STATIC_FOLDER = os.path.join(HTML_FOLDER, 'assets')  # 静态资源文件夹
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 创建上传和结果目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 验证文件格式
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 图像变化检测算法
def detect_changes(image1_path, image2_path):
    # 读取两幅图像
    img1 = cv2.imread(image1_path, 0)  # 读取为灰度图
    img2 = cv2.imread(image2_path, 0)

    # 图像预处理（如去噪、归一化）
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    # 计算图像的绝对差异
    diff = cv2.absdiff(img1, img2)

    # 对差异图像进行二值化
    _, thresholded = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # 保存结果图像
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.png')
    cv2.imwrite(result_path, thresholded)
    return result_path

# API路由：上传图像并检测变化
@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查是否有图像上传
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': '请上传两幅图像文件！'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    # 验证文件类型
    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        # 保存图像文件
        file1.save(filepath1)
        file2.save(filepath2)

        # 执行图像变化检测
        result_path = detect_changes(filepath1, filepath2)

        # 返回检测结果
        return jsonify({'result': f'/results/{os.path.basename(result_path)}'}), 200
    else:
        return jsonify({'error': '文件类型不支持！'}), 400

# 静态文件路由：返回检测结果图像
@app.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

# 静态文件路由：返回静态资源文件
@app.route('/assets/<path:filename>', methods=['GET'])
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

# 静态文件路由：返回上层目录的index.html
@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory(HTML_FOLDER, 'index.html')

# 启动服务
if __name__ == '__main__':
    app.run(debug=True)
