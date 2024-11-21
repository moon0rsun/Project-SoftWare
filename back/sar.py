from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask import after_this_request

# Flask应用初始化
app = Flask(__name__)

# 配置文件夹路径
UPLOAD_FOLDER = 'back/uploads' 
RESULT_FOLDER = 'back/results' 
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
    # 读取图像
    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)

    
    if img1.shape != img2.shape:
        raise ValueError("输入图像大小不一致，请检查上传的图像！")

    # 打印图像统计信息
    print("Image1 mean:", np.mean(img1), "stddev:", np.std(img1))
    print("Image2 mean:", np.mean(img2), "stddev:", np.std(img2))

    # 计算差异
    diff = cv2.absdiff(img1, img2)
    print("Difference mean:", np.mean(diff), "max:", np.max(diff), "min:", np.min(diff))

    # 二值化
    _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 保存结果
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.png')
    cv2.imwrite(result_path, thresholded)
    print("Result saved at:", result_path)

    return os.path.abspath(result_path) 

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
        print(result_path)
        
        # 直接返回结果图像
        return send_file(result_path, mimetype='image/png')
    else:
        return jsonify({'error': '文件类型不支持！'}), 400
    
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

