# Project-SoftWare
这是一个用于进行中国海洋大学软件工程原理与实践的课程作业仓库。

## 项目简介：

## 项目架构：
该项目是一个基于Flask框架的Web应用程序，用于实现图像变化检测功能。项目包含前端页面和后端服务两部分。

### 前端页面
前端页面位于/SeaSealji/Project-SoftWare/前端页面版本/前端页面正式版01目录下，主要包含以下文件：

- index.html：主页面，包含图像上传和显示结果的区域。
- assets/css/：包含CSS样式文件。
- assets/js/：包含JavaScript文件，用于处理页面交互和调用后端API。
### 后端服务
后端服务位于/SeaSealji/Project-SoftWare/back目录下，主要包含以下文件：

- sar.py：Flask应用的主文件，定义了API路由和主要功能。
- model.py：包含图像变化检测算法的实现。
- preclassify.py：包含图像预分类算法的实现。

## 项目环境
python环境，Windows或者Linux系统
安装flask
~~~bash
pip install flask numpy skimage torch torchvision torch.nn torch.optim torch.nn.functional cv2 os matplotlib sklearn
~~~
## 项目部署：
1. 下载源码
2. 运行back文件夹中的sar.py
3. 打开浏览器的对应端口
## Collaborators

| ![刘海涵](https://github.com/SeaSealji.png?size=1000) | ![吕茂宁](https://github.com/tianshuking.png?size=10) | ![刘奕鹏](https://github.com/moon0rsun.png?size=1000) |
| :---: | :---: | :---: |
| **刘海涵** - *Project Lead* - [GitHub](https://github.com/SeaSealji) | **吕茂宁** - *Group Membership* - [GitHub](https://github.com/tianshuking) | **刘奕鹏** - *Group Membership* - [GitHub](https://github.com/moon0rsun) |
