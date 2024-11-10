function generateResults() {
    const image1 = document.getElementById('image1').files[0];
    const image2 = document.getElementById('image2').files[0];

    if (!image1 || !image2) {
        alert('请上传两幅图像');
        return;
    }

    // 模拟图像处理
    document.getElementById('result').style.display = 'block';
    document.getElementById('result').innerHTML = `<p>变化检测结果展示（模拟）：</p><img src="${URL.createObjectURL(image1)}" alt="Image 1" style="max-width: 45%; margin-right: 10px;"><img src="${URL.createObjectURL(image2)}" alt="Image 2" style="max-width: 45%;">`;
}
