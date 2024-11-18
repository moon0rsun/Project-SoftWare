// document.addEventListener('DOMContentLoaded', function() {
//     document.getElementById('compareButton').addEventListener('click', function(event) {
//         event.preventDefault();
//         var formData = new FormData();
//         var image1 = document.getElementById('image1').files[0];
//         var image2 = document.getElementById('image2').files[0];

//         if (!image1 || !image2) {
//             alert('请选择两张图片');
//             return;
//         }

//         formData.append('image1', image1);
//         formData.append('image2', image2);

//         fetch('/upload', {  // Flask后端的URL
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.blob())
//         .then(blob => {
//             var url = window.URL.createObjectURL(blob);
//             document.getElementById('resultImage').src = url;
//             document.getElementById('resultImage').style.display = 'block'; // 显示图片
//         })
//         .catch(error => console.error('Error:', error));
//     });
// });

document.addEventListener('DOMContentLoaded', function() {
    // 监听文件输入框的变化，实现图片预览
    document.getElementById('image1').addEventListener('change', function(event) {
        var reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('image1Display').src = e.target.result;
        };
        reader.readAsDataURL(event.target.files[0]);
    });

    document.getElementById('image2').addEventListener('change', function(event) {
        var reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('image2Display').src = e.target.result;
        };
        reader.readAsDataURL(event.target.files[0]);
    });

    // 监听对比按钮的点击事件
    document.getElementById('compareButton').addEventListener('click', function(event) {
        event.preventDefault();
        var formData = new FormData();
        var image1 = document.getElementById('image1').files[0];
        var image2 = document.getElementById('image2').files[0];

        if (!image1 || !image2) {
            alert('请选择两张图片');
            return;
        }

        formData.append('image1', image1);
        formData.append('image2', image2);

        fetch('/upload', { // Flask后端的URL
                method: 'POST',
                body: formData
            })
            .then(response => response.blob())
            .then(blob => {
                var url = window.URL.createObjectURL(blob);
                document.getElementById('resultImage').src = url;
                document.getElementById('resultImage').style.display = 'block'; // 显示图片
                document.getElementById('three').style.display = 'block'; // 显示结果区域
            })
            .catch(error => console.error('Error:', error));
    });
});