## 深度学习小组作业

- ### 如何训练、测试以及应用

  ```pyhton 
    parser = argparse.ArgumentParser(description='Fer2013 Emotion Recognition')
    parser.add_argument('--model', type=str, default='CNN', help='Model to use: CNN, VGG, ResNet, mini_XCEPTION')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train, test, demo')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer: sgd, adam, rmsprop')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='Loss function: cross_entropy, mse')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    ```
    **示例：在终端输指令即可**

    `python main.py --mode train --model CNN --epochs 20`
- ### 列表里的一些东西
  
  - #### CNNepoch_1.phg：训练时每个Epoch的loss曲线
  - #### checkpoint：训练时，每个模型最好的模型参数
  - #### data/emojis：demo中显示的表情（抹茶旦旦）
  - #### data/test && data/train：测试集和训练集（感觉模型不会太好，就没搞验证集）
  - #### data/video：demo中的视频 
- ### 函数说明
  - #### demo：我每次在微信里发的就是运行的demo
  - #### main：主函数（不多说了）
  - #### train && test && model：训练模型、测试模型、五个需要训练的模型
  - #### haarcascade_frontalface_default.xml：OpenCv中检测人脸的模型
  - #### Csv_to_Image：将fer2013.csv里的内容转换为Image

