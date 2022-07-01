from Resnet101 import *
import gradio as gr
from PIL import Image

print("Loading Resnet101 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("resnet101_ckpt.pth")
net = ResNet101()
net.to(device)
net = torch.nn.DataParallel(net)
net.load_state_dict(model['net'])

print("Model loaded")
print("Device: ", device)

# Define a transform to convert the image to tensor
transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def predict_image(image):

    # Convert the image to PyTorch tensor
    img_tensor = transform(Image.fromarray(image))
    img_tensor.to(device)
    with torch.no_grad():
        outputs = net(img_tensor[None, ...])
        _, predicted = outputs.max(1)
        classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
        res = classes[predicted[0].item()]
        print("Predicted class: ", res)
        if res == 'car':
            return Image.open("samples/car2.jpeg"), Image.open("samples/car3.jpg"), Image.open("samples/car4.jpg"), Image.open("samples/car5.jpg")
        elif res == 'cat':
            return Image.open("samples/cat2.jpg"), Image.open("samples/cat3.jpeg"), Image.open("samples/cat4.png"), Image.open("samples/cat5.jpg")
        elif res == 'dog':
            return Image.open("samples/dog2.jpg"), Image.open("samples/dog3.jpg"), Image.open("samples/dog4.jpg"), Image.open("samples/dog5.jpg")
        elif res == 'horse':
            return Image.open("samples/horse2.jpg"), Image.open("samples/horse3.jpeg"), Image.open("samples/horse4.jpg"), Image.open("samples/horse5.jpg")

def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])

demo = gr.Blocks()
with demo:
    gr.Markdown('''
    <center>
    <h1>Image Classification trained on Resnet101</h1>
    <p>
    This is a demo of the image classification model trained on Resnet101. The dataset used is the CIFAR-10 dataset.
    </p>
    </center>
    ''')
    
    with gr.Row():
        input_image = gr.Image(label="Input image")
    with gr.Row():
        output_imgs = [gr.Image(label='Closest Image 1', type='numpy', interactive=False),
                        gr.Image(label='Closest Image 2', type='numpy', interactive=False),
                        gr.Image(label='Closest Image 3', type='numpy', interactive=False),
                        gr.Image(label='Closest Image 4', type='numpy', interactive=False)]
    button = gr.Button("Classifier")
    with gr.Row():
        example_images = gr.Dataset(components=[input_image],
                                    samples=[["samples/cat1.jpg"], ["samples/car1.jpg"], ["samples/dog1.jpeg"], ["samples/horse1.jpg"]])
    example_images.click(fn=set_example_image, inputs=example_images, outputs=example_images.components)
    button.click(predict_image, inputs=input_image, outputs=output_imgs)

demo.launch(server_port=5000, share=True)