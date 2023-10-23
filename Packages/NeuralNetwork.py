import logging
import platform

import numpy as np
from PIL import Image
from PIL.Image import Resampling

IS_TORCH_AVAILABLE = True
IS_TIMM_AVAILABLE = True
IS_ONNX_AVAILABLE = True
IS_RKNN_AVAILABLE = True

try:
    import torch
    from torchvision import transforms
except ImportError as e:
    torch = None
    transforms = None
    IS_TORCH_AVAILABLE = False


try:
    import timm
    from timm.data import create_transform, resolve_data_config
except ImportError as e:
    timm = None
    create_transform, resolve_data_config = None, None
    IS_TIMM_AVAILABLE = False


try:
    import onnxruntime as ort
except ImportError as e:
    ort = None
    IS_ONNX_AVAILABLE = False


try:
    from rknnlite.api import RKNNLite
except ImportError as e:
    RKNNLite = None
    IS_RKNN_AVAILABLE = False


class ImageClassifier_Pytorch:
    """
    A class used to encapsulate the image classification model.

    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch model to use for image classification.
    labels : list of str
        The labels for the classification model.
    device : torch.device
        The device (CPU or GPU) to use for running the model.
    transform : torchvision.transforms.Compose
        The transformations to apply to the input image.
    is_initialized : bool
        Flag indicating whether the model is initialized.

    Methods
    -------
    _create_transform():
        Creates the image transformations.
    predict(image_path: str):
        Predicts the class of the given image.
    """

    def __init__(self, model_name='davit_tiny', checkpoint_path='./Checkpoint/model_best.pth.tar', num_classes=4,
                 labels=('hazardous', 'kitchen', 'other', 'recyclable')):
        """
        Initializes the image classifier with the given model and labels.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use, by default 'davit_tiny'
        checkpoint_path : str, optional
            The path to the checkpoint file, by default './Checkpoint/model_best.pth.tar'
        num_classes : int, optional
            The number of classes, by default 4
        labels : list or tuple of str, optional
            The labels for the classification model, by default ('hazardous', 'kitchen', 'other', 'recyclable')
        """
        self.logger = logging.getLogger("epic2023.ImageClassifier")
        self.logger.info("Using Backend: PyTorch")

        if not IS_TORCH_AVAILABLE or not IS_TIMM_AVAILABLE:
            self.logger.error("Could not import necessary packages.")
            raise SystemExit

        self.model = timm.create_model(
            model_name,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes
        )
        self.logger.info(f"Loaded model '{model_name}' from checkpoint '{checkpoint_path}'")
        self.labels = labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Running model on {self.device}")

        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        print(self.transform)

        self.is_initialized = True

    def predict(self, image):
        """
        Predicts the class of the given image.

        Parameters
        ----------
        image : PIL.Image
            The RGB image to classify.

        Returns
        -------
        list of dict
            The top predicted classes and their probabilities.

        Raises
        ------
        RuntimeError
            If the model is not initialized.
        """
        if not self.is_initialized:
            self.logger.error("Could not predict: model is not initialized.")
            return None

        # Process PIL image with transforms and add a batch dimension
        x = self.transform(image).unsqueeze(0)
        x = x.to(self.device)

        # Pass inputs to model forward function to get outputs
        out = self.model(x)

        # Apply softmax to get predicted probabilities for each class
        probabilities = torch.nn.functional.softmax(out[0], dim=0)

        # Grab the values and indices of top 5 predicted classes
        top_k = min(len(self.labels), 5)
        values, indices = torch.topk(probabilities, top_k)

        # Prepare a nice dict of top k predictions
        predictions = [
            {"label": self.labels[i], "score": v.item()}
            for i, v in zip(indices, values)
        ]

        self.logger.info(f"Inference result: {predictions}")

        return predictions


class ImageClassifier_ONNX:
    """
        A class used to perform image classification with a pre-trained model.

        Attributes:
            session (onnxruntime.InferenceSession): An instance of the Onnx Runtime
                Inference Session for executing the model.
            input_name (str): The name of the model's input.
            is_initialized (bool): A flag indicating if the model has been initialized.
            labels (tuple of str): A tuple containing the labels of the classes.
            transform (torchvision.transforms.Compose): The transformations to be applied
                to the input images.
            logger (logging.Logger): A logger for recording events during model inference.

        Methods:
            predict(image): Predicts the class of the given image.

    """

    def __init__(self, checkpoint_path='./Checkpoint/best_onnx.onnx', num_classes=4,
                 labels=('hazardous', 'kitchen', 'other', 'recyclable')):
        """
            Initializes an instance of the ImageClassifier class.

            Args:
                checkpoint_path (str, optional): The path of the checkpoint file. Default is
                    './Checkpoint/best_onnx.onnx'.
                num_classes (int, optional): The number of classes. Default is 4.
                labels (tuple of str, optional): The labels of the classes. Default is
                    ('hazardous', 'kitchen', 'other', 'recyclable').
        """
        # Initialize the logger
        self.logger = logging.getLogger("epic2023.ImageClassifier")
        self.logger.info("Using Backend: ONNX Runtime")

        if not IS_ONNX_AVAILABLE or not IS_TORCH_AVAILABLE:
            self.logger.error("Could not import necessary packages.")
            raise SystemExit

        self.session = ort.InferenceSession(checkpoint_path)
        self.input_name = self.session.get_inputs()[0].name
        self.is_initialized = True if self.session else False
        self.labels = labels

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize(235, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Log the model and checkpoint path information
        self.logger.info(f"Model loaded from {checkpoint_path}")

        # Log the runtime device
        self.logger.info(f"Model running on: {self.session.get_providers()}")

        if len(self.labels) != num_classes:
            self.logger.error(
                f"Number of labels ({len(self.labels)}) does not match number of classes ({num_classes}).")

    def predict(self, image):
        """
            Predicts the class of the given image.

            Args:
                image (PIL.Image): The input image.

            Returns:
                list of dict: The top k predicted classes and their probabilities.

        """
        if not self.is_initialized:
            self.logger.error("Could not predict: model is not initialized.")
            return None

        # Apply transforms
        image = self.transform(image)

        # Add a batch dimension and convert to numpy array
        image = np.expand_dims(image.numpy(), axis=0)

        # Insert the image into the input tensor
        inputs = {self.input_name: image}

        # Perform inference
        outputs = self.session.run(None, inputs)[0]  # Assuming model has a single output

        # Apply softmax to get predicted probabilities for each class
        probabilities = torch.nn.functional.softmax(torch.from_numpy(outputs[0]), dim=0)

        # Grab the values and indices of top 5 predicted classes
        top_k = min(len(self.labels), 5)
        values, indices = torch.topk(probabilities, top_k)

        # Prepare a nice dict of top k predictions
        predictions = [
            {"label": self.labels[i], "score": float(v)}
            for i, v in zip(indices, values)
        ]

        self.logger.info(f"Inference result: {predictions}")

        return predictions


class ImageClassifier_RKNN:
    def __init__(self, checkpoint_path='./Checkpoint/best_rknn.rknn', num_classes=4,
                 labels=('hazardous', 'kitchen', 'other', 'recyclable')):
        self.logger = logging.getLogger("epic2023.ImageClassifier")
        self.logger.info("Using Backend: RKNN_Lite")

        if not IS_RKNN_AVAILABLE:
            self.logger.error("Could not import necessary packages.")
            raise SystemExit

        if not (platform.system() == 'Linux' and platform.machine() == 'aarch64'):
            self.logger.error("RKNN_Lite only supports Linux aarch64.")
            raise SystemExit
        self.rknn_lite = RKNNLite()
        if self.rknn_lite.load_rknn(checkpoint_path) != 0:
            self.logger.error(f"Could not load model from {checkpoint_path}.")
            raise SystemExit
        self.logger.info(f"Succeeded loading model from {checkpoint_path}.")

        self.labels = labels
        self.is_initialized = True

    @staticmethod
    def resize_and_crop(pil_image, resize_size=(235, 235), crop_size=(224, 224)):
        # 首先，我们使用 PIL 的 resize 方法将图像缩放到指定的大小
        resized_image = pil_image.resize(resize_size, Resampling.BICUBIC)

        # 然后，我们计算裁剪的起始和结束位置
        start = (resized_image.width - crop_size[0]) // 2
        end = start + crop_size[0]

        # 使用 PIL 的 crop 方法进行裁剪
        cropped_image = resized_image.crop((start, start, end, end))

        # 将 PIL 图像转换为 NumPy 数组，并增加一个维度
        final_image = np.expand_dims(np.array(cropped_image), axis=0)

        return final_image

    def predict(self, image):
        if not self.is_initialized:
            self.logger.error("Could not predict: model is not initialized.")
            return None

        image = self.resize_and_crop(image, resize_size=(235, 235), crop_size=(224, 224))

        outputs = self.rknn_lite.inference(inputs=[image])
        outputs = outputs[0][0]

        # 使用numpy的exp和sum替代torch.nn.functional.softmax操作
        probabilities = np.exp(outputs) / np.sum(np.exp(outputs))

        top_k = min(len(self.labels), 5)
        # 使用numpy的argsort和切片操作替代torch.topk操作
        indices = np.argsort(probabilities)[-top_k:][::-1]
        values = probabilities[indices]

        predictions = [
            {"label": self.labels[i], "score": float(v)}
            for i, v in zip(indices, values)
        ]

        self.logger.info(f"Inference result: {predictions}")

        return predictions


if __name__ == '__main__':
    from PIL import Image

    logger = logging.getLogger("epic2023")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # cf = ImageClassifier(checkpoint_path=r'../Checkpoint/model_best.pth.tar')
    cf = ImageClassifier_ONNX(checkpoint_path=r'../Checkpoint/best_onnx.onnx')
    print(cf.predict(Image.open(r'../test_images/k.png')))
