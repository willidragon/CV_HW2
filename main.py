#main.py
#main gui implemented here

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame, QGridLayout, QFileDialog
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
import cv2
from graffiti import GraffitiBoard

from q1 import perform_background_subtraction
import q2
from q3 import pca_dimension_reduction
from q4 import show_model_structure, showAccuracyAndLoss, predict
from q5 import show_images_q5, show_model_structure_q5, show_comparison, show_inference_catdog


from PyQt5.QtWidgets import QLineEdit




class ImageDisplayArea(QWidget):
    def __init__(self, *args, **kwargs):
        super(ImageDisplayArea, self).__init__(*args, **kwargs)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def loadImage(self, fileName):
        self.image.load(fileName)
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left-side panel
        left_panel_layout = QVBoxLayout()

        # Loading Functionalities section
        loading_layout = QVBoxLayout()
        loading_frame = QFrame()
        loading_frame.setFrameShape(QFrame.StyledPanel)
        loading_frame.setLayout(loading_layout)

        #initialize model path
        self.mnist_model_path = "model/vgg19_bn_mnist.pth"

        self.cat_dog_model_path = "model/cat_dog.pth"
        self.cat_dog_with_RandoErasing_model_path = "model/cat_dog_with_RandoErasing.pth"

        #initialize accuray and loss path
        self.accuracy_loss_path_q4 = "model/q4_acc_loss/acc_loss_graph_q4.png"

        # Initialize video path
        self.video_path = None

        # Initialize image path
        self.image_path = None

        # initialize image path for resnet
        self.resnet_image_path = None


        # Create an instance of GraffitiBoard
        self.graffitiBoard = GraffitiBoard()
        self.graffitiBoard.setFixedSize(200, 200)

        # Initialize the labels for displaying prediction results
        self.predicted_class_label = QLabel('Predicted class: None')
        self.max_probability_label = QLabel('Probability: None')

        # Initialize the label for displaying prediction result from ResNet50
        self.resnet_predicted_class_label = QLabel('Predicted class: None')


        # Button for loading image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.loadImage)
        loading_layout.addWidget(self.load_image_button)
        
        # Button for loading video
        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.clicked.connect(self.loadVideo)
        loading_layout.addWidget(self.load_video_button)

        left_panel_layout.addWidget(loading_frame)

        # Background Subtraction section
        bg_sub_layout = QVBoxLayout()
        bg_sub_frame = QFrame()
        bg_sub_frame.setFrameShape(QFrame.StyledPanel)
        bg_sub_frame.setLayout(bg_sub_layout)
        
        bg_sub_label = QLabel("1. Background Subtraction")
        background_subtraction_button = QPushButton("1.1 background Subtraction")
        background_subtraction_button.clicked.connect(self.performBackgroundSubtraction)


        bg_sub_layout.addWidget(bg_sub_label)
        bg_sub_layout.addWidget(background_subtraction_button)

        left_panel_layout.addWidget(bg_sub_frame)

        # Optical Flow section
        optical_flow_layout = QVBoxLayout()
        optical_flow_frame = QFrame()
        optical_flow_frame.setFrameShape(QFrame.StyledPanel)
        optical_flow_frame.setLayout(optical_flow_layout)
        
        optical_flow_label = QLabel("2. Optical Flow")
        preprocessing_button = QPushButton("2.1 Preprocessing")
        preprocessing_button.clicked.connect(self.callPreprocessing)
        videotracking_button = QPushButton("2.2 Videotracking")
        videotracking_button.clicked.connect(self.callVideoTracking)

        optical_flow_layout.addWidget(optical_flow_label)
        optical_flow_layout.addWidget(preprocessing_button)
        optical_flow_layout.addWidget(videotracking_button)

        left_panel_layout.addWidget(optical_flow_frame)

        # PCA section
        pca_layout = QVBoxLayout()
        pca_frame = QFrame()
        pca_frame.setFrameShape(QFrame.StyledPanel)
        pca_frame.setLayout(pca_layout)

        pca_label = QLabel("3. PCA")

        # Input field for reconstruction error threshold
        # Label for reconstruction error threshold
        reconstruction_error_label = QLabel("Reconstruction Error")
        pca_layout.addWidget(reconstruction_error_label)

        self.error_threshold_input = QLineEdit("3.0")  # Default value set to 3.0
        self.error_threshold_input.setPlaceholderText("Enter MSE threshold")
        pca_layout.addWidget(self.error_threshold_input)

        dimension_reduction_button = QPushButton("3. Dimension Reduction")
        dimension_reduction_button.clicked.connect(self.performPCADimensionReduction)


        pca_layout.addWidget(pca_label)
        pca_layout.addWidget(dimension_reduction_button)

        left_panel_layout.addWidget(pca_frame)

        # Right-side panel for display
        right_panel_layout = QVBoxLayout()

        # MNIST Classifier section with a drawing area
        mnist_layout = QGridLayout()
        mnist_frame = QFrame()
        mnist_frame.setFrameShape(QFrame.StyledPanel)
        mnist_frame.setLayout(mnist_layout)

        mnist_layout.addWidget(self.predicted_class_label, 5, 1)  # Adjust the grid position as needed
        mnist_layout.addWidget(self.max_probability_label, 6, 1) 

        mnist_label = QLabel("4. MNIST Classifier Using VGG19")
        mnist_canvas = self.graffitiBoard
        mnist_canvas.setMinimumSize(200, 200)

        mnist_layout.addWidget(mnist_label, 0, 0, 1, 2)

        # Define buttons and connect them to functions
        mnist_button_names_functions = [
            ("Show Model Structure", self.displayModelStructure_q4),
            ("Show Accuracy and Loss", self.showAccuracyAndLoss_q4),
            ("Predict", self.on_predict_clicked),
            ("Reset", self.graffitiBoard.resetCanvas)
        ]

        for i, (name, function) in enumerate(mnist_button_names_functions, start=1):
            button = QPushButton(f"4.{i} {name}")
            button.clicked.connect(function)  # Connect each button to its respective function
            mnist_layout.addWidget(button, i, 0)

        mnist_layout.addWidget(mnist_canvas, 1, 1, 4, 1)

        right_panel_layout.addWidget(mnist_frame)

        # Right-side panel for display
        # right_panel_layout = QVBoxLayout()

        # ResNet50 section with an image display area
        resnet_layout = QGridLayout()
        resnet_frame = QFrame()
        resnet_frame.setFrameShape(QFrame.StyledPanel)
        resnet_frame.setLayout(resnet_layout)

        resnet_label = QLabel("5. ResNet50")

        # Initialize the ImageDisplayArea for ResNet50 and assign it to self.resnet_image_display
        self.resnet_image_display = ImageDisplayArea()  # This line is crucial
        self.resnet_image_display.setMinimumSize(200, 200)

        resnet_load_image_button = QPushButton("5.1 Load Image")
        resnet_load_image_button.clicked.connect(self.loadImage_resnet)  # Not self.loadImage
        # connect the load_image specific to the resnet section #chatgpt look here
        resnet_layout.addWidget(resnet_label, 0, 0, 1, 2)
        resnet_layout.addWidget(resnet_load_image_button, 1, 0)
        # Adding the label to the ResNet50 layout
        resnet_layout.addWidget(self.resnet_predicted_class_label, 6, 1)  # Adjust the grid position as needed
        # Define buttons and connect them to functions
        resnet_button_names_functions = [
            ("Show Images", self.show_images_q5),
            ("Show Model Structure", self.showModelStructure),
            ("Show Comparison", show_comparison),
            ("Inference", self.show_inference_q5)
        ]

        for i, (name, function) in enumerate(resnet_button_names_functions, start=2):  # Start at 2 due to the Load Image button
            button = QPushButton(f"5.{i} {name}")
            button.clicked.connect(function)  # Connect each button to its respective function
            resnet_layout.addWidget(button, i, 0)  # Adjust grid placement for each button

        resnet_layout.addWidget(self.resnet_image_display, 1, 1, 5, 1)  # Span 5 rows to align with the buttons

        right_panel_layout.addWidget(resnet_frame)

        
        # Set the main layout
        main_layout.addLayout(left_panel_layout, 1)
        main_layout.addLayout(right_panel_layout, 2)
        
        # Set the central widget and the main window layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set window title and geometry
        self.setWindowTitle("PyQt UI")
        self.setGeometry(100, 100, 800, 600)

        # Correct indentation for method definitions

    def loadImage(self):
        # Open QFileDialog to select an image file
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)")
        if self.image_path:
            # For now, just display the path in the status bar
            self.statusBar().showMessage(f"Loaded image: {self.image_path}")


    def loadImage_resnet(self):
        # Open QFileDialog to select an image file
        self.resnet_image_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)")
        if self.resnet_image_path:
            # Make sure you're using the same instance of ImageDisplayArea that's added to the layout
            self.resnet_image_display.loadImage(self.resnet_image_path)
            self.statusBar().showMessage(f"Loaded image: {self.resnet_image_path}")

    def loadVideo(self):
        # Open QFileDialog to select a video file
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi)")
        if self.video_path:
            self.statusBar().showMessage(f"Loaded video: {self.video_path}")

    def performBackgroundSubtraction(self):
        if not self.video_path:
            self.statusBar().showMessage('No video loaded!')
            return
        
        # Call the perform_background_subtraction function from q1.py
        perform_background_subtraction(self.video_path)

    def callPreprocessing(self):
        if not self.video_path:
            self.statusBar().showMessage('No video loaded!')
            return
        
        # Call the preprocess_video function from q2.py
        q2.preprocess_video(self.video_path)

    def callVideoTracking(self):
        if not self.video_path:
            self.statusBar().showMessage('No video loaded!')
            return
        
        # Call the video_tracking function from q2.py
        q2.video_tracking(self.video_path)

    def performPCADimensionReduction(self):
        # Open QFileDialog to select an image file
        
        if self.image_path:
            # Get the error threshold value from the input field
            try:
                error_threshold = float(self.error_threshold_input.text())
            except ValueError:
                self.statusBar().showMessage('Invalid MSE threshold value. Please enter a valid number.')
                return
            
            # Call the pca_dimension_reduction function from q3.py with the selected image and error threshold
            pca_dimension_reduction(self.image_path, error_threshold)

    def displayModelStructure_q4(self):
        # Call the function from q4.py and pass the path of the trained model
        show_model_structure(self.mnist_model_path)

    def showAccuracyAndLoss_q4(self):
        # Call the function from q4.py and pass the path of the accuracy and loss graph
        showAccuracyAndLoss(self.accuracy_loss_path_q4)

    # This would be a slot connected to the "Predict" button in your PyQt application
    def on_predict_clicked(self):
        # Grab the current pixmap from the GraffitiBoard
        graffiti_board_pixmap = self.graffitiBoard.grab()

        # Call the predict function from q4.py and pass the necessary arguments
        predicted_class, max_probability = predict(graffiti_board_pixmap, self.mnist_model_path)

        # Update the GUI with the predicted class and probability
        self.predicted_class_label.setText(f'Predicted class: {predicted_class}')
        self.max_probability_label.setText(f'Probability: {max_probability:.2f}')

    def show_images_q5(self):
        # Call the showImages function from q5.py
        show_images_q5()    

    def showModelStructure(self):
        # Call the showModelStructure function from q5.py
        show_model_structure_q5(self.cat_dog_model_path)

    def show_inference_q5(self):
        if not self.resnet_image_path:
            self.statusBar().showMessage('No image loaded for ResNet prediction!')
            return

        # Assuming show_inference_q5 function returns only the predicted class
        prediction = show_inference_catdog(self.cat_dog_with_RandoErasing_model_path, self.resnet_image_path)

        # Update the GUI with the ResNet predicted class
        self.resnet_predicted_class_label.setText(f'Predicted class: {prediction}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())