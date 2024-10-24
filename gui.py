import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

# Load the pre-trained MNIST model (you can train or download a model beforehand)
model = tf.keras.models.load_model('mnist_model.h5')

class DigitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Digit Classifier")

        # Set up the canvas
        self.canvas = Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, columnspan=4)

        # Create an image object to store the drawing
        self.image = Image.new("L", (200, 200), color=255)  # 'L' for grayscale
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        self.classify_button = Button(self.root, text="Classify", command=self.classify_digit)
        self.classify_button.grid(row=2, column=1, pady=2, padx=2)

        self.clear_button = Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=2, pady=2, padx=2)

        self.result_label = Label(self.root, text="Draw a digit on the canvas and click Classify.")
        self.result_label.grid(row=1, column=0, pady=2, padx=2, columnspan=4)

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.draw_digit)

    def draw_digit(self, event):
        """Handle the mouse event and draw on the canvas"""
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill='black', width=10)
        self.draw.ellipse([x-8, y-8, x+8, y+8], fill='black')

    def clear_canvas(self):
        """Clear the canvas and reset the image"""
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill="white")

    def classify_digit(self):
        """Classify the drawn digit using the trained model"""
        # Resize the image to 28x28 pixels (MNIST format)
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)  # Invert to get black digits on white background
        img = np.array(img)

        # Normalize the image and reshape for the model
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict the digit using the pre-trained model
        prediction = model.predict([img])
        predicted_digit = np.argmax(prediction)

        # Display the result
        self.result_label.config(text=f"Predicted Digit: {predicted_digit}")

# Set up the application window
root = Tk()
app = DigitClassifierApp(root)
root.mainloop()
