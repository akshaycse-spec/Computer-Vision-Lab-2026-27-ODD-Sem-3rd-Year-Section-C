import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------------------------
# Load MNIST Dataset
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# --------------------------------------------------
# Normalize Images
# --------------------------------------------------

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]


# --------------------------------------------------
# CNN Model
# --------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])


# --------------------------------------------------
# Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# Train Model
# --------------------------------------------------

print("\nTraining CNN Model...\n")

history = model.fit(
    x_train,
    y_train,
    epochs=3,
    validation_split=0.1
)


# --------------------------------------------------
# Test Model
# --------------------------------------------------

loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy:", round(acc * 100, 2), "%")


# --------------------------------------------------
# Accuracy Graph
# --------------------------------------------------

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy.png")

plt.close()


# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

pred = np.argmax(
    model.predict(x_test, verbose=0),
    axis=1
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    pred
)

plt.figure()

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

plt.savefig("confusion_matrix.png")

plt.close()


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

report = classification_report(
    y_test,
    pred
)

with open("report.txt", "w") as f:
    f.write(report)

print("\nClassification Report saved as report.txt")


# --------------------------------------------------
# Predict Your Own Image
# --------------------------------------------------

image_path = "img10.jpg"

if os.path.exists(image_path):

    print("\nProcessing your image...")

    img = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        print("Error: Unable to read img10.jpg")

    else:

        # Resize image to 28x28
        img = cv2.resize(
            img,
            (28, 28)
        )

        # Invert image if background is white
        if np.mean(img) > 127:
            img = 255 - img

        # Normalize
        img = img / 255.0

        # Reshape image
        img_input = img.reshape(
            1,
            28,
            28,
            1
        )

        # Predict digit
        result = model.predict(
            img_input,
            verbose=0
        )

        digit = np.argmax(result)

        confidence = np.max(result) * 100

        print(
            "Your Image Prediction:",
            digit
        )

        print(
            "Prediction Confidence:",
            round(confidence, 2),
            "%"
        )


        # --------------------------------------------------
        # Save Your Image Prediction
        # --------------------------------------------------

        plt.figure()

        plt.imshow(
            img,
            cmap="gray"
        )

        plt.title(
            f"Predicted Digit: {digit}"
        )

        plt.axis("off")

        plt.savefig(
            "my_digit_prediction.png"
        )

        plt.close()

else:

    print("\nimg10.jpg not found!")

    print(
        "Please put img10.jpg inside the Experiment_10 folder."
    )


# --------------------------------------------------
# Final Message
# --------------------------------------------------

print("\nAll outputs saved in Experiment_10 folder!")