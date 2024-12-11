import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
                             QLineEdit, QFormLayout, QComboBox)
from PyQt5.QtCore import Qt


class MushroomClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Аналіз грибів за введеними параметрами')

        # Створюємо форму для введення параметрів
        self.form_layout = QFormLayout()

        # Поля для числових параметрів
        self.input_cap_diameter = QLineEdit()
        self.form_layout.addRow("Діаметр шапки (cap-diameter):", self.input_cap_diameter)

        self.input_stem_height = QLineEdit()
        self.form_layout.addRow("Висота ніжки (stem-height):", self.input_stem_height)

        self.input_stem_width = QLineEdit()
        self.form_layout.addRow("Ширина ніжки (stem-width):", self.input_stem_width)

        # Випадаючі списки для категоріальних параметрів
        self.input_cap_shape = QComboBox()
        self.input_cap_shape.addItems(["кругла", "конічна", "плоска"])
        self.form_layout.addRow("Форма шапки (cap-shape):", self.input_cap_shape)

        self.input_gill_attachment = QComboBox()
        self.input_gill_attachment.addItems(["вільна", "приросла"])
        self.form_layout.addRow("Прикріплення пластинок (gill-attachment):", self.input_gill_attachment)

        self.input_gill_color = QComboBox()
        self.input_gill_color.addItems(["червоний", "білий", "жовтий", "чорний"])
        self.form_layout.addRow("Колір пластинок (gill-color):", self.input_gill_color)

        self.input_stem_color = QComboBox()
        self.input_stem_color.addItems(["білий", "коричневий", "жовтий", "червоний"])
        self.form_layout.addRow("Колір ніжки (stem-color):", self.input_stem_color)

        self.input_season = QComboBox()
        self.input_season.addItems(["зима", "весна", "літо", "осінь"])
        self.form_layout.addRow("Сезон (season):", self.input_season)

        # Кнопка для запуску аналізу
        self.analyze_button = QPushButton("Аналізувати")
        self.analyze_button.clicked.connect(self.analyze_mushroom)

        # Мітка для результату
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)

        # Загальний лейаут
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.analyze_button)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

        # Завантажуємо модель і скейлер
        self.model = tf.keras.models.load_model('mushroom_model.h5')
        self.scaler_mean = np.load('scaler_mean.npy')
        self.scaler_scale = np.load('scaler_scale.npy')

        # Словники для кодування категоріальних ознак
        self.cap_shape_dict = {"кругла": 0, "конічна": 1, "плоска": 2}
        self.gill_attachment_dict = {"вільна": 0, "приросла": 1}
        self.gill_color_dict = {"червоний": 0, "білий": 1, "жовтий": 2, "чорний": 3}
        self.stem_color_dict = {"білий": 0, "коричневий": 1, "жовтий": 2, "червоний": 3}
        self.season_dict = {"зима": 0, "весна": 1, "літо": 2, "осінь": 3}

    def analyze_mushroom(self):
        try:
            # Зчитуємо числові значення
            cap_diameter = float(self.input_cap_diameter.text())
            stem_height = float(self.input_stem_height.text())
            stem_width = float(self.input_stem_width.text())

            # Зчитуємо значення з випадаючих списків і кодуємо їх
            cap_shape = self.cap_shape_dict[self.input_cap_shape.currentText()]
            gill_attachment = self.gill_attachment_dict[self.input_gill_attachment.currentText()]
            gill_color = self.gill_color_dict[self.input_gill_color.currentText()]
            stem_color = self.stem_color_dict[self.input_stem_color.currentText()]
            season = self.season_dict[self.input_season.currentText()]

            # Формуємо вектор ознак для моделі
            features = np.array([[cap_diameter, cap_shape, gill_attachment, gill_color,
                                  stem_height, stem_width, stem_color, season]])

            # Масштабування даних
            X_input_scaled = (features - self.scaler_mean) / self.scaler_scale

            # Прогноз
            prediction = self.model.predict(X_input_scaled)
            class_idx = np.argmax(prediction)

            if class_idx == 1:
                self.result_label.setText("Гриб їстівний!")
            else:
                self.result_label.setText("Гриб неїстівний!")

        except ValueError:
            self.result_label.setText("Помилка: введіть коректні числові значення.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MushroomClassifierApp()
    window.show()
    sys.exit(app.exec_())