import cv2


class SmileDetector:
    def __init__(self):
        # Carga de clasificadores en cascada (Haar Cascades)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        if self.face_cascade.empty() or self.smile_cascade.empty():
            raise IOError("No se pudieron cargar los archivos XML de Haar Cascades.")

    def run(self):
        # Inicialización de la captura de video (0 es la webcam por defecto)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: No se puede acceder a la webcam.")
            return

        print("Presione 'q' para salir.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Pre-procesamiento: Conversión a escala de grises para Viola-Jones
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Ecualización de histograma para mejorar el contraste en biomedicina/entornos variables
            gray = cv2.equalizeHist(gray)

            # Detección de rostros
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            for x, y, w, h in faces:
                # Dibujar ROI del rostro
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray = gray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]

                # Detección de sonrisa dentro de la ROI del rostro
                # Los parámetros scaleFactor y minNeighbors definen la sensibilidad
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.5, minNeighbors=32, minSize=(25, 25)
                )

                if len(smiles) > 0:
                    for sx, sy, sw, sh in smiles:
                        # Representación de la "Característica" detectada
                        cv2.rectangle(
                            roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2
                        )
                        cv2.putText(
                            frame,
                            "Sonrisa Detectada",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                else:
                    cv2.putText(
                        frame,
                        "No detectada / Serio",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("Deteccion Viola-Jones (Webcam)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = SmileDetector()
    detector.run()
