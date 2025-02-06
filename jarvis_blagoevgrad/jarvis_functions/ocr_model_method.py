import easyocr

reader = easyocr.Reader(['bg', 'en'])  # Load Bulgarian and English models
results = reader.readtext('testing_image_for_OCR_model.png')  # Perform OCR on your image

# Extract only the recognized text and join them into a sentence
sentence = ' '.join([result[1] for result in results])
print(sentence)
