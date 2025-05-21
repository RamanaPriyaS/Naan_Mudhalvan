import easyocr
reader = easyocr.Reader(['en'])
def extract_text(img):
    result = reader.readtext(img)
    return " ".join([text for _, text, _ in result])
