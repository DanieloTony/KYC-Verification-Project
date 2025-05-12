import cv2
import pytesseract
import re
import json
from tkinter import Tk, Label, Button, filedialog, Text, messagebox, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
import numpy as np
import os

# Set path for tesseract executable
output_dir = os.path.join(os.path.expanduser("~"), "KYC_Outputs")
os.makedirs(output_dir, exist_ok=True)

parsed_details = {}
aadhaar_face = None
selfie_face = None

# Enhanced image preprocessing for better OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Reduce noise
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Parse Aadhaar text using improved rules and config
def parse_aadhaar_details(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    aadhaar_number = re.search(r'\d{4}\s\d{4}\s\d{4}', text)
    aadhaar_number = aadhaar_number.group() if aadhaar_number else "Not found"

    dob_match = re.search(r'(DOB|YOB)[:\s]*([\d]{2}[-/][\d]{2}[-/][\d]{4}|\d{4})', text, re.IGNORECASE)
    dob = dob_match.group(2) if dob_match else "Not found"

    gender_match = re.search(r'\b(Male|Female|Others)\b', text, re.IGNORECASE)
    gender = gender_match.group() if gender_match else "Not found"

    name = "Not found"
    dob_line_index = -1
    for idx, line in enumerate(lines):
        if 'dob' in line.lower() or 'yob' in line.lower():
            dob_line_index = idx
            break

    if dob_line_index > 0:
        candidate = lines[dob_line_index - 1]
        if re.match(r'^[A-Za-z ]{3,}$', candidate) and not any(w in candidate.lower() for w in ['government', 'india']):
            name = candidate.strip()

    if name == "Not found":
        for line in lines[:5]:
            if re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)+$', line):
                name = line.strip()
                break

    return {
        "Name": name,
        "DOB/YOB": dob,
        "Gender": gender,
        "Aadhaar Number": aadhaar_number
    }

# Parse PAN card text
def parse_pan_details(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    pan_number_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text)
    pan_number = pan_number_match.group() if pan_number_match else "Not found"

    dob_match = re.search(r'\d{2}/\d{2}/\d{4}', text)
    dob = dob_match.group() if dob_match else "Not found"

    name = "Not found"
    for line in lines:
        if re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)+$', line):
            name = line.strip()
            break

    return {
        "Name": name,
        "DOB": dob,
        "PAN Number": pan_number
    }

# Verify DOB match between Aadhaar and PAN
def verify_dob_match(pan_details):
    aadhaar_dob = parsed_details.get("DOB/YOB", "").replace("-", "/").replace(" ", "")
    pan_dob = pan_details.get("DOB", "").replace(" ", "")

    if aadhaar_dob in pan_dob or pan_dob in aadhaar_dob:
        messagebox.showinfo("DOB Match", "Date of Birth matches between Aadhaar and PAN.")
    else:
        messagebox.showwarning("DOB Mismatch", f"Aadhaar DOB: {aadhaar_dob}\nPAN DOB: {pan_dob}")

# Save PAN JSON only with name, dob, pan number
def save_pan_json(pan_details):
    name = pan_details.get("Name", "PAN_output").replace(" ", "_")
    directory = "your path to save json"
    os.makedirs(directory, exist_ok=True)

    minimal_details = {
        "Name": pan_details.get("Name", "Not found"),
        "DOB": pan_details.get("DOB", "Not found"),
        "PAN Number": pan_details.get("PAN Number", "Not found")
    }

    json_path = os.path.join(directory, f"{name}_pan.json")
    with open(json_path, 'w') as f:
        json.dump(minimal_details, f, indent=4)

    messagebox.showinfo("Saved", f"PAN details saved to {json_path}")

# OCR with config
def process_image(file_path):
    image = cv2.imread(file_path)
    preprocessed = preprocess_image(image)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed, config=config)
    return text, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Upload Aadhaar image
def upload_aadhaar_image():
    global parsed_details, aadhaar_face
    file_path = filedialog.askopenfilename()
    if file_path:
        text, rgb_image = process_image(file_path)
        parsed_details = parse_aadhaar_details(text)

        image_pil = Image.fromarray(rgb_image)
        image_tk = ImageTk.PhotoImage(image_pil)
        image_label.config(image=image_tk)
        image_label.image = image_tk

        result_box.delete("1.0", "end")
        result_box.insert("end", text)

        aadhaar_face = extract_face_encoding(rgb_image)

# Upload PAN image
def upload_pan_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        text, rgb_image = process_image(file_path)
        pan_details = parse_pan_details(text)

        result_box.insert("end", f"\n--- PAN Card ---\n{text}\n")
        verify_dob_match(pan_details)
        save_pan_json(pan_details)

# Capture selfie
def capture_selfie():
    global selfie_face
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_pil = Image.fromarray(rgb_image)
        image_tk = ImageTk.PhotoImage(image_pil)
        selfie_image_label.config(image=image_tk)
        selfie_image_label.image = image_tk

        selfie_face = extract_face_encoding(rgb_image)

        if aadhaar_face is not None and selfie_face is not None:
            verify_faces_and_save(rgb_image)

# Extract face
def extract_face_encoding(image):
    faces = face_app.get(image)
    if len(faces) > 0:
        return faces[0].embedding
    return None

# Cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Verify and save
def verify_faces_and_save(selfie_image):
    similarity = cosine_similarity(aadhaar_face, selfie_face)
    if similarity > 0.4:
        messagebox.showinfo("Success", "Faces match!")
        save_json_auto(selfie_image)
    else:
        messagebox.showwarning("Failed", "Faces do not match.")

# Auto save Aadhaar JSON and selfie image
def save_json_auto(selfie_image):
    if not parsed_details:
        return
    name = parsed_details.get("Name", "output").replace(" ", "_")
    directory = "your path"
    os.makedirs(directory, exist_ok=True)

    json_path = os.path.join(directory, f"{name}.json")
    image_path = os.path.join(directory, f"{name}.jpg")

    with open(json_path, 'w') as f:
        json.dump(parsed_details, f, indent=4)

    selfie_image_bgr = cv2.cvtColor(np.array(selfie_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, selfie_image_bgr)

    messagebox.showinfo("Saved", f"JSON and selfie saved as {name}.json and {name}.jpg")

# GUI setup
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

root = Tk()
root.title("Aadhaar & PAN OCR with Face Verification")
root.geometry("650x650")

canvas = Canvas(root, borderwidth=0)
scroll_frame = Frame(canvas)
vsb = Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=vsb.set)

vsb.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

Label(scroll_frame, text="Aadhaar & PAN OCR with Live Face Verification", font=("Helvetica", 16)).pack(pady=10)
Button(scroll_frame, text="Upload Aadhaar Image", command=upload_aadhaar_image).pack(pady=5)
Button(scroll_frame, text="Upload PAN Card Image", command=upload_pan_image).pack(pady=5)
Button(scroll_frame, text="Capture Selfie", command=capture_selfie).pack(pady=5)


image_label = Label(scroll_frame)
image_label.pack(pady=5)

selfie_image_label = Label(scroll_frame)
selfie_image_label.pack(pady=5)

result_box = Text(scroll_frame, height=10, width=60)
result_box.pack(pady=10)

root.mainloop()
