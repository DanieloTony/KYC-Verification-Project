# KYC-Verification-Project
# KYC Verification System (Aadhaar + PAN + Face Match)  

Automated KYC verification tool that extracts data from Aadhaar/PAN cards using OCR, verifies DOB consistency, and performs face matching between Aadhaar photo and live selfie.  

## Features  
- **Aadhaar OCR**: Extracts name, DOB, gender, and Aadhaar number.  
- **PAN OCR**: Extracts name, DOB, and PAN number.  
- **DOB Verification**: Cross-checks dates between Aadhaar and PAN.  
- **Face Matching**: Uses `insightface` to compare Aadhaar photo with selfie.  
- **JSON Export**: Saves extracted data to structured files.  

## Tech Stack  
- Python 3.x  
- OpenCV (`cv2`) for image processing  
- Tesseract OCR (`pytesseract`) for text extraction  
- `insightface` for face recognition  
- Tkinter for GUI  

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt


