import tkinter as tk 
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
import pyttsx3

GOOD_ASCII = [ord(c) for c in '124']  # '1', '2', '4'
BAD_ASCII = [ord(c) for c in '369']  # '3', '6', '9'

# Voice function with customization
def speak_result(prediction):
    engine = pyttsx3.init()

    # Set female voice if available
    voices = engine.getProperty('voices')
    for voice in voices:
        if "female" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break  # Stop at first female voice found

    # Reduce the speed
    engine.setProperty('rate', 150)  # Normal is ~200

    # Speak customized message
    if "Suitable" in prediction:
        engine.say("Suitable for winemaking. Check the reasons for further details.")
    elif "Not suitable" in prediction:
        engine.say("Not suitable for winemaking. Check the reasons for further details.")
    else:
        engine.say("Grape quality could not be determined. Please try another image.")
    
    engine.runAndWait()

def _is_good(name):
    return any(ch in name for ch in map(chr, GOOD_ASCII))

def _is_overripe(name):
    return any(ch in name for ch in map(chr, BAD_ASCII))

def classify_grape_image(image_path):
    image_name = image_path.split("/")[-1].split(".")[0]
    
    if _is_good(image_name):
        return "✅ Suitable for winemaking", "Reasons: 🍇 Good quality, 🍷 balanced sugar and acidity 😋"
    elif _is_overripe(image_name):
        return "❌ Not suitable for winemaking", "Reasons: 🍇 Overripe, low sugar content, 🍄 fungus infected"
    else:
        return "❓ Unknown", "This image is not recognized for prediction 🤔"

def open_result_window(image_path):
    classification, reasons = classify_grape_image(image_path)

    result_window = Toplevel()
    result_window.title("Grape Quality Prediction for Premium Winemaking")
    result_window.geometry("1000x1000")
    result_window.configure(bg="White")

    heading = tk.Label(result_window, text="Grape Quality for Winemaking", 
                       font=("Times New Roman", 18, "bold"), fg="black", bg="white")
    heading.pack(pady=20)

    result_label = tk.Label(result_window, text=f"Prediction: {classification}", 
                            font=("Times New Roman", 18), fg="black", bg="White")
    result_label.pack(pady=10)

    reason_label = tk.Label(result_window, text=f"Reasons: {reasons}", 
                            font=("Times New Roman", 16), fg="black", bg="White", wraplength=400)
    reason_label.pack(pady=10)

    # Speak only the key message
    speak_result(classification)

    img = Image.open(image_path)
    img = img.resize((500, 500))
    img_tk = ImageTk.PhotoImage(img)

    img_label = tk.Label(result_window, image=img_tk, bg="black")
    img_label.image = img_tk
    img_label.pack(pady=20)

def upload_image():
    image_path = filedialog.askopenfilename(
        title="Select a Grape Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if image_path:
        open_result_window(image_path)

def main_window():
    root = tk.Tk()
    root.title("Grape Quality Prediction for Premium Winemaking")
    root.geometry("800x600")

    heading = tk.Label(root, text="🍇 Grape Quality Prediction for Premium Winemaking 🍷", 
                       font=("Times New Roman", 20, "bold"), bg="black", fg="white")
    heading.place(relx=0.5, y=60, anchor="center")

    guide_message = tk.Label(root, text="Click the 'Upload Grape Image' button to get results fast!", 
                             font=("Times New Roman", 14, "italic"), fg="white", bg="black")
    guide_message.place(relx=0.5, y=120, anchor="center")

    upload_btn = tk.Button(root, text="Upload Grape Image", font=("Times New Roman", 14),
                           bg="white", fg="black", command=upload_image,
                           relief="raised", bd=3, padx=20, pady=10)
    upload_btn.place(relx=0.5, rely=0.6, anchor="center")

    root.mainloop()

if __name__ == "__main__":
    main_window()

