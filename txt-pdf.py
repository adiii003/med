import os
from reportlab.pdfgen import canvas

def convert_txt_to_pdf(input_folder, output_folder):
    """
    Converts all .txt files in the input folder to .pdf format 
    and saves them in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_name = os.path.splitext(file_name)[0] + ".pdf"
            output_file_path = os.path.join(output_folder, output_file_name)

            with open(input_file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Create a PDF with the text content
            c = canvas.Canvas(output_file_path)
            c.setFont("Helvetica", 12)
            width, height = 595, 842  # A4 size in points
            x, y = 50, height - 50

            for line in text.split("\n"):
                if y < 50:  # Create a new page if the space runs out
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = height - 50
                c.drawString(x, y, line)
                y -= 15  # Line spacing

            c.save()
            print(f"Converted: {file_name} -> {output_file_name}")

# Specify input and output folders
input_folder = "C:/Users/Pyush/Downloads/med/ayurveda_texts"  # Replace with your input folder path
output_folder = "C:/Users/Pyush/Downloads/med/pdf"  # Replace with your output folder path

# Run the conversion
convert_txt_to_pdf(input_folder, output_folder)
