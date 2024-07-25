import os
import re
import fitz  # PyMuPDF

def get_pdf_number(filename):
    """
    Extracts the numeric value from the filename using regex.
    """
    match = re.search(r'_disc_pred_(\d+)k Samples\.pdf', filename)
    if match:
        return int(match.group(1))
    return -1

def combine_pdfs_horizontally(pdf_list, output_file):
    """
    Combines a list of PDFs horizontally into a single wide aspect ratio PDF.
    """
    # Open the first PDF to get its dimensions
    first_pdf = fitz.open(pdf_list[0])
    first_page = first_pdf.load_page(0)
    first_width, first_height = first_page.rect.width, first_page.rect.height
    first_pdf.close()
    
    # Create a new PDF document
    combined_pdf = fitz.open()

    # Width and height to accommodate all pages horizontally
    total_width = len(pdf_list) * first_width
    max_height = first_height
    
    # Create a blank page with the combined width
    combined_page = combined_pdf.new_page(width=total_width, height=max_height)

    x_offset = 0

    for pdf in pdf_list:
        doc = fitz.open(pdf)
        page = doc.load_page(0)
        # Get the page's rectangle dimensions
        rect = page.rect
        # Insert the page into the combined PDF page at the correct position
        combined_page.show_pdf_page(fitz.Rect(x_offset, 0, x_offset + rect.width, max_height), doc, 0)
        x_offset += rect.width
        doc.close()
    
    # Save the combined PDF
    combined_pdf.save(output_file)
    combined_pdf.close()

def main(directory, output_file):
    """
    Main function to combine all PDFs in a directory.
    """
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    
    # Sort files by the numeric value in the filename
    pdf_files.sort(key=lambda x: get_pdf_number(os.path.basename(x)))
    
    combine_pdfs_horizontally(pdf_files, output_file)
    print(f"Combined PDF saved as {output_file}")

if __name__ == "__main__":
    # Example usage
    input_directory = './nn/'
    output_pdf = 'combined_output.pdf'
    main(input_directory, output_pdf)
