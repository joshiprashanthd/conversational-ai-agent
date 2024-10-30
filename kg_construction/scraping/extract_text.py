import PyPDF2
import argparse


def extract_text_from_pdf(pdf_path, start_page=0, end_page=None):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        if end_page is None or end_page > total_pages:
            end_page = total_pages

        extracted_text = ""

        for page_num in range(start_page, end_page):
            page = pdf_reader.pages[page_num]
            extracted_text += page.extract_text()

        return extracted_text


def main():
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument(
        "pdf_path", metavar="PDF_PATH", type=str, help="Path to the PDF file"
    )
    parser.add_argument(
        "--start_page",
        type=int,
        default=0,
        help="Start page (default is 0, the first page)",
    )
    parser.add_argument(
        "--end_page", type=int, help="End page (optional, default is the last page)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs.txt",
        help="Output file to save the extracted text (optional)",
    )

    args = parser.parse_args()

    extracted_text = extract_text_from_pdf(
        args.pdf_path, args.start_page, args.end_page
    )

    if args.output:
        with open(args.output, "w") as output_file:
            output_file.write(extracted_text)
        print(f"Text has been written to {args.output}")
    else:
        print(extracted_text)


if __name__ == "__main__":
    main()
