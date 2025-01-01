import base64
import json
import xml.etree.ElementTree as ET
import streamlit as st
import time
from io import BytesIO
from typing import List
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import tempfile
from pdf2image import convert_from_path
import io
from PIL import Image
import chardet


load_dotenv()
client = OpenAI()
model = "gpt-4o"


def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        st.error(f"Error reading image file: {e}")
        return None


def convert_image_to_base64(file):
    # Open the image
    image = Image.open(file)
    # Create a byte stream
    byte_stream = io.BytesIO()
    # Save the image to the byte stream
    image.save(byte_stream, format='PNG')
    # Get the byte data from the stream
    byte_data = byte_stream.getvalue()
    # Encode the byte data to base64
    base64_image = base64.b64encode(byte_data).decode('utf-8')
    return base64_image


def load_sectors(json_file_path='sectors.json'):
    try:
        with open(json_file_path, 'r') as f:
            sectors_data = json.load(f)
            if not sectors_data:
                sectors_data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        sectors_data = {}
    return sectors_data


def xml_to_dict(element):
    result = {}
    for child in element:
        if len(child) == 0:
            result[child.tag] = child.text
        else:
            result[child.tag] = xml_to_dict(child)
    return result


def retry_request(input_file_id,
                  metadata,
                  endpoint="/v1/chat/completions",
                  completion_window="24h",
                  max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            return client.batches.create(input_file_id=input_file_id,
                                         endpoint=endpoint,
                                         completion_window=completion_window,
                                         metadata=metadata,
                                         )
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Request failed. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e


def retry_request_create(file, purpose, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            return client.files.create(file=file, purpose=purpose)
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Request failed. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e


def save_sectors(sectors_data, json_file_path='sectors.json'):
    try:
        with open(json_file_path, 'w') as f:
            json.dump(sectors_data, f, indent=2)
        return True
    except IOError as e:
        st.error(f"Error saving sectors data: {e}")
        return False


def parse_pdfs(file, file_name, save_path, pdf_progress, pdf_status, overall_status, file_index, total_files, batching):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    try:
        pages = convert_from_path(pdf_path=temp_file_path, dpi=200, grayscale=True, output_folder=save_path)
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return None

    total_pages = len(pages)

    pdf_info = {
        "file_name": file_name,
        "total_pages": total_pages,
        "pages": []
    }

    context = ""
    for page_number, page in enumerate(pages, start=1):
        image_path = f'{save_path}/pdf_{file_name}_page_{page_number}.jpg'
        page.save(image_path, 'JPEG')
        base64_image = encode_image(image_path)

        jsonl_file_path = os.path.join(save_path, f'output_{file_name}_page_{page_number}.jsonl')
        batch_job = None

        if batching:
            payload = {
                "custom_id": f"{page_number}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text and tables from this image. For tables, format them "
                                            "as markdown tables. Explain any info-graphics "
                                            "Do not change or fill any value by yourself."
                                            "Stick to the text in image only."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000
                }
            }

            with open(jsonl_file_path, 'w') as f:
                json.dump(payload, f)
                f.write('\n')

            batch_input_file = retry_request_create(file=open(jsonl_file_path, "rb"), purpose="batch")

            batch_job = retry_request(input_file_id=batch_input_file.id,
                                      endpoint="/v1/chat/completions",
                                      completion_window="24h",
                                      metadata={
                                          "description": f"extract text and tables"
                                      })
        else:
            batch_job = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract all text and tables from this image. For tables, format them "
                                    "as markdown tables. Explain any info-graphics. "
                                    "Do not change or fill any value by yourself. "
                                    "Stick to the text in image only."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            context = batch_job.choices[0].message.content
            #st.write(response)


        pdf_info["pages"].append({
            "page_number": page_number,
            "batch_id": batch_job.id,
            "image_path": image_path,
            "context": context
        })

        pdf_progress.progress(page_number / total_pages)
        pdf_status.markdown(
            f"<p class='status-text'>Processing PDF {file_name}, Page {page_number}/{total_pages}</p>",
            unsafe_allow_html=True)

        overall_progress_percentage = ((file_index * total_pages + page_number) / (total_files * total_pages)) * 100
        overall_status.markdown(
            f"<p class='status-text'>Overall Progress: {overall_progress_percentage:.2f}% completed</p>",
            unsafe_allow_html=True
        )

        time.sleep(2)
    os.unlink(temp_file_path)
    return pdf_info


def parse_xml(file, file_name, save_path, xml_progress, xml_status):
    try:
        tree = ET.parse(file)
        root = tree.getroot()

        content = ""
        xml_dict = xml_to_dict(root)
        total_elements = len(xml_dict)

        for index, (key, value) in enumerate(xml_dict.items(), 1):
            modified_key = re.sub(r'\{.*?\}', '', key)
            modified_key = re.sub(r'(?<!^)(?=[A-Z])', ' ', modified_key)
            if isinstance(value, str):
                content += f"{modified_key}: {value}.\n"

            xml_progress.progress(index / total_elements)
            xml_status.markdown(f"<p class='status-text'>Processing XML element {index}/{total_elements}</p>",
                                unsafe_allow_html=True)

        xml_info = {"file_name": file_name, "page_content": content}
        return xml_info

    except ET.ParseError as e:
        st.error(f"Error parsing XML file {file_name}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error processing {file_name}: {e}")
        return None


def parse_jpgs(file, file_name, save_path, jpg_progress, jpg_status, overall_status, file_index, total_files, batching):
    jpg_info = {
        "file_name": file_name,
        "total_pages": 1,
        "pages": []
    }
    jsonl_file_path = os.path.join(save_path, f'output_{file_name}.jsonl')
    base64_image = convert_image_to_base64(file)

    context = ""
    batch_job = None

    if batching:
        payload = {
            "custom_id": f"{file_name}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text and tables from this image. For tables, format them "
                                        "as markdown tables. Explain any info-graphics "
                                        "Do not change or fill any value by yourself."
                                        "Stick to the text in image only."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2000
            }
        }
        with open(jsonl_file_path, 'w') as f:
            json.dump(payload, f)
            f.write('\n')
        batch_input_file = retry_request_create(file=open(jsonl_file_path, "rb"), purpose="batch")
        batch_job = retry_request(input_file_id=batch_input_file.id,
                                    endpoint="/v1/chat/completions",
                                    completion_window="24h",
                                    metadata={
                                        "description": f"extract text and tables"
                                })
    else:
        batch_job = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text and tables from this image. For tables, format them "
                                    "as markdown tables. Explain any info-graphics "
                                    "Do not change or fill any value by yourself."
                                    "Stick to the text in image only."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        context = batch_job.choices[0].message.content
   
    jpg_info["pages"].append({
        "page_number": -1,
        "batch_id": batch_job.id,
        "context": context
    })
    
    # Update JPG progress based on total number of image files
    jpg_progress.progress((file_index + 1) / total_files)
    
    jpg_status.markdown(
        f"<p class='status-text'>Processing Image {file_name}</p>",
        unsafe_allow_html=True)
   
    overall_progress_percentage = ((file_index + 1) / total_files) * 100
    overall_status.markdown(
        f"<p class='status-text'>Overall Progress: {overall_progress_percentage:.2f}% completed</p>",
        unsafe_allow_html=True
    )
    time.sleep(2)
    return jpg_info                                


def parse_text(file, file_name, save_path, text_progress, text_status, overall_status, file_index, total_files):
    try:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        text = raw_data.decode(encoding)
        text_info = {"file_name": file_name, "page_content": text}
        return text_info
    except Exception as e:
        st.error(f"Error processing text file {file_name}: {e}")
        return None


def parse_files(files: List[BytesIO], save_path: str, sector_name: str, project_name: str, year, batching, target_language) -> None:
    start_time = time.time()
    total_files = len(files)

    sectors_data = load_sectors()

    if sector_name not in sectors_data:
        sectors_data[sector_name] = {"projects": []}

    project = next((p for p in sectors_data[sector_name]["projects"] if p["project_name"] == project_name), None)
    if not project:
        project = {
            "project_id": str(uuid.uuid4()),
            "project_name": project_name,
            "files": [],
            "time_created": start_time,
            "path": save_path,
            "xml": [],
            "txt": [],
            "year": year,
            "batching": batching,
            "target_language": target_language,
        }
        sectors_data[sector_name]["projects"].append(project)

    with st.spinner('Processing files...'):
        st.subheader("Overall Progress")
        overall_progress = st.progress(0)
        file_status = st.empty()
        time_status = st.empty()
        
        # Track which file types are present
        has_pdfs = any(file.name.endswith(".pdf") for file in files)
        has_xmls = any(file.name.endswith(".xml") for file in files)
        has_images = any(file.name.endswith((".jpg", ".jpeg", ".png")) for file in files)
        has_txt = any(file.name.endswith(".txt") for file in files)
        
        # Conditionally create subheaders and progress bars
        if has_pdfs:
            st.subheader("PDF Progress")
            pdf_progress = st.progress(0)
            pdf_status = st.empty()
        
        if has_xmls:
            st.subheader("XML Progress")
            xml_progress = st.progress(0)
            xml_status = st.empty()
        
        if has_images:
            st.subheader("Image Progress")
            jpg_progress = st.progress(0)
            jpg_status = st.empty()

        if has_txt:
            st.subheader("Txt Progress")
            text_progress = st.progress(0)
            text_status = st.empty()
        
        # Track progress for each file type
        pdf_count = len([f for f in files if f.name.endswith(".pdf")])
        xml_count = len([f for f in files if f.name.endswith(".xml")])
        jpg_count = len([f for f in files if f.name.endswith((".jpg", ".jpeg", ".png"))])
        txt_count = len([f for f in files if f.name.endswith(".txt")])
        
        # Tracking variables for each file type
        pdf_processed = 0
        xml_processed = 0
        jpg_processed = 0
        txt_processed = 0
        
        for file_index, file in enumerate(files):
            file_name = file.name
            file_status.markdown(f"<p class='status-text'>Processing file {file_index + 1}/{total_files}: {file_name}</p>", unsafe_allow_html=True)
            
            if file_name.endswith(".pdf"):
                data = parse_pdfs(file, file_name, save_path, pdf_progress, pdf_status, overall_progress, file_index, total_files, batching)
                if data:
                    project["files"].append(data)
                    pdf_processed += 1
                    pdf_progress.progress(pdf_processed / pdf_count)
            
            elif file_name.endswith(".xml"):
                xml_data = parse_xml(file, file_name, save_path, xml_progress, xml_status)
                if xml_data:
                    project["xml"].append(xml_data)
                    xml_processed += 1
                    xml_progress.progress(xml_processed / xml_count)
            
            elif file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
                jpg_data = parse_jpgs(file, file_name, save_path, jpg_progress, jpg_status, overall_progress, file_index, total_files, batching)
                if jpg_data:
                    project["files"].append(jpg_data)
                    jpg_processed += 1
                    jpg_progress.progress(jpg_processed / jpg_count)

            elif file_name.endswith(".txt"):
                txt_data = parse_text(file, file_name, save_path, text_progress, text_status, overall_progress, file_index, total_files)
                if txt_data:
                    project["txt"].append(txt_data)
                    txt_processed += 1
                    text_progress.progress(txt_processed / txt_count)
            
            # Update overall progress
            overall_progress.progress((file_index + 1) / total_files)

            elapsed_time = time.time() - start_time
            time_status.markdown(f"<p class='status-text'>Elapsed Time: {elapsed_time:.2f} seconds</p>", unsafe_allow_html=True)

        save_sectors(sectors_data)

    end_time = time.time()
    total_time = end_time - start_time
    st.write(f"Total time taken: {total_time:.2f} seconds")
