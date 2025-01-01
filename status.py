import streamlit as st
import json
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import time
import nltk


nltk.download('punkt_tab')

client = OpenAI()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)


def retry_request(func, *args, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Request failed. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e


def translate(page_content, target_language, save_path):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant knowing all the languages."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"From this given text {page_content} detect in which"
                                f"language it is written. Only reply in one word saying"
                                f"which language it is and nothing else."
                    },
                ]
            }
        ]
    )
    language = response.choices[0].message.content.strip().lower()
    if language != target_language:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant knowing all the languages."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Translate this given text {page_content}"
                                    f"from {language} to {target_language}."
                                    f"Do not change any meaning or structure of the text."
                                    f"Do not add anything else except the translated text."
                        },
                    ]
                }
            ]
        )
        page_content = response.choices[0].message.content.strip()

    with open(save_path, "a") as file:
        file.write(page_content)
    return page_content



def check_project_batch_status(project):
    output = []
    overall_status = "completed"
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    save_path = f"{project['path']}/translated_file.txt"
    with open(save_path, "w") as file:
        pass

    if project["batching"]:
        for pdf_index, pdf in enumerate(project["files"]):
            with st.expander(f"Processing: {pdf['file_name']}"):
                for page_index, page in enumerate(sorted(pdf["pages"], key=lambda x: x['page_number'])):
                    try:
                        batch_job = retry_request(client.batches.retrieve, page["batch_id"])
                        status_placeholder.text(f"Processing {pdf['file_name']} - Page {page['page_number']}")

                        if batch_job.status == "failed":
                            overall_status = "failed"
                            st.error(f"Failed to process page {page['page_number']}")
                            break
                        elif batch_job.status in ["in_progress", "validating"]:
                            overall_status = "in_progress"
                            st.info(f"Page {page['page_number']} is still being processed")
                        elif batch_job.status == "completed":
                            file_content = client.files.content(batch_job.output_file_id)
                            response_content = json.loads(file_content.text)
                            page_content = ""
                            if 'response' in response_content and 'body' in response_content['response']:
                                page_content = response_content['response']['body'].get('choices', [{}])[0].get('message',
                                                                                                                {}).get(
                                    'content', '')
                                page_content = translate(page_content, project["target_language"], save_path)

                            chunks = splitter.split_text(page_content)
                            for chunk in chunks:
                                doc_obj = Document(
                                    page_content=chunk,
                                    metadata={
                                        "source": pdf['file_name'],
                                        "page_number": page['page_number'],
                                    }
                                )
                                output.append(doc_obj)

                            st.success(f"Successfully processed page {page['page_number']}")
                        else:
                            st.warning(
                                f"Unknown status for batch job of {pdf['file_name']} page {page['page_number']}: {batch_job.status}")
                            break
                    except Exception as e:
                        st.error(f"An error occurred while processing {pdf['file_name']} page {page['page_number']}: {e}")
                        overall_status = "failed"
                        break

                    progress = (pdf_index * len(pdf["pages"]) + page_index + 1) / (len(project["files"]) * len(pdf["pages"]))
                    progress_bar.progress(progress)
    else:
        for pdf_index, pdf in enumerate(project["files"]):
            with st.expander(f"Processing: {pdf['file_name']}"):
                for page_index, page in enumerate(sorted(pdf["pages"], key=lambda x: x['page_number'])):
                    status_placeholder.text(f"Processing {pdf['file_name']} - Page {page['page_number']}")
                    page_content = page["context"]
                    page_content = translate(page_content, project["target_language"], save_path)
                        
                    chunks = splitter.split_text(page_content)
                    for chunk in chunks:
                        doc_obj = Document(
                            page_content=chunk,
                            metadata={
                                "source": pdf['file_name'],
                                "page_number": page['page_number'],
                            }
                        )
                        output.append(doc_obj)


    status_placeholder.empty()

    all_files = project["xml"] + project["txt"]
    for temp in all_files:
        chunks = splitter.split_text(temp["page_content"])
        for chunk in chunks:
            chunk = translate(chunk, project["target_language"], save_path)

            doc_obj = Document(
                page_content=chunk,
                metadata={
                    "source": temp['file_name'],
                    "page_number": -1,
                }
            )
            output.append(doc_obj)

    return overall_status, output
