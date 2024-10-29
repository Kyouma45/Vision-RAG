import streamlit as st
import json
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import time

client = OpenAI()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
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


def check_project_batch_status(project):
    output = []
    overall_status = "completed"
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

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

    status_placeholder.empty()

    for xml in project["xml"]:
        chunks = splitter.split_text(xml["page_content"])
        for chunk in chunks:
            doc_obj = Document(
                page_content=chunk,
                metadata={
                    "source": xml['file_name'],
                    "page_number": -1,
                }
            )
            output.append(doc_obj)

    return overall_status, output
