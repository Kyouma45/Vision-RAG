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


def translate(page_content, target_language, save_path=None,):
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
    language = response.choices[0].message.content.strip().lower() # type: ignore
    print(f'Detected language is {language}')
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
        page_content = response.choices[0].message.content.strip() # type: ignore
        print(f'Translated Language: {page_content}')

    if save_path:
        with open(save_path, "a", encoding='utf-8') as file:
            file.write(page_content)
    return page_content


def summarize(content, max_chunk_size=8000, target_size=2000):
    try:
        print(f'Before Summarization: {len(content)} items')
        client = OpenAI()
        model = "gpt-4o"
        file_summaries = {}
        batch_size = 5
        
        # Step 1: Group content by file and combine
        file_content = {}
        for item in content:
            file_name = item.metadata.get("page", "Unknown")
            
            if file_name not in file_content:
                file_content[file_name] = []
            
            file_content[file_name].append(item.page_content)
        
        # Step 2: Process each file with iterative summarization
        for file_name, content_list in file_content.items():
            print(f"Processing file: {file_name}")
            
            # Combine all content for this file
            current_text = "\n\n".join(content_list)
            iteration = 1
            initial_tokens = len(current_text) // 4
            
            print(f"Initial text size: ~{initial_tokens} tokens")
            
            # Step 3: Always summarize if content exists, then check if we need to iterate
            if len(current_text.strip()) > 0:
                # First summarization pass
                print(f"Starting summarization - Current size: ~{len(current_text) // 4} tokens")
                
                # Break current text into chunks
                chunks = []
                for i in range(0, len(current_text), max_chunk_size):
                    chunks.append(current_text[i:i + max_chunk_size])
                
                print(f"Created {len(chunks)} chunks")
                
                # Summarize each chunk in batches
                summaries = []
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    
                    for j, chunk in enumerate(batch):
                        try:
                            print(f"Summarizing chunk {i+j+1}/{len(chunks)}")
                            batch_job = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant that provides concise, accurate summaries. "
                                                   "Preserve all important details while significantly reducing length. "
                                                   "Focus on key points, main ideas, and critical information."
                                                   "Give output in bullet points."
                                                   "Keep all the important dates or number or important information necessary"
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Summarize the following text, "
                                                   f"while preserving all important information:\n\n{chunk}"
                                    }
                                ],
                            )
                            
                            if batch_job.choices and batch_job.choices[0].message.content:
                                summary_content = batch_job.choices[0].message.content.strip()
                                summaries.append(summary_content)
                                print(f"Chunk {i+j+1} summarized: {len(chunk)} chars -> {len(summary_content)} chars")
                            else:
                                summaries.append(chunk)  # Keep original if API fails
                        
                        except Exception as e:
                            print(f"Failed to summarize chunk {i+j+1}: {str(e)}")
                            summaries.append(chunk)  # Keep original if summarization fails
                    
                    # Add delay between batches to respect rate limits
                    if i + batch_size < len(chunks):
                        time.sleep(0.5)  # Increased delay
                
                # Combine summaries
                current_text = "\n\n".join(summaries)
                iteration += 1
                
                print(f"After first pass: ~{len(current_text) // 4} tokens")
                
                # Step 4: Continue iterating if still too large
                while len(current_text) // 4 > target_size and iteration <= 5:
                    print(f"Iteration {iteration} - Current size: ~{len(current_text) // 4} tokens, target: {target_size}")
                    
                    # Break current text into chunks again
                    chunks = []
                    for i in range(0, len(current_text), max_chunk_size):
                        chunks.append(current_text[i:i + max_chunk_size])
                    
                    if len(chunks) == 1:
                        # If we can't break it down further, summarize as is
                        try:
                            print("Final summarization pass for single chunk")
                            batch_job = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant that provides concise, accurate summaries. "
                                                   "Preserve all important details while significantly reducing length. "
                                                   "Focus on key points, main ideas, and critical information."
                                                   "Give output in bullet points."
                                                   "Keep all the important dates or number or important information necessary"
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Summarize the following text, "
                                                   f"while preserving all important information:\n\n{current_text}"
                                    }
                                ],
                            )
                            
                            if batch_job.choices and batch_job.choices[0].message.content:
                                current_text = batch_job.choices[0].message.content.strip()
                        
                        except Exception as e:
                            print(f"Failed to summarize large text: {str(e)}")
                        break
                    
                    # Summarize each chunk again
                    summaries = []
                    
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        
                        for j, chunk in enumerate(batch):
                            try:
                                batch_job = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a helpful assistant that provides very concise summaries. "
                                                       "Further reduce length while keeping essential information."
                                        },
                                        {
                                            "role": "user",
                                            "content": f"Further summarize this text, reducing length by at least 50%:\n\n{chunk}"
                                        }
                                    ],
                                    max_tokens=400,
                                    temperature=0.3
                                )
                                
                                if batch_job.choices and batch_job.choices[0].message.content:
                                    summary_content = batch_job.choices[0].message.content.strip()
                                    summaries.append(summary_content)
                                else:
                                    summaries.append(chunk)
                            
                            except Exception as e:
                                print(f"Failed to summarize chunk {i+j+1} in iteration {iteration}: {str(e)}")
                                summaries.append(chunk)
                        
                        # Add delay between batches
                        if i + batch_size < len(chunks):
                            time.sleep(0.5)
                    
                    # Combine summaries for next iteration
                    current_text = "\n\n".join(summaries)
                    iteration += 1
                    
                    print(f"After iteration {iteration-1}: ~{len(current_text) // 4} tokens")
            
            # Store final result
            file_summaries[file_name] = [{
                "page": "All pages combined",
                "summary": current_text,
                "initial_size_tokens": initial_tokens,
                "final_size_tokens": len(current_text) // 4,
                "iterations_completed": iteration - 1,
                "compression_ratio": f"{(len(current_text) // 4) / initial_tokens * 100:.1f}%" if initial_tokens > 0 else "N/A"
            }]

            print(f'After summarization for {file_name}: {len(current_text)} characters, ~{len(current_text) // 4} tokens')
            
            try:
                st.success(f"Completed {file_name} - Final size: ~{len(current_text) // 4} tokens after {iteration-1} iterations")
            except:
                print(f"Completed {file_name} - Final size: ~{len(current_text) // 4} tokens after {iteration-1} iterations")

    except Exception as e:
        error_msg = f"Failed to generate summaries: {str(e)}"
        try:
            st.error(error_msg)
        except:
            print(error_msg)
        return {}

    return file_summaries


def check_project_batch_status(project):
    output = []
    overall_status = "completed"
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    save_path = f"{project['path']}/translated_file.txt"
    try:
        with open(save_path, "w") as file:
            pass
    except FileNotFoundError:
        print(f"Error: The file path '{save_path}' does not exist!!!")
    except IOError as e:
        print(f"An I/O error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

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
                            file_content = client.files.content(batch_job.output_file_id) # type: ignore
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
