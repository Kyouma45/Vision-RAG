import io
import numpy as np
import pandas as pd
import streamlit as st
import json
import shutil
import os
import time
import re
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pdf import parse_files
from vectorstore import create_vectorstore
from operator import itemgetter
from typing import List
from status import check_project_batch_status
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from typing import Optional, Dict, Any

load_dotenv()

if 'sectors' not in st.session_state:
    st.session_state.sectors = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_sector' not in st.session_state:
    st.session_state.current_sector = None
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'chain' not in st.session_state:
    st.session_state.chain = None

st.session_state.retriever = None
st.session_state.embeddings = OpenAIEmbeddings()
st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")
st.session_state.project_path = None
st.session_state.sector = None
st.session_state.project = None
st.session_state.selected_year = None

st.session_state.L1 = []
st.session_state.Theme = []
st.session_state.Topic = []
st.session_state.KPI_code = []
st.session_state.KPI = []
st.session_state.Prompt = []
st.session_state.Alias_1 = []
st.session_state.Alias_2 = []
st.session_state.Alias_3 = []
st.session_state.Alias_4 = []
st.session_state.Alias_5 = []
st.session_state.Format = []

pc = Pinecone(environment="us-east-1-aws")


def load_prompts(sector: Optional[str] = None) -> None:
    prompts_file_path = './data.json'

    try:
        # Ensure the prompts directory exists
        os.makedirs(os.path.dirname(prompts_file_path), exist_ok=True)

        # Attempt to open and read the JSON file
        with open(prompts_file_path, 'r') as file:
            all_prompts = json.load(file)
            all_prompts = all_prompts['sectors']

        if sector:
            if sector in all_prompts:
                # Load sector-specific prompts into session state
                keys = ["L1", "Theme", "Topic", "KPI_code", "KPI", "Prompt",
                        "Alias 1", "Alias 2", "Alias 3", "Alias 4", "Alias 5", "Format"]
                for key in keys:
                    st.session_state[key.replace(" ", "_")] = all_prompts[sector].get(key, "")

                with st.expander(f"Loaded prompts for sector: {sector}"):
                    st.write(st.session_state.Prompt)
            else:
                st.session_state.KPI = []
                st.session_state.Theme = []
                st.warning(f"No prompts found for sector: {sector}")
        else:
            st.info("No sector specified. Prompts not loaded.")

    except FileNotFoundError:
        st.error(f"Prompts file not found at {prompts_file_path}. Please ensure the file exists.")
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in {prompts_file_path}. Please check the file format.")
    except PermissionError:
        st.error(f"Permission denied when trying to access {prompts_file_path}. Check file permissions.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

    # Initialize session state variables if they don't exist
    default_keys = ["L1", "Theme", "Topic", "KPI_code", "KPI",
                    "Alias_1", "Alias_2", "Alias_3", "Alias_4", "Alias_5", "Format"]
    for key in default_keys:
        if key not in st.session_state:
            st.session_state[key] = "" if key != "KPI" else []


def load_sectors(json_file_path='sectors.json'):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            sectors_data = json.load(f)
    else:
        sectors_data = {"sectors": []}
    return sectors_data


def load_data() -> Dict[str, Any]:
    try:
        with open('data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"sectors": {}}
    except json.JSONDecodeError:
        st.error("Error decoding data.json. File may be corrupted.")
        return {"sectors": {}}


def save_data(data: Dict[str, Any]) -> None:
    try:
        with open('data.json', 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        st.error("Error saving data: {e}. Please check file permissions.")


def save_sectors(sectors_data, json_file_path='sectors.json'):
    try:
        with open(json_file_path, 'w') as f:
            json.dump(sectors_data, f, indent=2)
        return True
    except IOError as e:
        st.error(f"Error saving sectors data: {e}")
        return False


def to_excel(df):
    output = io.BytesIO()
    workbook = Workbook()
    sheet = workbook.active

    # Convert lists in the DataFrame to strings
    df = df.map(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

    for r in dataframe_to_rows(df, index=False, header=True):
        sheet.append(r)

    workbook.save(output)
    return output.getvalue()


def extract_text_from_jsonl(jsonl_path):
    extracted_text = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Assuming the text is in the 'content' field of the last message
            if 'messages' in data['body'] and data['body']['messages']:
                last_message = data['body']['messages'][-1]
                if 'content' in last_message:
                    extracted_text.append(last_message['content'])
    return '\n\n'.join(extracted_text)


def compress_parsed_pdfs(project_dir: Path):
    files_to_delete = [f for f in project_dir.glob('*') if not f.suffix in ('.faiss', '.pkl')]

    progress_bar = st.progress(0)
    for i, file_path in enumerate(files_to_delete):
        try:
            os.remove(file_path)
        except Exception as e:
            st.warning(f"Error deleting file {file_path}: {e}")

        progress = (i + 1) / len(files_to_delete)
        progress_bar.progress(progress)
        time.sleep(0.1)  # Small delay to make progress visible

    progress_bar.progress(1.0)


def validate_index_name(index_name: str) -> str:
    """
    Validate and format index name according to Pinecone requirements

    Args:
        index_name: Original index name

    Returns:
        Formatted valid index name
    """
    # Convert to lowercase
    index_name = index_name.lower()

    # Replace invalid characters with hyphens
    index_name = re.sub(r'[^a-z0-9-]', '-', index_name)

    # Remove leading/trailing hyphens
    index_name = index_name.strip('-')

    # Ensure name isn't too long (Pinecone has a 45 character limit)
    if len(index_name) > 45:
        index_name = index_name[:45].rstrip('-')

    # Ensure name isn't empty
    if not index_name:
        index_name = 'default-index'

    return index_name


def create_sector():
    df = pd.DataFrame({
        "L1": [], "Theme": [], "Topic": [], "KPI_code": [], "KPI": [], "Prompt": [],
        "Alias 1": [], "Alias 2": [], "Alias 3": [], "Alias 4": [], "Alias 5": [],
        "Format": [],
    })
    st.subheader("Create/Edit Sector")

    st.warning("Please follow this template")
    st.dataframe(df)

    sector_name = st.text_input("Enter sector name")
    csv_file = st.file_uploader("Upload KPIs (optional)", type=["csv"])

    if st.button("Apply", key="create_sector"):
        if not sector_name:
            st.error("Please enter a sector name")
            return

        data = load_data()

        if sector_name in data["sectors"]:
            projects = data["sectors"][sector_name]["projects"]
        else:
            projects = {}

        new_sector = {
            "projects": projects,
            "requests_parsing": 0,
            "requests_prompts": 0,
            "L1": [], "Theme": [], "Topic": [], "KPI_code": [], "KPI": [], "Prompt": [],
            "Alias 1": [], "Alias 2": [], "Alias 3": [], "Alias 4": [], "Alias 5": [],
            "Format": [],
        }

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                required_columns = ["L1", "Theme", "Topic", "KPI_code", "KPI", "Prompt",
                                    "Alias 1", "Alias 2", "Alias 3", "Alias 4", "Alias 5", "Format"]

                if not all(col in df.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    st.error(f"CSV is missing required columns: {', '.join(missing_cols)}")
                    return

                for col in required_columns:
                    new_sector[col] = df[col].tolist()
            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty.")
                return
            except pd.errors.ParserError:
                st.error("Error parsing the CSV file. Please check the file format.")
                return
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
                return

        data["sectors"][sector_name] = new_sector
        save_data(data)

        try:
            os.makedirs(f"./sectors/{sector_name}", exist_ok=True)
        except OSError as e:
            st.error(f"Error creating sector directory: {str(e)}")
            return

        st.success(f"Sector '{sector_name}' created successfully")
        st.balloons()


def create_project():
    st.subheader("Create a New project")
    data = load_data()
    sectors = list(data["sectors"].keys())
    sorted_sectors = sorted(sectors)
    sector_name = st.selectbox("Select a sector", sorted_sectors)
    load_prompts(sector_name)
    if sector_name:
        project_name = st.text_input("Enter project name")
        input_files = st.file_uploader("Upload files", type=["pdf", "xml"], accept_multiple_files=True)
        current_year = datetime.now().year
        years = list(range(current_year, current_year - 5, -1))
        default_index = years.index(current_year)
        st.session_state.selected_year = st.selectbox("Select year", years, index=default_index)
        if st.button("Create project"):
            if project_name and input_files:
                if project_name not in data["sectors"][sector_name]["projects"].keys():
                    project_dir = Path(f"./sectors/{sector_name}/{project_name}")
                    project_dir.mkdir(parents=True, exist_ok=True)

                    for i, file in enumerate(input_files, start=1):
                        save_path = project_dir / f"{i}.pdf"
                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())

                    st.write("Parsing Files...")
                    parse_files(input_files, str(project_dir), str(sector_name), str(project_name),
                                st.session_state.selected_year)

                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    st.success(f'All batches created successfully at {current_time}')
                else:
                    st.sidebar.error(f"project '{project_name}' already exists")
            else:
                st.sidebar.error("Please enter a project name and upload PDF files")


def load_project():
    st.sidebar.subheader("Load a project")
    data = load_data()
    sectors = list(data["sectors"].keys())
    sector_name = st.sidebar.selectbox("Select a sector", sectors)
    load_prompts(sector_name)
    if sector_name:
        projects = data["sectors"][sector_name]["projects"].keys()
        if not projects:
            st.write("No projects available for this sector.")
            return
        project_name = st.sidebar.selectbox("Select a project", projects)
        st.write(
            f"Prompts hit count for \"{project_name}\": {data['sectors'][sector_name]['projects'][project_name]['requests_prompts']}")
        st.write(f"Total Prompts (Sector-Wise): {data['sectors'][sector_name]['requests_prompts']}")
        if st.sidebar.button("Load project"):
            if project_name:
                st.session_state.current_sector = sector_name
                st.session_state.current_project = project_name
                st.session_state.selected_year = data['sectors'][sector_name]['projects'][project_name]['year']
                st.session_state.chat_history = []
                index_name = f"./sectors/{st.session_state.current_sector}/{st.session_state.current_project}".lower()
                index_name = validate_index_name(index_name)
                index = pc.Index(index_name)
                # try:
                if True:
                    vector_store = PineconeVectorStore(index=index,
                                                       embedding=st.session_state.embeddings
                                                       )
                    st.session_state.retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                                                           search_kwargs={"k": 4,
                                                                                          "score_threshold": 0.5},
                                                                           )
                    st.session_state.project_path = str(f"./sectors/{sector_name}/{project_name}")
                    st.session_state.sector = sector_name
                    st.session_state.project = project_name
                    st.sidebar.success(f"project '{project_name}' loaded successfully")
                    return sector_name
                # except Exception as e:
                #     st.sidebar.error(f"Error loading project '{project_name}': {str(e)}")
            else:
                st.sidebar.error("Please select a project")


def delete_project():
    st.subheader("Delete projects")
    data = load_data()
    sectors = list(data["sectors"].keys())
    sector_name = st.selectbox("Select a sector", sectors)
    temp = load_sectors()

    if sector_name:
        projects = data["sectors"][sector_name]["projects"]
        selected_projects = st.multiselect("Select projects to delete", projects)

        if st.button("Delete selected projects"):
            if selected_projects:
                for project_name in selected_projects:
                    try:
                        shutil.rmtree(f'./sectors/{sector_name}/{project_name}')
                        del data["sectors"][sector_name]["projects"][project_name]
                        temp[sector_name]["projects"] = [project for project in temp[sector_name]['projects'] if
                                                         project['project_name'] != project_name]
                        index_name = f"./sectors/{sector_name}/{project_name}".lower()
                        index_name = validate_index_name(index_name)
                        pc.delete_index(index_name)

                        if st.session_state.get('current_project') == project_name:
                            st.session_state.current_sector = None
                            st.session_state.current_project = None
                            st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"Error deleting project '{project_name}': {e}")
                        continue

                save_sectors(temp)
                save_data(data)
                st.success(f"Selected projects have been removed: {', '.join(selected_projects)}")
            else:
                st.error("Please select at least one project to delete")


def delete_sector():
    st.subheader("Delete a Sector")

    try:
        data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    sectors = list(data["sectors"].keys())
    sector_name = st.selectbox("Select sectors to delete", sectors)

    try:
        temp = load_sectors()
    except Exception as e:
        st.error(f"Error loading sectors: {str(e)}")
        return

    if st.button("Delete Sector", key="delete_sector"):
        if sector_name:
            try:
                with st.spinner(f"Deleting sector '{sector_name}'..."):
                    sector_path = f'./sectors/{sector_name}'

                    for project_name in data['sectors'][sector_name]["projects"].keys():
                        index_name = f"./sectors/{sector_name}/{project_name}".lower()
                        index_name = validate_index_name(index_name)
                        pc.delete_index(index_name)

                    # Check if the directory exists
                    if not os.path.exists(sector_path):
                        raise FileNotFoundError(f"Sector directory '{sector_path}' does not exist.")

                    # Attempt to delete the directory
                    try:
                        shutil.rmtree(sector_path)
                    except Exception as e:
                        st.error(f"Error deleting directory {sector_path}: {str(e)}", exc_info=True)
                        raise

                    # Update data structures
                    if sector_name in data["sectors"]:
                        del data["sectors"][sector_name]
                    if sector_name in temp:
                        del temp[sector_name]
                    save_sectors(temp)
                    save_data(data)

                    time.sleep(1)  # Simulate processing time

                st.success(f"Sector '{sector_name}' has been removed")
                if st.session_state.get('current_sector') == sector_name:
                    st.session_state.current_sector = None
                    st.session_state.current_project = None
                    st.session_state.chat_history = []
            except FileNotFoundError as e:
                st.error(f"Error: {str(e)}")
                st.error(f"FileNotFoundError: {str(e)}")
            except PermissionError as e:
                st.error(f"Permission denied: Unable to delete sector '{sector_name}'. Check file permissions.")
                st.error(f"PermissionError: {str(e)}")
            except KeyError as e:
                st.error(f"Data inconsistency: {str(e)}")
                st.error(f"KeyError: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error deleting sector: {str(e)}")
                st.error(f"Unexpected error: {str(e)}", exc_info=True)
        else:
            st.error("Please select a sector to delete")


def chat_interface(sector):
    # Check if the retriever is loaded
    if st.session_state.get('retriever') is None:
        st.error("Please load a project.")
        return

    # Check if the LLM is initialized
    if st.session_state.get('llm') is None:
        st.error("Language model not initialized. Please check your configuration.")
        return

    # Clear previous chat display
    st.empty()
    data = load_data()

    try:
        # Debugging: Log LLM and retriever status
        st.success("LLM and retriever are initialized.")

        # Create the chain
        if st.session_state.retriever is None:
            st.error("Please load a project first.")
            return

        prompt = ChatPromptTemplate.from_messages([
            (
                "system", "You are an assistant for an ESG firm. Use the data that you are provided from their "
                          "clients to answer the queries in a manner that a consultant"
                          "would respond to these questions. If the"
                          "format of the answer is described use that format to answer the question else answer the"
                          "question in a concise manner in three to five sentences. "
                          "Keep the answer concise.\n\nContext: {context}"),
            ("human", "{question}"),
        ])

        def format_docs(docs: List[Document]) -> str:
            formatted = [
                (  # f"Page Number: {doc.metadata.get('page', 'N/A')}\n"
                    f"Page Content: {doc.page_content}")
                for doc in docs
            ]
            return "\n\n" + "\n\n".join(formatted)

        format_doc = itemgetter("docs") | RunnableLambda(format_docs)
        answer = prompt | st.session_state.llm | StrOutputParser()
        chain = (
            RunnableParallel(question=RunnablePassthrough(), docs=st.session_state.retriever)
            .assign(context=format_doc)
            .assign(answer=answer)
            .pick(["answer", "docs"])
        )

        # Debugging: Confirm the chain creation
        st.success("Chain successfully created.")
    except Exception as e:
        st.error(f"Error creating chain: {str(e)}")
        st.error("Please check your configuration and try again.")
        return

    if not os.path.isfile(f"{st.session_state.project_path}/{st.session_state.project}.csv") or st.button(
            "Generate answers", key="generate_csv"):
        answers_n = []
        page_number_n = []
        answers_n_1 = []
        page_number_n_1 = []
        unit = []

        progress_bar = st.progress(0)
        percent_complete = st.empty()

        with (st.spinner('Processing prompts...')):
            start_time = time.time()
            total_prompts = len(st.session_state.KPI)

            # Year n
            for i, prompt in enumerate(st.session_state.Prompt):
                try:
                    response = None
                    for m in range(1, 6):
                        alias = getattr(st.session_state, f"Alias_{m}")[i]

                        if alias is None or (isinstance(alias, float) and np.isnan(alias)):
                            break

                        template_string = prompt
                        prompt_template = ChatPromptTemplate.from_template(template_string)
                        formatted_prompt = prompt_template.format(
                            alias=alias,
                            year=str(st.session_state.selected_year) + "-" + str(st.session_state.selected_year - 1)[
                                                                             -2:] + " or current reporting year"
                        )

                        response = chain.invoke(formatted_prompt)

                        # Update request counts
                        data["sectors"][st.session_state.sector]["requests_prompts"] += 1
                        data["sectors"][st.session_state.sector]['projects'][st.session_state.project][
                            'requests_prompts'] += 1

                        if response['answer'].lower().strip() != "not available":
                            break

                    if response['answer'].lower().strip() != "not available" and st.session_state.Format[
                        i].lower().strip() in ['number', 'currency']:
                        t = 0
                        while t < len(response['answer']) and response['answer'][t] != ' ':
                            t += 1
                        unit.append(response['answer'][t:].strip())
                    else:
                        unit.append(None)

                    answer = response.get('answer', 'No answer provided') if response else 'Not available'
                    if answer != 'Not Available':
                        t = 0
                        while t < len(answer) and answer[t] != ' ':
                            t += 1
                        answer = answer[:t].strip()
                    answers_n.append(answer)

                    docs = response.get('docs', []) if response else []
                    page_nums = []
                    for doc in docs:
                        if hasattr(doc, 'metadata'):
                            page_nums.append({
                                doc.metadata.get('source', 'N/A'):
                                    doc.metadata.get('page_number', 'N/A')
                            })
                        else:
                            page_nums.append(['N/A', 'N/A'])

                    page_number_n.append(page_nums if page_nums else [{'N/A', 'N/A'}])

                except Exception as e:
                    st.error(f"Error processing prompt {i + 1}: {str(e)}")
                    answers_n.append(f"Error: {str(e)}")
                    page_number_n.append([{'Error': 'N/A'}])
                    unit.append(None)

                # For year n-1
                try:
                    response = None
                    for m in range(1, 6):
                        data["sectors"][st.session_state.sector]["requests_prompts"] += 1
                        data["sectors"][st.session_state.sector]['projects'][st.session_state.project][
                            'requests_prompts'] += 1

                        alias = getattr(st.session_state, f"Alias_{m}")[i]

                        if alias is None or (isinstance(alias, float) and np.isnan(alias)):
                            break

                        template_string = (
                            st.session_state.Prompt[i]
                        )
                        prompt_template = ChatPromptTemplate.from_template(template_string)
                        formatted_prompt = prompt_template.format(
                            alias=alias,
                            year=str(st.session_state.selected_year - 1) + "-" + str(
                                st.session_state.selected_year - 2)[-2:] + " or previous reporting year"
                        )

                        response = chain.invoke(formatted_prompt)

                        if response['answer'].lower().strip() != "not available":
                            break

                    answer = response.get('answer', 'No answer provided') if response else 'Not available'
                    if answer != 'Not Available':
                        t = 0
                        while t < len(answer) and answer[t] != ' ':
                            t += 1
                        answer = answer[:t].strip()
                    answers_n_1.append(answer)

                    docs = response.get('docs', []) if response else []
                    page_nums = []
                    for doc in docs:
                        if hasattr(doc, 'metadata'):
                            page_nums.append({
                                doc.metadata.get('source', 'N/A'):
                                    doc.metadata.get('page_number', 'N/A')
                            })
                        else:
                            page_nums.append({'N/A': 'N/A'})

                    page_number_n_1.append(page_nums if page_nums else [{'N/A': 'N/A'}])

                except Exception as e:
                    st.error(f"Error processing prompt {i + 1}: {str(e)}")
                    answers_n_1.append(f"Error: {str(e)}")
                    page_number_n_1.append([{'Error': 'N/A'}])

                progress = (i + 1) / total_prompts
                progress_bar.progress(progress)
                percent_complete.text(f"{progress:.1%} completed")

            # Calculate total time taken
            total_time = time.time() - start_time
        st.write(f"Total time taken: {total_time:.2f} seconds")

        df = pd.DataFrame({
            "L1": st.session_state.L1,
            "Theme": st.session_state.Theme,
            "Topic": st.session_state.Topic,
            "KPI Code": st.session_state.KPI_code,
            'KPI': st.session_state.KPI,
            'Format': st.session_state.Format,
            f'Year {st.session_state.selected_year}-{st.session_state.selected_year - 1}': answers_n,
            f'Page Numbers {st.session_state.selected_year}-{st.session_state.selected_year - 1}': page_number_n,
            f'Year {st.session_state.selected_year - 1}-{st.session_state.selected_year - 2}': answers_n_1,
            f'Page Numbers {st.session_state.selected_year - 1}-{st.session_state.selected_year - 2}': page_number_n_1,
            'Unit': unit
            # 'Source': file,

        })

        df.to_csv(f"{st.session_state.project_path}/{st.session_state.project}.csv", index=False)
    else:
        df = pd.read_csv(f"{st.session_state.project_path}/{st.session_state.project}.csv")

    save_data(data)
    if df.shape[0] > 0:
        st.dataframe(
            df.drop(columns=[f'Page Numbers {st.session_state.selected_year}-{st.session_state.selected_year - 1}',
                             f'Page Numbers {st.session_state.selected_year - 1}-{st.session_state.selected_year - 2}']).head())
    else:
        st.write("No Predetermined KPIs!!!")

    # Create download buttons
    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        csv_bytes = csv.encode()
        st.download_button(
            label="Download as CSV",
            data=csv_bytes,
            file_name=f"{st.session_state.sector}_{st.session_state.project}.csv",
            mime="text/csv",
        )

    with col2:
        excel_bytes = to_excel(df)
        st.download_button(
            label="Download as Excel",
            data=excel_bytes,
            file_name=f"{st.session_state.sector}_{st.session_state.project}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    file_path = f"{st.session_state.project_path}/temp.csv"

    # Load or create the DataFrame
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Ensure 'Page Numbers' is stored as a list
        df['Page Numbers'] = df['Page Numbers'].apply(eval)
    else:
        df = pd.DataFrame({
            'Prompt': [],
            'Response': [],
            'Page Numbers': []
        })
        df.to_csv(file_path, index=False)

    # Chat input
    user_input = st.text_input("Ask a question:", key="chat_input", placeholder="Type your question here...")
    if st.button("Send", key="send_message"):
        if user_input:
            if st.session_state.get('retriever') is None:
                st.error("Document retriever not initialized. Please load a project first.")
            else:
                try:
                    response = chain.invoke(user_input)
                    data["sectors"][st.session_state.sector]["requests_prompts"] += 1
                    data["sectors"][st.session_state.sector]['projects'][st.session_state.project][
                        'requests_prompts'] += 1

                    st.markdown(f""" <div style="background-color: #1E3A5F; color: #FFFFFF; padding: 10px; border-radius: 
                                5px; margin-bottom: 10px;"> <strong>Prompt:</strong> {user_input}
                                        </div>
                                        """, unsafe_allow_html=True)

                    st.markdown(
                        f'<div style="background-color: #1F3B1F; color: #FFFFFF; padding: 10px; border-radius: 5px; '
                        f'margin-bottom: 10px;"><strong>Assistant:</strong> {response["answer"]}</div>',
                        unsafe_allow_html=True)

                    page_nums = []
                    with st.expander("View Source"):
                        for i, doc in enumerate(response['docs']):
                            st.markdown(f"**Source {i + 1}:**")
                            page_num = doc.metadata.get('page_number', 'N/A')
                            st.markdown(f"Page Number: {page_num}")
                            st.markdown(f"File: {doc.metadata.get('source', 'N/A')}")
                            st.markdown("---")
                            page_nums.append(str(page_num))

                    # Update chat history
                    st.session_state.chat_history.extend([
                        HumanMessage(content=user_input),
                        AIMessage(content=response['answer'])
                    ])

                    # Update DataFrame
                    new_row = pd.DataFrame({
                        'Prompt': [user_input],
                        'Response': [response['answer']],
                        'Page Numbers': [page_nums]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)

                    # Save updated DataFrame
                    df.to_csv(file_path, index=False)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please try reloading the project or contact support if the issue persists.")

    save_data(data)

    # Add a button to clear chat history
    if st.button("Clear Chat History", key="clear_history"):
        st.session_state.chat_history = []
        st.rerun()


def project_status():
    st.markdown('<p class="sidebar-text">Project Status Checker</p>', unsafe_allow_html=True)

    sectors_data = load_sectors()
    data = load_data()

    # Select sector
    sector_names = sorted(sectors_data.keys())
    selected_sector = st.selectbox("Select a sector:", sector_names)
    selected_sector_data = sectors_data.get(selected_sector)

    if not selected_sector_data:
        st.error("Selected sector not found.")
        return

    # Select project within the sector
    project_names = sorted(project["project_name"] for project in selected_sector_data["projects"])
    if not project_names:
        st.warning(f"No projects found in the {selected_sector} sector.")
        return

    selected_project = st.selectbox("Select a project:", project_names)

    if st.button("Check Status"):
        # Find the selected project
        project_data = next(
            (project for project in selected_sector_data["projects"] if project["project_name"] == selected_project),
            None)

        if not project_data:
            st.error("Selected project not found.")
            return

        project_dir = project_data["path"]
        time_created = project_data.get("time_created", 0)
        st.write(f"Project created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_created))}")

        status, text = check_project_batch_status(project_data)
        st.session_state.selected_year = project_data["year"]

        if status == "completed":
            st.success(f"Project '{selected_project}' is ready!")

            project_dir_path = Path(f"./sectors/{selected_sector}/{selected_project}".lower())
            st.write("Creating vector store...")
            create_vectorstore(text, str(project_dir_path))

            st.write("Compressing Parsed PDFs...")
            compress_parsed_pdfs(project_dir_path)

            end_time = time.time()

            sectors_data[selected_sector]["projects"] = [project for project in
                                                         sectors_data[selected_sector]["projects"]
                                                         if
                                                         project["project_name"] != selected_project]
            data["sectors"][selected_sector]["projects"][selected_project] = {
                "requests_prompts": 0,
                "year": st.session_state.selected_year
            }

            save_sectors(sectors_data)
            save_data(data)

            st.success(f"Project '{selected_project}' created successfully")
            st.write(f"Total Time Taken: {(end_time - time_created) / 3600:.2f} Hrs")

        elif status == "failed":
            st.error(f"Project '{selected_project}' has failed!")
        else:
            st.warning(f"Project '{selected_project}' is in progress!")
