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
from vectorstore import create_vectorstore  # , hybrid_search, HybridRetriever
from operator import itemgetter
from status import check_project_batch_status
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from typing import Optional, Dict, Any, Tuple, List
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_cohere import CohereRerank
import chardet

# from pinecone_text.sparse import BM25Encoder

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


def load_prompts() -> None:
    required_columns = [
        "L1", "Theme", "Topic", "KPI_code", "KPI", "Prompt",
        "Alias 1", "Alias 2", "Alias 3", "Alias 4", "Alias 5", "Format"
    ]
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

    try:
        # Check if file exists
        file_path = Path("./KPIs.csv")
        if not file_path.exists():
            raise FileNotFoundError(f"KPIs.csv not found in {file_path.absolute()}")

        # Try to read the CSV file
        df = pd.read_csv(file_path)

        # Check if file is empty
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")

        # Verify all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")

        # Handle missing values
        df = df.fillna("")  # Replace NaN with empty string

        # Load data into session state
        for column in required_columns:
            column_key = column.replace(" ", "_")  # Convert "Alias 1" to "Alias_1"
            st.session_state[column_key] = df[column]

        # Optional: Display success message to user
        st.success("✅ Successfully loaded KPI prompts")

    except pd.errors.EmptyDataError as e:
        error_msg = "Error: The KPIs.csv file is empty"
        st.error(error_msg)

    except FileNotFoundError as e:
        error_msg = f"Warning: {str(e)}"
        st.warning(error_msg)

    except KeyError as e:
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)

    except pd.errors.ParserError as e:
        error_msg = "Error: Invalid CSV file format"
        st.error(error_msg)

    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        st.error(error_msg)


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
    files_to_delete = [f for f in project_dir.glob('*') if f.name != 'translated_file.txt' and not f.suffix in ('.faiss', '.pkl')]
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
    index_name = re.sub(r'[^a-z0-9-_]', '-', index_name)

    # Remove leading/trailing hyphens
    index_name = index_name.strip('-')

    # Ensure name isn't too long (Pinecone has a 45 character limit)
    if len(index_name) > 45:
        index_name = index_name[:45].rstrip('-')

    # Ensure name isn't empty
    if not index_name:
        index_name = 'default-index'

    return index_name


def refresh_data(sectors, data):
    with st.spinner("🔄 Refreshing Data..."):
        indexes = pc.list_indexes()
        sector_pinecone = []
        project_pinecone = []

        # Process Pinecone indexes
        for index in indexes:
            index_name = index["name"]
            sector = ""
            i = 0

            while i < len(index_name) and index_name[i] != "-":
                i += 1
            i += 1

            while i < len(index_name) and index_name[i] != "-":
                sector += index_name[i]
                i += 1
            i += 1

            sector_pinecone.append(sector)

            # Create sector if it doesn't exist
            if sector not in sectors:
                new_sector = {
                    "projects": {},
                    "requests_parsing": 0,
                    "requests_prompts": 0,
                }
                data["sectors"][sector] = new_sector

                try:
                    os.makedirs(f"./sectors/{sector}", exist_ok=True)
                except OSError as e:
                    st.error(f"🚫 Error creating sector directory: {str(e)}")
                    return

                save_data(data)

            project = index_name[i:-5]
            project_pinecone.append(project)
            st.session_state.selected_year = int(index_name[-4:])

            # Create project if it doesn't exist
            if project not in data["sectors"][sector]["projects"]:
                project_dir = Path(f"./sectors/{sector}/{project}")
                try:
                    os.makedirs(project_dir, exist_ok=True)
                except OSError as e:
                    st.error(f"🚫 Error creating project directory: {str(e)}")
                    return
                data["sectors"][sector]["projects"][project] = {
                    "requests_prompts": 0,
                    "year": st.session_state.selected_year
                }

        # Clean up obsolete sectors and projects
        for x in sectors:
            if x not in sector_pinecone:
                sector_path = f'./sectors/{x}'
                try:
                    shutil.rmtree(sector_path)
                except Exception as e:
                    st.error(f"🚫 Error removing sector {x}: {str(e)}")

                del data["sectors"][x]
            else:
                for y in list(data["sectors"][x]["projects"].keys()):
                    if y not in project_pinecone:
                        try:
                            shutil.rmtree(f'./sectors/{x}/{y}')
                        except Exception as e:
                            st.error(f"🚫 Error removing project {y}: {str(e)}")
                        del data["sectors"][x]["projects"][y]

        save_data(data)
        st.success("✅ Data synchronized successfully")


def update_KPIs():
    st.title("KPI Update Interface")

    # Create two columns for the header section
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        ### Instructions
        1. Download the template CSV file
        2. Fill in your KPI data following the template
        3. Upload your completed CSV file
        4. Click Submit to update the KPIs
        """)

    # Create template DataFrame with required columns
    template_df = pd.DataFrame({
        "L1": [], "Theme": [], "Topic": [], "KPI_code": [], "KPI": [], "Prompt": [],
        "Alias 1": [], "Alias 2": [], "Alias 3": [], "Alias 4": [], "Alias 5": [],
        "Format": [],
    })

    with col2:
        # Create download button for template
        st.download_button(
            label="📥 Download Template CSV",
            data=template_df.to_csv(index=False),
            file_name="kpi_template.csv",
            mime="text/csv"
        )

    with col3:
        # Add reset button
        if st.button("🔄 Reset KPIs", type="secondary"):
            try:
                # Save empty DataFrame with required columns
                template_df.to_csv("./KPIs.csv", index=False)
                st.success("✅ KPIs reset successfully!")
            except Exception as e:
                st.error(f"❌ Error resetting KPIs: {str(e)}")
                st.info("Please make sure you have write permissions in the directory.")

    # Display existing KPIs if present
    try:
        existing_kpis = pd.read_csv("./KPIs.csv")
        if not existing_kpis.empty:
            st.subheader("Current KPIs")
            st.dataframe(
                existing_kpis,
                use_container_width=True,
                height=200
            )

            # Show summary statistics for existing KPIs
            st.subheader("Current KPIs Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total KPIs", len(existing_kpis))
            with col2:
                st.metric("Themes", existing_kpis['Theme'].nunique())
            with col3:
                st.metric("Topics", existing_kpis['Topic'].nunique())

            # Add download button for existing KPIs
            st.download_button(
                label="📥 Download Current KPIs",
                data=existing_kpis.to_csv(index=False),
                file_name="current_KPIs.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.info("No existing KPIs file found.")
    except Exception as e:
        st.error(f"Error reading existing KPIs: {str(e)}")

    # Create a separator
    st.markdown("---")

    # File uploader with clear instructions
    st.subheader("Upload KPI Data")
    csv_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file containing KPI data. Make sure it follows the template format."
    )

    required_columns = [
        "L1", "Theme", "Topic", "KPI_code", "KPI", "Prompt",
        "Alias 1", "Alias 2", "Alias 3", "Alias 4", "Alias 5", "Format"
    ]

    if csv_file is not None:
        try:
            # Create a progress bar for file processing
            with st.spinner("Processing file..."):
                df = pd.read_csv(csv_file)

                # Check required columns
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ CSV is missing required columns: {', '.join(missing_cols)}")
                    st.info("Please use the template provided above.")
                    return

                if len(df) != 0:
                    # Display preview of the data
                    st.subheader("Data Preview")
                    st.dataframe(
                        df.head(5),
                        use_container_width=True,
                        height=200
                    )

                # Show summary statistics
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total KPIs", len(df))
                with col2:
                    st.metric("Themes", df['Theme'].nunique())
                with col3:
                    st.metric("Topics", df['Topic'].nunique())

                # Submit button with confirmation
                if st.button("Submit", type="primary"):
                    try:
                        df.to_csv("./KPIs.csv", index=False)
                        st.success("✅ KPIs successfully updated!")

                        # Provide download link for the updated file
                        st.download_button(
                            label="📥 Download Updated KPIs",
                            data=df.to_csv(index=False),
                            file_name="updated_KPIs.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"❌ Error saving file: {str(e)}")
                        st.info("Please make sure you have write permissions in the directory.")

        except pd.errors.EmptyDataError:
            st.error("⚠️ The uploaded CSV file is empty.")
            df.to_csv("./KPIs.csv", index=False)
        except pd.errors.ParserError:
            st.error("❌ Error parsing the CSV file. Please check the file format.")
        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")
            st.info("Please make sure the file is a valid CSV and try again.")


def create_sector():
    st.title("Create Sector")
    st.markdown("---")

    # Initial data loading with error handling
    try:
        data = load_data()
        existing_sectors = data.get("sectors", {})
    except Exception as e:
        st.error("🚫 Unable to load data. Please check your permissions and try again.")
        st.error(f"Details: {str(e)}")
        return

    # Show existing sectors summary
    with st.expander("📊 Existing Sectors Overview", expanded=False):
        if existing_sectors:
            for sector, info in existing_sectors.items():
                st.markdown(f"""
                    **{sector.title()}**
                    - Projects: {len(info['projects'])}
                    - Parsing Requests: {info['requests_parsing']}
                    - Prompt Requests: {info['requests_prompts']}
                """)
        else:
            st.info("No sectors exist yet. Create your first sector below!")

    # Create main form
    with st.form("create_sector_form"):
        st.markdown("### Sector Information")

        # Create input field with help text
        sector_name = st.text_input(
            "Sector Name",
            help="Enter a unique name for your sector (will be converted to lowercase)",
            placeholder="e.g., healthcare, technology, finance"
        ).strip().lower()

        # Add description field (optional)
        sector_description = st.text_area(
            "Sector Description (Optional)",
            help="Add a brief description of this sector's purpose",
            placeholder="Describe the purpose and scope of this sector..."
        )

        # Submit button
        submit_button = st.form_submit_button("Create Sector", type="primary")

    # Process form submission
    if submit_button:
        # Input validation
        if not sector_name:
            st.error("⚠️ Please enter a sector name")
            return

        # Validate sector name
        if not sector_name.isalnum() and not all(c.isalnum() or c == '-' for c in sector_name):
            st.error("⚠️ Sector name can only contain letters, numbers, and hyphens")
            return

        # Check if sector exists
        sector_exists = sector_name in existing_sectors
        if sector_exists:
            st.error("🚫 Sector already exists")
            return

        try:
            with st.spinner("🔄 Creating sector..."):
                # Create sector directory
                sector_path = Path(f"./sectors/{sector_name}")
                try:
                    sector_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    st.error(f"🚫 Error creating sector directory: {str(e)}")
                    return

                # Create new sector data
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                new_sector = {
                    "projects": {},
                    "requests_parsing": 0,
                    "requests_prompts": 0,
                    "description": sector_description,
                    "created_at": current_time
                }

                # Update data structure
                data["sectors"][sector_name] = new_sector

                # Save updated data
                try:
                    save_data(data)
                except Exception as e:
                    st.error(f"🚫 Error saving sector data: {str(e)}")
                    # Cleanup on save failure
                    if sector_path.exists():
                        shutil.rmtree(sector_path)
                    return

                # Show success message and details
                st.success(f"✅ Sector '{sector_name}' created successfully!")

                with st.expander("📋 Sector Details", expanded=True):
                    st.markdown(f"""
                        **Sector Information**
                        - Name: {sector_name}
                        - Status: Created
                        - Created At: {current_time}
                        - Projects: 0
                        {f"- Description: {sector_description}" if sector_description else ""}
                    """)

                # Add celebration animation
                st.balloons()

                # Brief delay for UI feedback
                time.sleep(0.5)

        except Exception as e:
            st.error("❌ An unexpected error occurred")
            st.error(f"Details: {str(e)}")

            # Cleanup on failure
            try:
                if sector_path.exists():
                    shutil.rmtree(sector_path)
            except Exception as cleanup_error:
                st.error(f"Failed to clean up sector directory: {str(cleanup_error)}")


def create_project():
    st.title("Create a New Project")

    # Initial data loading with error handling
    try:
        data = load_data()
        if not data.get("sectors"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.error("🚫 No sectors found. Please create a sector first.")
            with col2:
                if st.button("➕ Create New Sector", type="primary"):
                    st.switch_page("pages/create_sector.py")
            return
    except Exception as e:
        st.error("🚫 Unable to load data. Please check your permissions and try again.")
        st.error(f"Details: {str(e)}")
        return

    # Load prompts with error handling
    try:
        load_prompts()
    except Exception as e:
        st.warning("⚠️ Failed to load prompts. Some features may be limited.")
        st.warning(f"Details: {str(e)}")
        if st.button("🔄 Retry Loading Prompts"):
            st.rerun()

    # Create main container for better spacing
    with st.container():
        # Project information section
        st.markdown("### 📋 Project Information")

        # Create three columns for better layout
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            sorted_sectors = sorted(list(data["sectors"].keys()))
            sector_name = st.selectbox(
                "Select Sector",
                sorted_sectors,
                help="Choose the sector where you want to create the project"
            )

            # Show sector stats
            if sector_name:
                st.caption(f"Projects in sector: {len(data['sectors'][sector_name]['projects'])}")

        with col2:
            project_name = st.text_input(
                "Project Name",
                help="Enter a unique name ending with a 4-digit year (e.g., project2024)",
                placeholder="project 2024"
            ).strip().lower()

        with col3:
            target_language = st.text_input(
                "Target Language",
                help="Enter the target language for this project (e.g., English)",
                placeholder="English"
            ).strip().lower()

        # File upload section
        st.markdown("### 📁 Upload Files")

        # Create columns for upload area
        upload_col1, upload_col2 = st.columns([3, 1])

        with upload_col1:
            input_files = st.file_uploader(
                "Upload PDF/XML/JPEG/JPG/PNG/TXT Files",
                type=["pdf", "xml", "jpeg", "jpg", "png", "txt"],
                accept_multiple_files=True,
                help="Select one or more PDF/XML/JPEG/JPG/PNG files to include in the project"
            )

        with upload_col2:
            if input_files:
                st.metric("Files Selected", len(input_files))

        # Create action buttons
        col1, col2 = st.columns([2, 2])
        with col1:
            instant_submit_button = st.button("📂 Create Project", type="primary", use_container_width=True)
        with col2:
            submit_button = st.button("🚀 Create Batches", type="primary", use_container_width=True)


        if submit_button or instant_submit_button:
            # Validate inputs
            validation_errors = []

            if not project_name:
                validation_errors.append("⚠️ Please enter a project name")
            if not input_files:
                validation_errors.append("⚠️ Please upload at least one file")
            if project_name:
                try:
                    if not project_name[-4:].isdigit():
                        validation_errors.append("⚠️ Project name must end with a 4-digit year (e.g., project 2024)")
                except IndexError:
                    validation_errors.append("⚠️ Project name must be at least 4 characters long")

            # Display all validation errors if any
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return

            # Check for existing project
            if project_name in data["sectors"][sector_name]["projects"]:
                st.error(f"🚫 Project '{project_name}' already exists in sector '{sector_name}'")
                return

            # Set the year
            st.session_state.selected_year = project_name[-4:]
            project_name = project_name[:-4]
            project_name = project_name.strip()

            # Create project with progress tracking
            status_container = st.empty()
            with status_container.status("Creating project...", expanded=True) as status:
                try:
                    st.write("🔄 Setting up project structure...")
                    project_dir = Path(f"./sectors/{sector_name}/{project_name}")
                    project_dir.mkdir(parents=True, exist_ok=True)

                    # Save files with detailed progress
                    total_size = sum(file.size for file in input_files)
                    progress_bar = st.progress(0)

                    for i, file in enumerate(input_files, 1):
                        try:
                            st.write(f"📄 Processing: {file.name}")
                            save_path = project_dir / f"{i}_{file.name}"
                            with open(save_path, "wb") as f:
                                f.write(file.getbuffer())
                            progress_bar.progress(i / len(input_files))
                        except Exception as e:
                            st.error(f"❌ Error saving {file.name}: {str(e)}")

                    st.write("🔍 Parsing files...")
                    if submit_button:
                        parse_files(input_files, str(project_dir), str(sector_name), str(project_name),
                                    st.session_state.selected_year, True, target_language)
                    else:
                        parse_files(input_files, str(project_dir), str(sector_name), str(project_name),
                                    st.session_state.selected_year, False, target_language)

                    # Mark status as complete
                    status.update(label="✅ Project created successfully!", state="complete")

                    # Show success message and summary in the main flow
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    st.success("Project created successfully!")

                    # Display summary using columns instead of an expander
                    st.markdown("#### 📊 Project Summary")
                    sum_col1, sum_col2 = st.columns(2)
                    with sum_col1:
                        st.metric("Files Processed", len(input_files))
                        st.metric("Total Size", f"{total_size / 1024 / 1024:.1f} MB")
                    with sum_col2:
                        st.markdown("**Project Details**")
                        st.markdown(f"- 📁 Sector: {sector_name}")
                        st.markdown(f"- 📑 Project: {project_name}")
                        st.markdown(f"- ⏰ Created: {current_time}")

                    # Auto-refresh after short delay
                    time.sleep(0.5)

                except Exception as e:
                    status.update(label="❌ Error creating project", state="error")
                    st.error(f"An error occurred: {str(e)}")

                    # Cleanup on failure
                    try:
                        if project_dir.exists():
                            shutil.rmtree(project_dir)
                            st.info("🧹 Cleaned up temporary files")
                    except Exception as cleanup_error:
                        st.error(f"⚠️ Temporary files could not be removed: {str(cleanup_error)}")


def load_project():
    st.sidebar.subheader("📂 Load a project")

    try:
        data = load_data()
        sectors = list(data["sectors"].keys())
    except Exception as e:
        st.error("🚫 Unable to load data. Please check your permissions.")
        st.error(f"Details: {str(e)}")
        return

    # Project selection UI
    sector_name = st.sidebar.selectbox(
        "Select Sector",
        sectors,
        help="Choose a sector to view its projects"
    )

    if not sector_name:
        st.sidebar.warning("⚠️ Please select a sector")
        return

    try:
        load_prompts()
    except Exception as e:
        st.warning("⚠️ Failed to load prompts. Some features may be limited.")
        st.warning(f"Details: {str(e)}")

    if sector_name:
        projects = data["sectors"][sector_name]["projects"].keys()
        if not projects:
            st.info("📝 No projects available in this sector yet.")
            return

        project_name = st.sidebar.selectbox(
            "Select Project",
            projects,
            help="Choose a project to load"
        )

        if not project_name:
            st.sidebar.warning("⚠️ Please select a project")
            return

        # Load project button in sidebar
        if st.sidebar.button("📂 Load Project", type="primary", use_container_width=True):
            with st.spinner("🔄 Loading project..."):
                st.session_state.current_sector = sector_name
                st.session_state.current_project = project_name
                st.session_state.selected_year = data['sectors'][sector_name]['projects'][project_name]['year']
                st.session_state.chat_history = []
                index_name = f"./sectors/{st.session_state.current_sector}/{st.session_state.current_project}-{st.session_state.selected_year}"
                # bm25_encoder = BM25Encoder().default()
                index_name = validate_index_name(index_name)
                st.write("Index: {}".format(index_name))
                index = pc.Index(index_name)
                try:
                    vector_store = PineconeVectorStore(index=index,
                                                       embedding=st.session_state.embeddings,
                                                       text_key="text",
                                                       )
                    retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                                          search_kwargs={"k": 4,
                                                                         "score_threshold": 0.85},
                                                          )

                    if data["sectors"][sector_name]['projects'][project_name]['target_language'] == 'english':
                        compressor = CohereRerank(model="rerank-english-v3.0")
                        st.session_state.retriever = ContextualCompressionRetriever(
                            base_compressor=compressor, base_retriever=retriever
                        )
                    st.session_state.retriever = retriever
                    st.session_state.project_path = str(f"./sectors/{sector_name}/{project_name}")
                    os.makedirs(st.session_state.project_path, exist_ok=True)
                    st.session_state.sector = sector_name
                    st.session_state.project = project_name
                    st.sidebar.success(f"project '{project_name}' loaded successfully")
                    # Display project metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Project Prompt Count",
                            f"{data['sectors'][sector_name]['projects'][project_name]['requests_prompts']:,}",
                            help="Number of prompts used in this project"
                        )
                    with col2:
                        st.metric(
                            "Total Sector Prompts",
                            f"{data['sectors'][sector_name]['requests_prompts']:,}",
                            help="Total prompts used across all projects in this sector"
                        )

                    file_path = f"./sectors/{st.session_state.sector}/{st.session_state.project}/translated_file.txt"
                    try:
                        # Open the file in binary mode to detect encoding with chardet
                        with open(file_path, "rb") as file:
                            raw_data = file.read()
                            result = chardet.detect(raw_data)  # Detect the encoding
                            encoding = result['encoding']  # Get the detected encoding

                        # Open the file with the detected encoding
                        with open(file_path, "r", encoding=encoding) as file:
                            txt = file.read()

                        st.download_button(
                            label="Translated Txt",
                            data=txt,
                            file_name="Translated_File.txt",
                            mime="text/plain"
                        )

                    except FileNotFoundError:
                        st.error(f"Error: The file {file_path} does not exist.")
                    except IOError as e:
                        st.error(f"Error reading the file {file_path}: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

                    return sector_name
                except Exception as e:
                    st.sidebar.error(f"Error loading project '{project_name}': {str(e)}")


def delete_project():
    st.title("Delete Projects")
    st.warning(
        "⚠️ Warning: Deleting projects will permanently remove all associated data. This action cannot be undone.")

    try:
        data = load_data()
        if not data.get("sectors"):
            st.info("No sectors available.")
            return
    except Exception as e:
        st.error("🚫 Unable to load data. Please check your permissions and try again.")
        st.error(f"Details: {str(e)}")
        return

    # Create columns for better layout
    col1, col2 = st.columns([3, 1])

    with col1:
        sectors = sorted(list(data["sectors"].keys()))
        sector_name = st.selectbox(
            "Select sector",
            sectors,
            help="Choose the sector containing the projects you want to delete"
        )

    try:
        temp = load_sectors()
    except Exception as e:
        st.error("🚫 Error loading sector data. Please try again.")
        st.error(f"Details: {str(e)}")
        return

    if sector_name:
        try:
            projects = data["sectors"][sector_name]["projects"]
            if not projects:
                st.info(f"No projects available in sector '{sector_name}'.")
                return

            project_list = sorted(list(projects.keys()))
            selected_projects = st.multiselect(
                "Select projects to delete",
                project_list,
                help="You can select multiple projects to delete at once"
            )

            if st.button("Delete Selected Projects", type="primary"):
                if not selected_projects:
                    st.error("⚠️ Please select at least one project to delete")
                    return

                deletion_results = {"success": [], "failed": []}

                with st.spinner("🗑️ Deleting selected projects..."):
                    for project_name in selected_projects:
                        try:
                            # Store year before deletion for index removal
                            project_year = data['sectors'][sector_name]['projects'][project_name]['year']
                            project_path = os.path.join('./sectors', sector_name, project_name)

                            # Delete project directory
                            if os.path.exists(project_path):
                                shutil.rmtree(project_path)

                            # Delete project index
                            try:
                                index_name = os.path.join('./sectors', sector_name, project_name,
                                                          str(project_year))
                                index_name = validate_index_name(index_name.lower())
                                pc.delete_index(index_name)
                            except Exception as e:
                                st.warning(f"⚠️ Could not delete index for '{project_name}': {str(e)}")
                                return

                            # Update data structures
                            del data["sectors"][sector_name]["projects"][project_name]
                            temp[sector_name]["projects"] = [
                                project for project in temp[sector_name]['projects']
                                if project['project_name'] != project_name
                            ]

                            # Clear session state if needed
                            if st.session_state.get('current_project') == project_name:
                                st.session_state.current_sector = None
                                st.session_state.current_project = None
                                st.session_state.chat_history = []

                            deletion_results["success"].append(project_name)

                        except Exception as e:
                            deletion_results["failed"].append((project_name, str(e)))
                            st.error(f"❌ Error deleting project '{project_name}': {str(e)}")
                            continue

                    try:
                        # Save updated data
                        save_sectors(temp)
                        save_data(data)
                    except Exception as e:
                        st.error(f"🚫 Error saving data after deletion: {str(e)}")
                        return

                # Show results
                if deletion_results["success"]:
                    st.success(f"✅ Successfully deleted: {', '.join(deletion_results['success'])}")

                if deletion_results["failed"]:
                    st.error("❌ Failed to delete the following projects:")
                    for project, error in deletion_results["failed"]:
                        st.error(f"- {project}: {error}")

                time.sleep(3)  # Brief delay for UI feedback
                st.rerun()  # Refresh the page to show updated state

        except KeyError as e:
            st.error(f"🚫 Error accessing project data: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")


def delete_sector():
    st.title("Delete a Sector")

    # Add warning message about permanent deletion
    st.warning(
        "⚠️ Warning: Deleting a sector will permanently remove all associated projects and data. This action cannot "
        "be undone.")

    try:
        data = load_data()
        if not data.get("sectors"):
            st.info("No sectors available to delete.")
            return
    except Exception as e:
        st.error("🚫 Unable to load data. Please check your permissions and try again.")
        st.error(f"Details: {str(e)}")
        return

    # Create columns for better layout
    col1, col2 = st.columns([3, 1])

    with col1:
        sectors = sorted(list(data["sectors"].keys()))
        sector_name = st.selectbox(
            "Select sector to delete",
            sectors,
            help="Choose the sector you want to delete"
        )

    try:
        temp = load_sectors()
    except Exception as e:
        st.error("🚫 Error loading sector data. Please try again.")
        st.error(f"Details: {str(e)}")
        return

    # Add confirmation dialog
    if st.button("Delete Sector", key="delete_sector", type="primary"):
        if not sector_name:
            st.error("⚠️ Please select a sector to delete")
            return

        # try:
        if True:
            with st.spinner(f"🗑️ Deleting sector '{sector_name}'..."):
                sector_path = os.path.join('./sectors', sector_name)

                # Delete associated indexes first
                for project_name, project_data in data['sectors'][sector_name]["projects"].items():
                    try:
                        st.session_state.selected_year = project_data['year']
                        index_name = os.path.join('./sectors', sector_name, project_name,
                                                  str(st.session_state.selected_year)).lower()
                        index_name = validate_index_name(index_name)
                        pc.delete_index(index_name)
                    except Exception as e:
                        st.warning(f"⚠️ Warning: Could not delete index for project '{project_name}': {str(e)}")
                        continue

                # Verify directory exists
                if os.path.exists(sector_path):
                    try:
                        shutil.rmtree(sector_path)
                    except PermissionError:
                        st.error("🔒 Permission denied: Unable to delete sector files. Check your permissions.")
                    except Exception as e:
                        st.error(f"🚫 Error deleting sector files: {str(e)}")

                # Update data structures
                if sector_name in data["sectors"]:
                    del data["sectors"][sector_name]
                if sector_name in temp:
                    del temp[sector_name]
                save_sectors(temp)
                save_data(data)

                # Clear session state if needed
                if st.session_state.get('current_sector') == sector_name:
                    st.session_state.current_sector = None
                    st.session_state.current_project = None
                    st.session_state.chat_history = []

                st.success(f"✅ Sector '{sector_name}' has been successfully deleted")
                time.sleep(0.5)  # Brief delay for UI feedback
                st.rerun()  # Refresh the page to show updated state

        # except FileNotFoundError:
        #     st.error(f"🚫 Error: Sector directory not found at '{sector_path}'")
        # except PermissionError as e:
        #     st.error("🔒 Permission denied: Unable to delete sector. Please check your permissions.")
        #     st.error(f"Details: {str(e)}")
        # except Exception as e:
        #     st.error("❌ An unexpected error occurred while deleting the sector")
        #     st.error(f"Details: {str(e)}")


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
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", "You are an assistant for an ESG firm. Use the data that you are provided from their "
                          "clients to answer the queries in a manner that a consultant"
                          "would respond to these questions. If the"
                          "format of the answer is described use that format to answer the question else answer the"
                          "question in a concise manner in three to five sentences. "
                          "Keep the answer concise.\n\nContext: {context}"
                          "If no Context is given, tell user Not Available"),
            ("human", "{question}"),
        ])

        def format_docs(documents: List[Document]) -> str:
            formatted = [
                (  # f"Page Number: {document.metadata.get('page', 'N/A')}\n"
                    f"Page Content: {document.page_content}")
                for document in documents
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
    except Exception as e:
        st.error(f"Error creating chain: {str(e)}")
        st.error("Please check your configuration and try again.")
        return
    
    generate_answers = False
    if st.session_state.Prompt:
        generate_answers = st.button("Generate answers", key="generate_csv")

    if generate_answers:
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

            for i, prompt in enumerate(st.session_state.Prompt):
                # Year n
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

                        # time.sleep(30)

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

                        # time.sleep(30)

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

        try:
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
            })
            df.to_csv(f"{st.session_state.project_path}/{st.session_state.project}.csv", index=False)
        except Exception as e:
            st.error(f"Error: {str(e)}")

        try:
            save_data(data)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if os.path.exists(f"{st.session_state.project_path}/{st.session_state.project}.csv"):
        df = pd.read_csv(f"{st.session_state.project_path}/{st.session_state.project}.csv")
        if df.shape[0] > 0:
            st.dataframe(df)
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
                            st.markdown(f"Text: {doc.page_content}")
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
    st.title('Project Status Checker')

    # Initialize empty placeholders for error messages
    error_placeholder = st.empty()
    status_placeholder = st.empty()
    target_language = ""

    try:
        sectors_data = load_sectors()
        data = load_data()
    except Exception as e:
        error_placeholder.error(f"Error loading data: {str(e)}")
        return

    # Deleting the empty sectors if any
    keys_to_delete = [key for key, value in sectors_data.items() if len(value.get("projects", [])) == 0]
    for key in keys_to_delete:
        del sectors_data[key]

    # Add a container for selection controls
    with st.container():
        # Create two columns for sector and project selection
        col1, col2 = st.columns(2)

        with col1:
            # Select sector with error handling
            sector_names = sorted(sectors_data.keys()) if sectors_data else []
            if not sector_names:
                error_placeholder.error("No sectors found in the database.")
                return

            selected_sector = st.selectbox(
                "Select Sector",
                sector_names,
                help="Choose the sector to check project status",
                key="sector_selector"
            )

        selected_sector_data = sectors_data.get(selected_sector, {})
        if not selected_sector_data:
            error_placeholder.error(f"No data found for sector: {selected_sector}")
            return

        with col2:
            # Select project with error handling
            project_names = sorted(
                project["project_name"]
                for project in selected_sector_data.get("projects", [])
            )
            if not project_names:
                status_placeholder.warning(f"No active projects found in the {selected_sector} sector.")
                return

            selected_project = st.selectbox(
                "Select Project",
                project_names,
                help="Choose the project to check status",
                key="project_selector"
            )

    # Add a progress tracker
    progress_container = st.container()

    # Add a divider for better visual separation
    st.divider()

    # Use a container for the status check results
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            check_status = st.button(
                "Check Status",
                type="primary",
                use_container_width=True,
                help="Click to check the current status of the selected project"
            )

        if check_status:
            progress_text = "Operation in progress. Please wait..."
            my_bar = progress_container.progress(0, text=progress_text)

            # Find the selected project with error handling
            project_data = next(
                (p for p in selected_sector_data.get("projects", [])
                 if p["project_name"] == selected_project),
                None
            )

            if not project_data:
                error_placeholder.error("⚠️ Selected project data not found.")
                return

            my_bar.progress(25, text="Project data loaded...")

            # Display project information in a formatted container
            with st.expander("Project Details", expanded=True):
                time_created = project_data.get("time_created", 0)
                creation_date = time.strftime('%Y-%m-%d %H:%M:%S',
                                              time.localtime(time_created))

                details_col1, details_col2 = st.columns(2)
                with details_col1:
                    st.markdown("#### Project Information")
                    st.markdown(f"**Created:** {creation_date}")
                    st.markdown(f"**Year:** {project_data.get('year', 'N/A')}")
                with details_col2:
                    st.markdown("#### Location")
                    st.markdown(f"**Path:** {project_data.get('path', 'N/A')}")

            try:
                my_bar.progress(50, text="Checking project status...")
                status, text = check_project_batch_status(project_data)
                st.session_state.selected_year = project_data["year"]
                target_language = project_data["target_language"]

                if status == "completed":
                    project_dir_path = Path(f"./sectors/{selected_sector}/{selected_project}")
                    path = f"./sectors/{selected_sector}/{selected_project}"

                    my_bar.progress(75, text="Creating vector store...")
                    # Create vector store with progress indicator
                    with st.status("Creating vector store...") as status:
                        create_vectorstore(text, path+f"-{st.session_state.selected_year}")
                        status.update(label="Vector store created!", state="complete")

                    # Compress PDFs with progress indicator
                    with st.status("Compressing parsed PDFs...") as status:
                        compress_parsed_pdfs(project_dir_path)
                        status.update(label="PDFs compressed!", state="complete")

                    end_time = time.time()
                    processing_time = (end_time - time_created) / 3600

                    # Update project data
                    sectors_data[selected_sector]['projects'] = [
                        p for p in sectors_data[selected_sector]["projects"]
                        if p["project_name"] != selected_project
                    ]

                    data["sectors"][selected_sector]["projects"][selected_project] = {
                        "requests_prompts": 0,
                        "year": st.session_state.selected_year,
                        "target_language": target_language,
                    }

                    # Save updated data with error handling
                    try:
                        save_sectors(sectors_data)
                        save_data(data)
                    except Exception as e:
                        st.error(f"Error saving data: {str(e)}")
                        return

                    my_bar.progress(100, text="Process completed!")

                    # Success message with metrics
                    st.success("✅ Project processed successfully!")

                    file_path = f"./sectors/{selected_sector}/{selected_project}/translated_file.txt"
                    try:
                        # Open the file in binary mode to detect encoding with chardet
                        with open(file_path, "rb") as file:
                            raw_data = file.read()
                            result = chardet.detect(raw_data)  # Detect the encoding
                            encoding = result['encoding']  # Get the detected encoding

                        # Open the file with the detected encoding
                        with open(file_path, "r", encoding=encoding) as file:
                            txt = file.read()

                        # Display the file content in the app and provide a download button
                        st.title("Translated File")
                        st.download_button(
                            label="Download the file",
                            data=txt,
                            file_name="Translated_File.txt",
                            mime="text/plain"
                        )

                    except FileNotFoundError:
                        st.error(f"Error: The file {file_path} does not exist.")
                    except IOError as e:
                        st.error(f"Error reading the file {file_path}: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Processing Time", f"{processing_time:.2f} hrs")
                    with metric_col2:
                        st.metric("Status", "Completed")
                    with metric_col3:
                        st.metric("Files Processed", len(text) if isinstance(text, list) else 1)

                elif status == "failed":
                    my_bar.progress(100, text="Process failed!")
                    st.error("❌ Project processing failed!")
                else:
                    my_bar.progress(75, text="Project in progress...")
                    st.info("⏳ Project is still in progress...")

            except Exception as e:
                my_bar.progress(100, text="Error occurred!")
                st.error(f"Error processing project: {str(e)}")
                st.exception(e)
