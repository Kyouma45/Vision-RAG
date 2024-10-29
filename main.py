from Utilities import (load_data, create_sector, create_project,
                       load_project, delete_project, chat_interface,
                       delete_sector, project_status)
import streamlit as st


# Blue #3079bd
# Green #45b653
# Orange #ffc900


def main():
    action = st.sidebar.radio(
        "Choose an action:",
        [
            "Create/Edit Sector",
            "Create Project",
            "Load Project",
            "Delete Project",
            "Delete Sector",
            "Check Project Status",
            "Generate CSV"
        ],
    )

    if action == "Create/Edit Sector":
        create_sector()
    elif action == "Create Project":
        create_project()
    elif action == "Load Project":
        sector = load_project()
        chat_interface(sector)
    elif action == "Delete Project":
        delete_project()
    elif action == "Delete Sector":
        delete_sector()
    elif action == "Check Project Status":
        project_status()
    elif action == "Generate CSV":
        st.error("Not Functional Yet (Please come back later)")


if __name__ == "__main__":
    main()
