from Utilities import (load_data, create_sector, create_project,
                       load_project, delete_project, chat_interface,
                       delete_sector, project_status, update_KPIs, refresh_data)
import streamlit as st


# Blue #3079bd
# Green #45b653
# Orange #ffc900


def main():
    action = st.sidebar.radio(
        "Choose an action:",
        [
            "Create Sector",
            "Create Project",
            "Load Project",
            "Delete Project",
            "Delete Sector",
            "Check Project Status",
            "Update KPIs"
        ],
    )

    if action == "Create Sector":
        create_sector()
    elif action == "Create Project":
        create_project()
    elif action == "Load Project":
        try:
            data = load_data()
            sectors = list(data["sectors"].keys())
        except Exception as e:
            st.error("🚫 Unable to load data. Please check your permissions.")
            st.error(f"Details: {str(e)}")
            return
        refresh_data(sectors, data)
        sector = load_project()
        chat_interface(sector)
    elif action == "Delete Project":
        delete_project()
    elif action == "Delete Sector":
        delete_sector()
    elif action == "Check Project Status":
        project_status()
    elif action == "Update KPIs":
        update_KPIs()


if __name__ == "__main__":
    main()
