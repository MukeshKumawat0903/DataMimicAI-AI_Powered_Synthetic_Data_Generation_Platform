import streamlit as st

def show_feature_placeholders():
    """
    Displays the DataMimicAI Roadmap, marking implemented and upcoming features.
    Use in a dedicated tab or as a sidebar section.
    """
    with st.expander("🚧 Enhanced Data Generation Capabilities", expanded=False):
        st.markdown("""
        - **Conditional Data Synthesis:** Generate for specific segments  
        - **Multi-table Synthesis:** Related tables (CRM, finance, etc.)  
        - **Time-Series Pattern Controls:** Seasonality, trends  
        - **Data Quality Score:** Quantitative similarity metrics  
        """)
        st.info("Advanced generation features will let you customize and assess datasets.")

    with st.expander("🚧 Business Templates", expanded=False):
        st.markdown("""
        - **Industry Templates:** Domain schemas (finance, health, retail, energy, etc.)  
        - **Scenario Simulators:** Business logic wizards  
        - **Guided Playbooks:** Step-by-step business use cases  
        """)
        st.info("Domain templates and business simulation coming soon.")

    with st.expander("🚧 Advanced Analytics & Model Evaluation", expanded=False):
        st.markdown("""
        - **ML Task Evaluation:** Test ML models on synthetic vs real  
        - **Drift & Feature Importance:** Automated reports  
        - **Explainability:** Key difference analysis  
        """)
        st.info("Synthetic data validation and ML comparison soon available.")

    with st.expander("🚧 Integration & Team Collaboration", expanded=False):
        st.markdown("""
        - **API/SDK:** Automate workflows  
        - **Multi-user Permissions:** RBAC and auditing  
        - **Cloud Export:** Parquet, SQL, BI connectors  
        """)
        st.info("APIs, multi-user, and cloud export features are planned.")