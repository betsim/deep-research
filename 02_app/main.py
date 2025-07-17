import streamlit as st
import numpy as np
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
from _core.logger import custom_logger
from _core.prompts import INFO_TEXT
from _core.workflow import ResearchWorkflow
from _core.llm_processing import create_final_report
from _core.utils import get_model_and_workflow_config, create_docx_from_markdown
from _core.config import config


@st.cache_resource()
def load_data():
    """Load the decision data"""
    return pd.read_parquet(config["app"]["docs_file"])


@st.dialog("Deep-Research-App", width="large")
def info_dialog():
    st.markdown(INFO_TEXT)


def main():
    st.set_page_config(page_title="Deep Research", page_icon="🔍", layout="wide")

    # Sidebar for configuration
    with st.sidebar:
        st.title("🔍 Deep Research")
        st.markdown(
            "Recherchewerkzeug für eigene Dokumentsammlungen.\n\n:red[Achtung: Dies ist ein experimenteller Prototyp. Gib nur als öffentlich klassifizierte Daten als Fragen ein. Die Ergebnisse können fehlerhaft oder unvollständig sein. **Überprüfe die Ergebnisse immer.**] \n\n Die Bearbeitung kann einige Minuten dauern, abhängig von der Komplexität der Anfrage und der Anzahl der relevanten Dokumente.\n\n"
        )
        if st.button("Mehr zu dieser App"):
            info_dialog()

        # Load data
        if "docs" not in st.session_state:
            with st.spinner("Lade Dokumente..."):
                st.session_state.docs = load_data()

        if st.session_state.docs is None:
            st.error(
                "Dokumente konnten nicht geladen werden. Bitte überprüfe die Installation und stelle sicher, dass die Datei vorhanden ist."
            )
            return

        st.success(f"✅ {len(st.session_state.docs):,.0f} Dokumente geladen")
        st.markdown("---")

        if "fast_mode" not in st.session_state:
            st.session_state.fast_mode = False
        _ = st.checkbox(
            "Schneller Modus",
            help="Aktiviere den schnellen Modus für eine verkürzte Recherche mit weniger Suchanfragen und kleineren Modellen.",
            key="fast_mode",
        )
        custom_logger.info_console(f"Dev mode: {config['development']['enabled']}")
        custom_logger.info_console(f"Fast mode: {st.session_state.fast_mode}")

        if "iterative_workflow" not in st.session_state:
            st.session_state.iterative_workflow = False
        iterative_workflow = st.checkbox(
            "Iterative Recherche",
            help="Aktiviere den iterativen Rechercheprozess, um die Suche zu vertiefen. Dies erhöht die Anzahl der Suchanfragen und verlängert die Recherchezeit.",
            key="iterative_workflow",
        )

    user_query = st.text_area(
        "Gib deine Frage ein...",
        value="Was hat der Kantonsrat zu Steuern entschieden?",
        label_visibility="visible",
        key="user_input",
        max_chars=2000,
    )

    cols = st.columns([1, 1, 2])
    with cols[0]:
        # Process button
        start_button = st.button(
            "Recherche starten",
            type="primary",
            disabled=not st.session_state.user_input.strip(),
        )
    with cols[1]:
        # Reset button
        reset_button = st.button("Neue Recherche", type="secondary", key="reset_button")

    if start_button:
        # Set start time for logging and add to session state
        st.session_state.start_time = datetime.now()
        process_query(
            user_query.strip(),
            iterative_workflow,
            st.session_state.fast_mode,
        )
    if reset_button:
        st.session_state.clear()
        st.rerun()

    # Display results if available
    if "final_report" in st.session_state:
        display_results()
        if "interaction_logged" not in st.session_state:
            st.session_state.interaction_logged = False
            log_interaction()
            st.session_state.interaction_logged = True


def log_interaction():
    clean_query = st.session_state.user_query.strip()
    # Remove control characters to ensure logging is parsable
    clean_query = re.sub(r"(\n|\r|\t)", " ", clean_query)
    clean_query = "".join(char for char in clean_query if ord(char) >= 32)
    elapsed_time = datetime.now() - st.session_state.start_time
    elapsed_time = np.round(elapsed_time.total_seconds(), 0)
    custom_logger.info(
        f"{clean_query}\t{elapsed_time}s\t{st.session_state.fast_mode}\t{st.session_state.iterative_workflow}\t{len(st.session_state.search_queries)}\t{len(st.session_state.search_results)}\t{len(st.session_state.relevant_doc_ids)}"
    )


def process_query(
    user_query,
    iterative_workflow,
    fast_mode=False,
):
    """Process the user query and update the UI with progress"""
    model_config, workflow_config = get_model_and_workflow_config(fast_mode)

    # Initialize workflow
    workflow = ResearchWorkflow(
        st.session_state.docs, workflow_config, model_config, iterative_workflow
    )

    if st.session_state.iterative_workflow:
        st.info(
            "Iterative Recherche gestartet. Der Prozess wird falls nötig in mehreren Durchgängen durchgeführt."
        )

    # Initialize progress tracking.
    progress_bar = st.progress(0)
    status_text_01 = st.empty()

    # Progress tracking variables.
    total_steps_per_iteration = 5  # 5 main steps in workflow.
    iteration_weight = (
        80 / config["app"]["max_iterations"]
    )  # 80% for iterations, 20% for final report.
    step_weight = iteration_weight / total_steps_per_iteration
    current_step = 0

    # Function to update status and progress.
    def update_status(message, step_increment=1):
        nonlocal current_step
        status_text_01.text(message)
        if step_increment > 0:
            current_step += step_increment
            progress = min(int(current_step * step_weight), 80)
            progress_bar.progress(progress)

    # Run research iterations.
    for loop_idx in range(config["app"]["max_iterations"]):
        custom_logger.info_console(
            f"Iteration {loop_idx + 1} von {config['app']['max_iterations']}"
        )
        custom_logger.info_console("-" * 50)

        if loop_idx == 0:
            update_status(
                f"🔄 Recherche-Iteration {loop_idx + 1} gestartet...", step_increment=0
            )
        else:
            update_status(
                f"🔄 Weitere Recherche-Iteration {loop_idx + 1} gestartet...",
                step_increment=0,
            )

        # Run iteration with status callback.
        finished, final_docs = workflow.run_iteration(
            user_query, loop_idx, update_status
        )

        if finished is None or final_docs is None:
            update_status(
                "❌ Keine (weiteren) relevanten Dokumente gefunden.", step_increment=0
            )
            break

        if (
            not iterative_workflow
            or finished
            or loop_idx >= config["app"]["max_iterations"] - 1
        ):
            update_status("✅ Dokumentenanalyse abgeschlossen", step_increment=0)
            break
        else:
            update_status(
                "🔄 Der Auftrag konnte noch nicht vollständig geklärt werden. Eine weitere Iteration wird gestartet...",
                step_increment=0,
            )

    if final_docs is None:
        progress_bar.progress(100)
        st.error(
            "❌ Keine relevanten Dokumente gefunden. Bitte versuche eine andere Frage."
        )
        st.stop()

    # Create final report.
    update_status("📝 Erstelle Abschlussbericht...", step_increment=0)
    progress_bar.progress(85)

    stream = create_final_report(
        user_query, final_docs, model_id=model_config["final_report"]
    )

    placeholder = st.empty()
    final_report = ""
    # Stream output to show users that the process is progressing.
    while True:
        try:
            chunk = next(stream)
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].delta.content is not None
            ):  # type: ignore
                final_report += chunk.choices[0].delta.content.replace("ß", "ss")  # type: ignore
                placeholder.markdown(f"### Recherchebericht\n\n{final_report}")
        except StopIteration:
            break
        except Exception as e:
            st.error(f"Fehler beim Streamen des Textes: {str(e)}")
            break

    # Sometimes the LLMs fail in the last step, so we check if the final report is empty.
    if final_report.strip() == "":
        st.error(
            "❌ Der Abschlussbericht konnte wegen eines Fehlers vom Sprachmodell nicht erstellt werden. Bitte versuche es erneut."
        )
        st.stop()

    update_status("✅ Recherche erfolgreich abgeschlossen!", step_increment=0)
    progress_bar.progress(100)
    placeholder.empty()

    # Get detailed results from workflow to show in UI.
    results = workflow.get_results()

    # Set results to session state.
    st.session_state.user_query = user_query
    st.session_state.search_queries = results["search_queries"]
    st.session_state.search_results = results["search_results"]
    st.session_state.relevant_doc_ids = results["relevant_doc_ids"]
    st.session_state.final_docs = results["final_docs"]
    st.session_state.final_report = final_report


def display_results():
    """Display the research results"""

    st.header("💡 Recherche-Ergebnisse")

    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Suchanfragen", len(st.session_state.search_queries))
    with col2:
        st.metric("Analyiserte Textabschnitte", len(st.session_state.search_results))
    with col3:
        st.metric("Relevante Dokumente", len(st.session_state.relevant_doc_ids))

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Recherche-Bericht", "🔍 Suchanfragen", "📄 Dokumente", "⬇️ Download"]
    )

    with tab1:
        st.markdown("### Recherche-Bericht")
        st.markdown(st.session_state.final_report)
    with tab2:
        st.markdown("### Generierte Suchanfragen")
        for i, query in enumerate(st.session_state.search_queries, 1):
            st.markdown(f"**{i}.** {query}")
    with tab3:
        st.markdown("### Relevante Dokumente")
        if st.session_state.final_docs is None:
            st.warning("Keine relevanten Dokumente gefunden.")
        else:
            for idx, (_, row) in enumerate(st.session_state.final_docs.iterrows(), 1):
                with st.expander(f"📄 Dokument {idx}: {row['title'][:100]}..."):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown(f"**Titel:** {row['title']}")
                        st.markdown(f"**Datum:** {row['date']}")

                    with col2:
                        if "pdf_link" in row and pd.notna(row["pdf_link"]):
                            st.markdown(
                                f"**PDF:** [Link zum Dokument]({row['pdf_link']})"
                            )

                    if "analysis" in row and pd.notna(row["analysis"]):
                        st.markdown("**Analyse:**")
                        st.write(row["analysis"])
    with tab4:
        st.markdown("### Download")
        # Create DOCX version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Recherche_{timestamp}.docx"
        try:
            doc_bytes = create_docx_from_markdown(
                st.session_state.user_query, st.session_state.final_report
            )
            if doc_bytes:
                reports_dir = Path(__file__).parent / config["app"]["save_reports_to"]
                reports_dir.mkdir(exist_ok=True)
                doc_path = reports_dir / filename
                with open(doc_path, "wb") as f:
                    f.write(doc_bytes)

                st.download_button(
                    label="📄 Download als DOCX",
                    data=doc_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
        except Exception as e:
            st.error(f"Fehler beim Erstellen des DOCX-Downloads: {e}")


if __name__ == "__main__":
    main()
