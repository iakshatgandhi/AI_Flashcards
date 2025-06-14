import streamlit as st
import json
import csv
import io
from llm_handler import generate_flashcards, check_api_connection
from content_processor import extract_text_from_file
import pandas as pd
from anki_utils import create_anki_package, export_to_anki_txt

st.set_page_config(page_title="AI Flashcard Generator", layout="wide")

def save_flashcards(flashcards, file_format):
    if file_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Question", "Answer", "Difficulty", "Topic"])
        for card in flashcards:
            writer.writerow([card["question"], card["answer"], card.get("difficulty", "Medium"), card.get("topic", "")])
        return output.getvalue()
    elif file_format == "anki":
        return export_to_anki_txt(flashcards)
    else:  # json format
        return json.dumps(flashcards, indent=2)

def main():
    st.title("AI Flashcard Generator")
    st.markdown("Convert your educational content into flashcards using AI!")
    
    # Initialize session state variables if needed
    if 'current_card' not in st.session_state:
        st.session_state.current_card = 0
    if 'study_mode' not in st.session_state:
        st.session_state.study_mode = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        # Add API check button
        if st.button("Check API Connection"):
            with st.spinner("Testing API connection..."):
                success, message = check_api_connection()
                if success:
                    st.success("✅ API connection successful!")
                else:
                    st.error(f"❌ API connection failed: {message}")
                    st.info("Please check your API key in the .env file.")
        
        subject = st.selectbox(
            "Select Subject",
            ["General", "Computer Science", "Mathematics", "Biology", "History", "Physics", "Chemistry", "Languages"]
        )
        
        # Ensure minimum 10 flashcards as per requirements
        num_cards = st.slider("Number of flashcards to generate", 10, 30, 15)
        
        advanced_features = st.expander("Advanced Features")
        with advanced_features:
            enable_difficulty = st.checkbox("Add difficulty levels", True)
            enable_topics = st.checkbox("Group by topics", True)
            enable_multilingual = st.checkbox("Enable multilingual support", False)
            if enable_multilingual:
                language = st.selectbox("Select language", ["English", "Spanish", "French", "German", "Chinese", "Japanese"])
            else:
                language = "English"
            
            # Add advanced processing options
            enable_advanced = st.checkbox("Use advanced NLP processing", True)
            if enable_advanced:
                chunking_strategy = st.radio(
                    "Content processing strategy:",
                    ["Auto-detect", "Topic-based", "Length-based"],
                    index=0
                )
            else:
                chunking_strategy = "Length-based"
    
    # Main content area for input
    st.header("Input Content")
    input_method = st.radio("Select input method:", ["Upload File", "Paste Text"])
    
    content_text = ""
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])
        if uploaded_file is not None:
            try:
                content_text = extract_text_from_file(uploaded_file)
                st.success("File processed successfully!")
                with st.expander("Preview Content"):
                    st.text(content_text[:500] + ("..." if len(content_text) > 500 else ""))
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        content_text = st.text_area("Paste your content here:", height=200)
    
    # Generate flashcards
    if st.button("Generate Flashcards") and content_text:
        with st.spinner("Generating flashcards... This may take a minute."):
            try:
                config = {
                    "subject": subject,
                    "num_cards": num_cards,
                    "add_difficulty": enable_difficulty,
                    "add_topics": enable_topics,
                    "language": language if enable_multilingual else "English",
                    "min_cards": 10,  # Ensure at least 10 cards
                    "chunking_strategy": chunking_strategy,
                    "advanced_processing": enable_advanced
                }
                
                # First, display a progress message
                progress_placeholder = st.empty()
                progress_placeholder.info("Step 1/3: Processing content...")
                
                # Check API before proceeding
                success, message = check_api_connection()
                if not success:
                    progress_placeholder.error(f"API connection failed: {message}")
                    st.error("Cannot generate flashcards due to API connection issues.")
                    return
                
                # Process content in stages for better user feedback
                progress_placeholder.info("Step 2/3: Generating flashcards with AI...")
                flashcards = generate_flashcards(content_text, config)
                
                progress_placeholder.info("Step 3/3: Post-processing flashcards...")
                
                # Final step
                progress_placeholder.empty()
                st.session_state.flashcards = flashcards
                st.session_state.current_card = 0
                st.success(f"Successfully generated {len(flashcards)} flashcards!")
                st.session_state.study_mode = False
                
            except Exception as e:
                st.error(f"Error generating flashcards: {str(e)}")
    
    # Display and manage flashcards
    if 'flashcards' in st.session_state and st.session_state.flashcards:
        st.header("Generated Flashcards")
        
        # Show metadata for debugging/transparency
        if any('_generated_by' in card for card in st.session_state.flashcards):
            with st.expander("Generation Info"):
                fallback_count = sum(1 for card in st.session_state.flashcards if card.get('_generated_by') == 'fallback')
                if fallback_count:
                    st.info(f"{fallback_count} cards were generated using fallback methods due to processing limitations.")
        
        # Convert to DataFrame for better display
        # Filter out metadata fields before displaying
        display_cards = []
        for card in st.session_state.flashcards:
            display_card = {k: v for k, v in card.items() if not k.startswith('_')}
            display_cards.append(display_card)
            
        flashcards_df = pd.DataFrame(display_cards)
        
        # Display tabs for different views
        tab1, tab2 = st.tabs(["Card View", "Table View"])
        
        with tab1:
            # Card view - one card at a time
            if len(st.session_state.flashcards) > 0:
                current_card = st.session_state.flashcards[st.session_state.current_card]
                
                col1, col2, col3 = st.columns([1, 10, 1])
                
                with col1:
                    if st.button("←") and st.session_state.current_card > 0:
                        st.session_state.current_card -= 1
                        st.rerun()  # Using st.rerun() instead of experimental_rerun
                
                with col3:
                    if st.button("→") and st.session_state.current_card < len(st.session_state.flashcards) - 1:
                        st.session_state.current_card += 1
                        st.rerun()  # Using st.rerun() instead of experimental_rerun
                
                with col2:
                    card_container = st.container()
                    with card_container:
                        st.markdown("### Question:")
                        st.markdown(f"**{current_card['question']}**")
                        
                        show_answer = st.checkbox("Show Answer")
                        if show_answer:
                            st.markdown("### Answer:")
                            st.markdown(current_card['answer'])
                        
                        if 'difficulty' in current_card:
                            diff_color = {
                                "Easy": "green",
                                "Medium": "orange",
                                "Hard": "red"
                            }.get(current_card['difficulty'], "gray")
                            st.markdown(f"<span style='color:{diff_color};'>Difficulty: {current_card['difficulty']}</span>", unsafe_allow_html=True)
                        
                        if 'topic' in current_card and current_card['topic']:
                            st.markdown(f"Topic: {current_card['topic']}")
                
                st.progress((st.session_state.current_card + 1) / len(st.session_state.flashcards))
                st.markdown(f"Card {st.session_state.current_card + 1} of {len(st.session_state.flashcards)}")
        
        with tab2:
            # Table view - see all cards
            editable_df = st.data_editor(flashcards_df, use_container_width=True, num_rows="dynamic")
            if not flashcards_df.equals(editable_df):
                st.session_state.flashcards = editable_df.to_dict('records')
                st.success("Flashcards updated!")
        
        # Add topic filtering with enhanced visualization
        if any('topic' in card and card['topic'] for card in st.session_state.flashcards):
            st.header("Topic Distribution")
            topics = list(set(card.get('topic', '') for card in st.session_state.flashcards if card.get('topic', '')))
            
            if topics:
                # Create a bar chart of topic distribution
                topic_counts = {}
                for topic in topics:
                    topic_counts[topic] = sum(1 for card in st.session_state.flashcards if card.get('topic', '') == topic)
                
                # Convert to DataFrame for charting
                topic_df = pd.DataFrame({
                    'Topic': list(topic_counts.keys()),
                    'Count': list(topic_counts.values())
                })
                
                # Sort by count for better visualization
                topic_df = topic_df.sort_values('Count', ascending=False)
                
                # Display the chart
                st.bar_chart(topic_df.set_index('Topic'))
                
                # Topic filter
                selected_topic = st.selectbox("Select topic to filter:", ["All Topics"] + topics)
                if selected_topic != "All Topics":
                    filtered_cards = [card for card in st.session_state.flashcards if card.get('topic', '') == selected_topic]
                    st.write(f"Showing {len(filtered_cards)} cards for topic: {selected_topic}")
                    # Create a temporary view of filtered cards
                    filtered_df = pd.DataFrame([{k: v for k, v in card.items() if not k.startswith('_')} for card in filtered_cards])
                    st.dataframe(filtered_df)
                    
                    # Add option to study specific topic
                    if st.button(f"Study '{selected_topic}' Cards"):
                        st.session_state.filtered_study = filtered_cards
                        st.session_state.study_mode = True
                        st.session_state.studied_cards = []
                        st.rerun()
        
        # Export options
        st.header("Export Flashcards")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_format = st.selectbox("Choose export format:", ["csv", "json", "anki"])
        
        with col2:
            if export_format == "anki":
                try:
                    export_data = create_anki_package(st.session_state.flashcards)
                    st.download_button(
                        label="Download for Anki",
                        data=export_data,
                        file_name="flashcards.txt",
                        mime="text/plain"
                    )
                except ImportError:
                    st.warning("Anki export requires genanki package. Using text format instead.")
                    export_data = save_flashcards(st.session_state.flashcards, "anki")
                    st.download_button(
                        label="Download for Anki (Text)",
                        data=export_data,
                        file_name="flashcards.txt",
                        mime="text/plain"
                    )
            else:
                export_data = save_flashcards(st.session_state.flashcards, export_format)
                st.download_button(
                    label=f"Download as {export_format.upper()}",
                    data=export_data,
                    file_name=f"flashcards.{export_format}",
                    mime="text/csv" if export_format == "csv" else "application/json"
                )
        
        with col3:
            # Add a study mode toggle
            if st.button("Start Study Session"):
                st.session_state.study_mode = True
                st.session_state.studied_cards = []
                # Use filtered cards if they exist
                if 'filtered_study' not in st.session_state:
                    st.session_state.filtered_study = st.session_state.flashcards
                st.rerun()  # Using st.rerun() instead of experimental_rerun
    
    # Add a study mode for flashcard review
    if 'study_mode' in st.session_state and st.session_state.study_mode and 'flashcards' in st.session_state:
        st.header("Study Session")
        
        # Exit study mode button
        if st.button("Exit Study Mode"):
            st.session_state.study_mode = False
            if 'filtered_study' in st.session_state:
                del st.session_state.filtered_study
            st.rerun()  # Using st.rerun() instead of experimental_rerun
        
        # Simple spaced repetition logic
        if 'studied_cards' not in st.session_state:
            st.session_state.studied_cards = []
        
        study_cards = st.session_state.filtered_study if 'filtered_study' in st.session_state else st.session_state.flashcards
        
        remaining_cards = [i for i in range(len(study_cards)) 
                          if i not in st.session_state.studied_cards]
        
        if not remaining_cards:
            st.success("🎉 You've reviewed all the flashcards! 🎉")
            if st.button("Start Over"):
                st.session_state.studied_cards = []
                st.rerun()  # Using st.rerun() instead of experimental_rerun
        else:
            # Select next card to study
            current_study_idx = remaining_cards[0]
            card = study_cards[current_study_idx]
            
            st.markdown(f"### Question ({len(st.session_state.studied_cards) + 1}/{len(study_cards)}):")
            st.markdown(f"**{card['question']}**")
            
            reveal = st.checkbox("Reveal Answer")
            if reveal:
                st.markdown("### Answer:")
                st.markdown(card['answer'])
                
                st.write("How well did you know this?")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Easy"):
                        st.session_state.studied_cards.append(current_study_idx)
                        st.rerun()  # Using st.rerun() instead of experimental_rerun
                
                with col2:
                    if st.button("Medium"):
                        st.session_state.studied_cards.append(current_study_idx)
                        st.rerun()  # Using st.rerun() instead of experimental_rerun
                
                with col3:
                    if st.button("Hard"):
                        st.session_state.studied_cards.append(current_study_idx)
                        st.rerun()  # Using st.rerun() instead of experimental_rerun
            
            # Display progress
            st.progress(len(st.session_state.studied_cards) / len(study_cards))

if __name__ == "__main__":
    main()
