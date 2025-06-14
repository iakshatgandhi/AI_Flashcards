import io
import csv

def create_anki_package(flashcards):
    """
    Create a text file that can be imported into Anki
    
    Args:
        flashcards (list): List of flashcard dictionaries
        
    Returns:
        str: Content that can be imported into Anki
    """
    return export_to_anki_txt(flashcards)

def export_to_anki_txt(flashcards):
    """
    Export flashcards to a txt file compatible with Anki import
    
    Args:
        flashcards (list): List of flashcard dictionaries
        
    Returns:
        str: Content that can be imported into Anki
    """
    output = io.StringIO()
    
    # First line explains the format
    output.write("# To import into Anki:\n")
    output.write("# 1. Open Anki\n")
    output.write("# 2. File > Import\n")
    output.write("# 3. Select this file\n")
    output.write("# 4. Ensure the fields are mapped correctly\n\n")
    
    # Tab-delimited format is preferred by Anki
    for card in flashcards:
        question = card['question'].replace("\n", " ").replace("\t", " ")
        answer = card['answer'].replace("\n", "<br>").replace("\t", " ")
        
        # Build tags
        tags = []
        if 'topic' in card and card['topic']:
            clean_topic = card['topic'].replace(" ", "_").replace("\t", "_")
            tags.append(f"topic:{clean_topic}")
        if 'difficulty' in card:
            tags.append(f"difficulty:{card['difficulty']}")
        
        # Write card as a tab-delimited line
        if tags:
            output.write(f"{question}\t{answer}\t{' '.join(tags)}\n")
        else:
            output.write(f"{question}\t{answer}\n")
    
    return output.getvalue()
