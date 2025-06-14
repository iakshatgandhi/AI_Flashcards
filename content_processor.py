import PyPDF2
import io
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def extract_text_from_file(uploaded_file):
    """
    Extract text content from uploaded files based on their format.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Extracted text content
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        return extract_from_txt(uploaded_file)
    elif file_extension == 'pdf':
        return extract_from_pdf(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: .{file_extension}")

def extract_from_txt(file):
    """Extract text from a .txt file"""
    try:
        return file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        try:
            return file.getvalue().decode("latin-1")
        except:
            raise Exception("Unable to decode text file. Please ensure it's properly encoded.")

def extract_from_pdf(file):
    """Extract text from a .pdf file with improved structure preservation"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        
        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            raise Exception("The PDF file is encrypted/password protected.")
            
        text = ""
        
        # Extract text with page markers to help preserve structure
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Add page marker for better structure preservation
                if i > 0:  # Don't add page break before the first page
                    text += f"\n\n--- Page {i+1} ---\n\n"
                text += page_text + "\n"
        
        # Apply enhanced preprocessing to clean up the extracted text
        return enhanced_preprocess_text(text)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def enhanced_preprocess_text(text):
    """
    Enhanced preprocessing to improve text quality for flashcard generation.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned and structured text
    """
    if not text:
        return ""
        
    # Normalize line breaks and spacing
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])-\s*\n([a-z])', r'\1\2', text)  # Fix hyphenation
    
    # Clean up bullet points and numbering
    text = re.sub(r'•\s*', '• ', text)  # Normalize bullet points
    text = re.sub(r'(\d+)\.\s*', r'\1. ', text)  # Normalize numbered lists
    
    # Remove headers/footers that might appear on every page
    lines = text.split('\n')
    if len(lines) > 10:
        repeating_lines = find_repeating_lines(lines)
        cleaned_lines = [line for line in lines if line.strip() not in repeating_lines]
        text = '\n'.join(cleaned_lines)
    
    # Preserve paragraph structure but fix broken lines
    paragraphs = []
    current_paragraph = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        elif re.match(r'^[A-Z]', line) and current_paragraph and current_paragraph[-1].endswith('.'):
            # Likely a new sentence starting a new paragraph
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [line]
        elif re.match(r'^[0-9]+\.\s', line) or re.match(r'^•\s', line) or re.match(r'^[A-Z][A-Za-z\s]+:', line):
            # Likely a list item or heading
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [line]
        else:
            current_paragraph.append(line)
    
    # Add the last paragraph if it exists
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Join paragraphs with double newline
    return '\n\n'.join(paragraphs)

def find_repeating_lines(lines):
    """
    Identify lines that repeat frequently and might be headers/footers
    """
    line_counts = {}
    for line in lines:
        line = line.strip()
        if len(line) > 5:  # Only consider non-trivial lines
            line_counts[line] = line_counts.get(line, 0) + 1
    
    # Lines that repeat on multiple pages might be headers/footers
    repeating_lines = set()
    for line, count in line_counts.items():
        if count >= 3:  # Appears at least 3 times
            repeating_lines.add(line)
    
    return repeating_lines

def extract_topics(text):
    """
    Extract potential topics from the text content using headings and keyword analysis.
    
    Args:
        text (str): Content text
        
    Returns:
        list: Potential topics identified in the text
    """
    topics = []
    
    # Look for section headers (patterns like "Chapter X:" or "X. Title")
    header_patterns = [
        r'^[A-Z][A-Za-z\s]+:',  # "Introduction:", "Summary:", etc.
        r'^\d+\.\s+[A-Z][A-Za-z\s]+',  # "1. Title", "2. Another Section"
        r'^Chapter\s+\d+[\.:]\s*[A-Z][A-Za-z\s]+',  # "Chapter 1: Title"
        r'^SECTION\s+\d+[\.:]\s*[A-Z][A-Za-z\s]+',  # "SECTION 1: Title"
    ]
    
    # Extract potential section headers
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) < 100:  # Avoid excessively long lines
            for pattern in header_patterns:
                if re.match(pattern, line):
                    # Clean up the topic name
                    topic = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                    topic = re.sub(r'^Chapter\s+\d+[\.:]\s*', '', topic)  # Remove "Chapter X:"
                    topic = re.sub(r'^SECTION\s+\d+[\.:]\s*', '', topic)  # Remove "SECTION X:"
                    topic = re.sub(r':$', '', topic)  # Remove trailing colon
                    topics.append(topic.strip())
                    break
    
    # If no topics found, try to extract key phrases
    if not topics:
        # This is a placeholder - in a real app, you'd use NLP techniques
        # Consider using TextRank or similar algorithms to extract key phrases
        pass
    
    # Ensure we don't have too many topics
    max_topics = 10
    if len(topics) > max_topics:
        # Keep only the first N topics
        topics = topics[:max_topics]
    
    return topics

def extract_topics_from_text(text, num_topics=5):
    """
    Extract potential topics from the text content using NLP techniques.
    
    Args:
        text (str): Content text
        num_topics (int): Maximum number of topics to extract
        
    Returns:
        list: List of detected topics
        dict: Dictionary mapping sections of text to topics
    """
    # Ensure we have enough content to work with
    if len(text.strip()) < 200:
        return ["General Content"], {"General Content": text}
    
    # First try to extract topics from headings
    heading_topics = extract_topics_from_headings(text)
    if heading_topics:
        # Use heading-based topic segmentation
        topic_sections = segment_by_headings(text, heading_topics)
        return heading_topics[:num_topics], topic_sections
    
    # If no headings found, try NMF topic modeling approach
    try:
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        
        # Skip this approach if we have too few sentences
        if len(sentences) < 5:
            return ["General Content"], {"General Content": text}
        
        # Create chunks of text by grouping sentences
        chunk_size = max(3, len(sentences) // 20)  # Adjust chunk size based on text length
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i+chunk_size])
            if len(chunk.strip()) > 50:  # Only include non-trivial chunks
                chunks.append(chunk)
        
        # Skip if we don't have enough chunks
        if len(chunks) < 3:
            return ["General Content"], {"General Content": text}
        
        # Extract features using TF-IDF
        stop_words = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(
            max_df=0.95, 
            min_df=2,
            max_features=1000,
            stop_words=stop_words
        )
        
        # Try to fit the vectorizer
        try:
            tfidf_matrix = vectorizer.fit_transform(chunks)
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply NMF for topic modeling
            nmf_model = NMF(n_components=min(num_topics, len(chunks)//2), random_state=1)
            nmf_features = nmf_model.fit_transform(tfidf_matrix)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
                top_words = [feature_names[i] for i in top_words_idx]
                topic_name = f"{top_words[0].capitalize()} & {top_words[1].capitalize()}"
                topics.append(topic_name)
            
            # Map chunks to topics
            chunk_topics = nmf_features.argmax(axis=1)
            topic_sections = {}
            for topic_idx, topic_name in enumerate(topics):
                topic_chunks = [chunks[i] for i in range(len(chunks)) if chunk_topics[i] == topic_idx]
                if topic_chunks:
                    topic_sections[topic_name] = "\n\n".join(topic_chunks)
            
            return topics, topic_sections
            
        except Exception as e:
            print(f"Error in NMF topic extraction: {str(e)}")
            return ["General Content"], {"General Content": text}
            
    except Exception as e:
        print(f"Error extracting topics with NLP: {str(e)}")
        return ["General Content"], {"General Content": text}

def extract_topics_from_headings(text):
    """Extract topics based on section headers in the text"""
    topics = []
    
    # Common heading patterns
    header_patterns = [
        r'^[A-Z][A-Za-z\s]+:',  # "Introduction:", "Summary:", etc.
        r'^\d+\.\s+[A-Z][A-Za-z\s]+',  # "1. Title", "2. Another Section"
        r'^Chapter\s+\d+[\.:]\s*[A-Z][A-Za-z\s]+',  # "Chapter 1: Title"
        r'^SECTION\s+\d+[\.:]\s*[A-Z][A-Za-z\s]+',  # "SECTION 1: Title"
    ]
    
    # Extract potential section headers
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) < 100:  # Avoid excessively long lines
            for pattern in header_patterns:
                if re.match(pattern, line):
                    # Clean up the topic name
                    topic = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                    topic = re.sub(r'^Chapter\s+\d+[\.:]\s*', '', topic)  # Remove "Chapter X:"
                    topic = re.sub(r'^SECTION\s+\d+[\.:]\s*', '', topic)  # Remove "SECTION X:"
                    topic = re.sub(r':$', '', topic)  # Remove trailing colon
                    topics.append(topic.strip())
                    break
    
    return topics

def segment_by_headings(text, headings):
    """Segment text by identified headings"""
    if not headings:
        return {"General Content": text}
    
    topic_sections = {}
    lines = text.split('\n')
    current_topic = None
    current_content = []
    
    # Convert headings to regex patterns for matching
    heading_patterns = []
    for heading in headings:
        # Escape special characters in the heading
        escaped_heading = re.escape(heading)
        # Create patterns that might match this heading in the text
        patterns = [
            f"^{escaped_heading}:?$",
            f"^\\d+\\.\\s+{escaped_heading}$",
            f"^Chapter\\s+\\d+[\\.:]*\\s*{escaped_heading}$",
            f"^SECTION\\s+\\d+[\\.:]*\\s*{escaped_heading}$"
        ]
        heading_patterns.append((heading, patterns))
    
    # Process the text line by line
    for line in lines:
        line_stripped = line.strip()
        matched = False
        
        # Check if this line matches any heading pattern
        for heading, patterns in heading_patterns:
            for pattern in patterns:
                if re.match(pattern, line_stripped) or heading in line_stripped:
                    # Found a new section
                    if current_topic:
                        # Save the previous section
                        topic_sections[current_topic] = '\n'.join(current_content)
                    
                    # Start a new section
                    current_topic = heading
                    current_content = []
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            # If no match and no current topic, start with general content
            if current_topic is None:
                current_topic = "General Content"
            
            # Add this line to the current section
            current_content.append(line)
    
    # Add the final section
    if current_topic:
        topic_sections[current_topic] = '\n'.join(current_content)
    
    return topic_sections

def chunk_content_for_processing(text, topics_dict, max_chunk_size=6000):
    """
    Divide content into chunks for more effective LLM processing,
    respecting topic boundaries where possible.
    
    Args:
        text (str): Full content text
        topics_dict (dict): Dictionary mapping topics to text sections
        max_chunk_size (int): Maximum size of each chunk
        
    Returns:
        list: List of tuples containing (chunk_text, topics)
    """
    if not topics_dict or len(topics_dict) <= 1:
        # If no topic segmentation, chunk by size
        return chunk_by_size(text, max_chunk_size)
    
    # Create chunks based on topics
    chunks = []
    for topic, content in topics_dict.items():
        if len(content) <= max_chunk_size:
            # Topic content fits in one chunk
            chunks.append((content, [topic]))
        else:
            # Topic content needs to be split
            topic_chunks = chunk_by_size(content, max_chunk_size)
            for chunk_text, _ in topic_chunks:
                chunks.append((chunk_text, [topic]))
    
    return chunks

def chunk_by_size(text, max_chunk_size):
    """Split text into chunks of maximum size, trying to preserve sentence boundaries"""
    chunks = []
    
    # First split by paragraphs
    paragraphs = text.split('\n\n')
    current_chunk = []
    current_size = 0
    current_topics = ["General Content"]
    
    for para in paragraphs:
        para_size = len(para)
        
        if current_size + para_size <= max_chunk_size:
            # Add to current chunk
            current_chunk.append(para)
            current_size += para_size
        else:
            # Check if paragraph itself is too large
            if para_size > max_chunk_size:
                # Split paragraph by sentences
                sentences = sent_tokenize(para)
                sentence_chunk = []
                sentence_size = 0
                
                for sentence in sentences:
                    sent_size = len(sentence)
                    if sentence_size + sent_size <= max_chunk_size:
                        # Add to current sentence chunk
                        sentence_chunk.append(sentence)
                        sentence_size += sent_size
                    else:
                        # This sentence chunk is full
                        if sentence_chunk:
                            # Save current sentence chunk
                            if current_chunk:
                                # First save the accumulated paragraphs
                                chunks.append(('\n\n'.join(current_chunk), current_topics))
                                current_chunk = []
                                current_size = 0
                            
                            chunks.append((' '.join(sentence_chunk), current_topics))
                            sentence_chunk = [sentence]
                            sentence_size = sent_size
                        else:
                            # Individual sentence is too long, must split by words
                            chunks.append((sentence[:max_chunk_size], current_topics))
                            # Process remainder in next iteration
                            remaining = sentence[max_chunk_size:]
                            if remaining:
                                sentences.insert(0, remaining)  # Reinsert remainder
                
                # Add any remaining sentences
                if sentence_chunk:
                    if current_chunk:
                        chunks.append(('\n\n'.join(current_chunk), current_topics))
                        current_chunk = []
                        current_size = 0
                    chunks.append((' '.join(sentence_chunk), current_topics))
            else:
                # Finish current chunk and start a new one
                if current_chunk:
                    chunks.append(('\n\n'.join(current_chunk), current_topics))
                current_chunk = [para]
                current_size = para_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(('\n\n'.join(current_chunk), current_topics))
    
    return chunks
