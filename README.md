# AI Flashcard Generator

An AI-powered tool that converts educational content into effective study flashcards using Large Language Models.

## Features

- **Content Ingestion**: Upload PDF, TXT files, or paste text directly
- **AI Processing**: Generate 10-30 flashcards from your educational content
- **Interactive UI**: View, filter, and study your flashcards
- **Export Options**: Save cards in CSV, JSON, or Anki-compatible formats

### Advanced Features

- **Difficulty Levels**: Cards are tagged as Easy, Medium, or Hard
- **Topic Grouping**: Cards are automatically organized by subject matter
- **Study Mode**: Simple spaced repetition system to review your cards
- **Multilingual Support**: Generate cards in different languages
- **Editable Cards**: Modify AI-generated content before export

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- OpenAI API key

### Setup

1. **Clone or download this repository**

2. **Create and activate a virtual environment**:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```
   pip install streamlit openai python-dotenv pandas nltk PyPDF2 scikit-learn
   ```
   
   Alternatively, create a requirements.txt file and install:
   ```
   pip install -r requirements.txt
   ```

4. **Configure OpenAI API key**:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_api_key_here`
   - Optional configuration:
     ```
     MODEL=gpt-3.5-turbo-16k
     MAX_TOKENS=4000
     TEMPERATURE=0.7
     ```

## Usage

1. **Start the application**:
   ```
   streamlit run app.py
   ```
   This will open the application in your default web browser.

2. **Check API Connection**:
   - In the sidebar, click the "Check API Connection" button
   - Ensure you see a green success message before proceeding

3. **Configure Generation Settings**:
   - Select the subject area (e.g., Computer Science, Mathematics)
   - Choose number of flashcards to generate (10-30)
   - Adjust advanced settings as needed:
     - Difficulty levels
     - Topic grouping
     - Language selection
     - Processing strategy

4. **Input Content**:
   - Upload a PDF or TXT file, or
   - Paste text directly in the text area

5. **Generate Flashcards**:
   - Click the "Generate Flashcards" button
   - Wait for processing (may take 30-60 seconds)

6. **Review and Study**:
   - Navigate cards with the arrow buttons
   - Switch between Card View and Table View
   - Filter by topic using the topic selection dropdown
   - Edit cards in Table View if needed

7. **Export Your Flashcards**:
   - Choose export format (CSV, JSON, Anki)
   - Download the file for use in other applications

8. **Study Mode**:
   - Click "Start Study Session" to begin studying
   - Rate your knowledge of each card (Easy, Medium, Hard)
   - Track progress through your flashcard deck

## Sample Outputs

### Computer Science Example

Input content about data structures might generate:

**Card 1**:
- **Question**: What is the difference between a stack and a queue?
- **Answer**: A stack follows Last-In-First-Out (LIFO) principle where elements are added and removed from the same end, while a queue follows First-In-First-Out (FIFO) principle where elements are added at one end and removed from the other end.
- **Difficulty**: Medium
- **Topic**: Data Structures

**Card 2**:
- **Question**: How does a binary search tree maintain its structure?
- **Answer**: A binary search tree maintains its structure by ensuring that for any given node, all nodes in its left subtree have smaller values and all nodes in its right subtree have greater values. This property enables efficient searching, insertion, and deletion operations.
- **Difficulty**: Hard
- **Topic**: Trees

### Biology Example

Input content about cell biology might generate:

**Card 1**:
- **Question**: What is the function of mitochondria in eukaryotic cells?
- **Answer**: Mitochondria are known as the "powerhouses" of the cell because they produce ATP (adenosine triphosphate) through cellular respiration. This ATP serves as the primary energy currency for cellular processes, enabling the cell to perform necessary functions.
- **Difficulty**: Easy
- **Topic**: Cell Biology

**Card 2**:
- **Question**: How does the phospholipid bilayer contribute to cell membrane function?
- **Answer**: The phospholipid bilayer forms the foundation of cell membranes with hydrophilic phosphate heads facing outward and hydrophobic fatty acid tails facing inward. This structure creates a selective barrier that regulates what enters and exits the cell, allowing passage of small, nonpolar molecules while blocking larger or charged molecules.
- **Difficulty**: Medium
- **Topic**: Cell Membranes

## Troubleshooting

- **API Connection Issues**: Verify your OpenAI API key is correctly set in the .env file.
- **No Cards Generated**: Check that your input content has sufficient length (minimum 100 characters).
- **Missing Topics**: For very short content, topic detection may default to general categories.
- **Export Issues**: Ensure you have the required permissions to write files to your download location.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
