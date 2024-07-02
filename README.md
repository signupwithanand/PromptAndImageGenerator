# Poetry Image Generator

## Description
The Poetry Image Generator is an innovative application that combines the art of poetry with advanced AI image generation. This tool analyzes poems, understands their context and themes, and creates unique, relevant images for each stanza.

## Features
- Poem analysis: Breaks down poems into stanzas and analyzes their context.
- Character identification: Recognizes and describes key characters in the poem.
- AI-powered image generation: Creates images based on the content and context of each stanza.
- User feedback system: Allows users to approve, regenerate, or provide feedback on generated images.
- Interactive UI: Provides a user-friendly interface for inputting poems and viewing generated images.

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/signupwithanand/PromptAndImageGenerator.git
   cd PromptAndImageGenerator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage
1. Run the application:
   ```
   python app.py
   ```

2. Input your poem in the text area provided.

3. Click "Generate Images" to start the process.

4. For each stanza, review the generated image and provide feedback or choose to regenerate.

5. Continue until all stanzas have been processed.

## Contributing
Contributions to improve the Poetry Image Generator are welcome. Please feel free to submit a Pull Request.

## License
Anand's app repository

## Acknowledgments
- OpenAI for the DALL-E API
- Anthropic for the Claude API
