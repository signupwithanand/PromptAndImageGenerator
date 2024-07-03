import os
import io
from io import BytesIO
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
import requests
from openai import OpenAI
import anthropic
import json
import time
from dotenv import load_dotenv
import PySimpleGUI as sg
import threading

# Load environment variables
load_dotenv()

# Initialize the OpenAI and Anthropic clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Character:
    def __init__(self, name: str, brief_description: str):
        self.name = name
        self.brief_description = brief_description
        self.detailed_description = ""
        self.reference_image = None

class Poem:
    def __init__(self, text: str):
        self.text = text
        self.stanzas = []
        self.characters: Dict[str, Character] = {}
        self.theme = ""
        self.context = ""
        self.time_period = ""
        self.location = ""
        self.style = ""

class ContextAnalyzer:
    def analyze_context(self, poem_text: str, window) -> Dict[str, str]:
        update_output(window, "Analyzing poem context...")
        context_prompt = f"""Analyze the following poem and provide:
        1. The main theme or subject of the poem
        2. The cultural, historical, or social context (be as specific as possible)
        3. The setting or time period (if applicable, otherwise state 'Not specified')
        4. Any significant symbols or motifs
        5. The geographical location or region associated with the poem's content (if applicable)
        6. The style or type of poem (e.g., narrative, lyric, free verse, etc.)

        Poem:
        {poem_text}

        Provide your analysis as a Python dictionary with keys 'theme', 'context', 'time_period', 'symbols', 'location', and 'style'. Be as detailed and specific as possible, but use 'Not specified' if any aspect is not clear from the poem."""
        
        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": context_prompt}]
        )
        
        try:
            context_dict = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            content = response.content[0].text
            context_dict = {}
            for line in content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    context_dict[key.strip().strip("'")] = value.strip().strip("'")

        required_keys = ['theme', 'context', 'time_period', 'symbols', 'location', 'style']
        for key in required_keys:
            if key not in context_dict:
                context_dict[key] = "Not specified"

        update_output(window, f"Context analysis complete: {json.dumps(context_dict, indent=2)}")
        return context_dict


class CharacterInitializer:
    def generate_detailed_description(self, name: str, brief_description: str, poem_context: str, window) -> str:
        update_output(window, f"Generating detailed description for character: {name}")
        prompt = f"""Generate a highly detailed physical description for the character or element "{name}" based on the brief description: "{brief_description}". Consider the poem's context: {poem_context}

        Include specific details about:
        1. Facial features (if applicable: shape of face, eyes, nose, mouth, ears, etc.)
        2. Skin tone and texture (if applicable: including any wrinkles, blemishes, or distinguishing marks)
        3. Hair color, style, and texture (if applicable)
        4. Body type and posture (if applicable)
        5. Clothing and accessories (be specific about styles, colors, and materials)
        6. Any unique or distinguishing characteristics
        7. Approximate age and how it shows in their appearance (if applicable)
        8. Typical facial expressions or emotions (if applicable)
        9. If this is not a person, describe its physical appearance, colors, textures, and any other relevant visual details

        Provide a comprehensive description that would allow an artist to create a consistent representation of this character or element across multiple illustrations."""

        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()

    def initialize_character(self, name: str, brief_description: str, poem_context: str, window) -> Character:
        character = Character(name, brief_description)
        character.detailed_description = self.generate_detailed_description(name, brief_description, poem_context, window)
        
        try:
            update_output(window, f"Generating image for character: {name}")
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=f"Create a detailed portrait of {character.detailed_description}. Ensure the image captures all the specific physical characteristics described.",
                n=1,
                size="1024x1024",
                quality="hd",
                style="vivid"
            )

            image_url = response.data[0].url
            image_response = requests.get(image_url)
            character.reference_image = Image.open(BytesIO(image_response.content))
            
        except Exception as e:
            update_output(window, f"Error generating image for character {name}: {str(e)}")
            character.reference_image = None
        
        return character

class PoemAnalyzer:
    def analyze_poem(self, poem_text: str, window) -> Poem:
        poem = Poem(poem_text)
        poem.stanzas = self.identify_stanzas(poem_text)
        context_analysis = self.analyze_poem_context(poem_text, window)
        poem.context = context_analysis.get('context', 'Unknown context')
        poem.time_period = context_analysis.get('time_period', 'Unknown time period')
        poem.cultural_setting = context_analysis.get('cultural_setting', 'Unknown cultural setting')
        return poem

    def identify_characters(self, poem_text: str, window) -> List[Dict[str, str]]:
        update_output(window, "Identifying characters in the poem...")
        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"""Identify key characters or subjects in this poem. For each, provide a name or identifier and a brief description. Format your response as a Python list of dictionaries, where each dictionary has 'name' and 'brief_description' keys. Only include characters or entities that are explicitly mentioned or clearly implied in the poem. If the poem doesn't have specific characters, identify key subjects or elements. Example format:
                [
                    {{"name": "The Soldier", "brief_description": "A young man in military uniform"}},
                    {{"name": "The City", "brief_description": "A bustling metropolis"}}
                ]

                Poem:
                {poem_text}"""}
            ]
        )
        
        content = response.content[0].text
        
        try:
            characters = json.loads(content)
            if isinstance(characters, list):
                return characters
        except json.JSONDecodeError:
            update_output(window, "Failed to parse the character list as JSON. Attempting manual extraction.")
            
        characters = []
        lines = content.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    name = parts[0].strip().strip('"{}')
                    brief_description = parts[1].strip().strip('"{}')
                    characters.append({"name": name, "brief_description": brief_description})
        
        if not characters:
            update_output(window, "No specific characters or key elements identified in the poem.")
            characters = [{"name": "General Scene", "brief_description": "A representation of the overall poem's atmosphere"}]
        
        return characters


    def identify_stanzas(self, poem_text: str) -> List[str]:
        return [stanza.strip() for stanza in poem_text.split('\n\n') if stanza.strip()]

    def analyze_poem_context(self, poem_text: str, window) -> Dict[str, str]:
        try:
            update_output(window, "Analyzing poem context...")
            response = claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"Analyze the following poem and provide:\n1. A brief summary of its overall context, including cultural and thematic elements.\n2. The specific time period the poem is set in or refers to.\n3. The cultural setting or background of the poem.\n\nFormat your response as a Python dictionary with keys 'context', 'time_period', and 'cultural_setting'.\n\n{poem_text}"}
                ]
            )
            
            try:
                context_dict = json.loads(response.content[0].text)
            except json.JSONDecodeError:
                content = response.content[0].text
                context_dict = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        context_dict[key.strip().strip("'")] = value.strip().strip("'")

            required_keys = ['context', 'time_period', 'cultural_setting']
            for key in required_keys:
                if key not in context_dict:
                    context_dict[key] = f"Unknown {key}"

            return context_dict
        except Exception as e:
            update_output(window, f"An error occurred during poem context analysis: {str(e)}")
            return {
                'context': 'Unknown context',
                'time_period': 'Unknown time period',
                'cultural_setting': 'Unknown cultural setting'
            }

class PromptGenerator:
    def __init__(self):
        self.feedback_history = []
    
    def generate_prompt(self, stanza: str, poem: Poem, characters: Dict[str, Character], window) -> str:
        update_output(window, "Analyzing stanza and generating prompt...")
        start_time = time.time()

        stanza_characters = self.identify_stanza_characters(stanza, characters)
        character_descriptions = "\n".join([f"{name}: {char.detailed_description}" for name, char in stanza_characters.items()])
        
        context_prompt = f"""Generate a vivid, accurate description for an illustration based on the following stanza and context:

        Stanza: "{stanza}"

        Overall poem theme: {poem.theme}
        Context: {poem.context}
        Time period: {poem.time_period}
        Location: {poem.location}
        Style: {poem.style}

        Characters in this stanza (use these exact descriptions for consistency):
        {character_descriptions}

        Previous feedback: {' '.join(self.feedback_history)}

        Instructions:
        1. Create a scene that EXACTLY matches the content of the given stanza, considering the poem's context and style.
        2. If specific characters are mentioned, use their detailed descriptions to ensure consistency across illustrations.
        3. If a specific time period is mentioned, ensure all elements (clothing, architecture, objects) are historically accurate.
        4. If specific locations are mentioned, accurately represent their geographical and cultural characteristics.
        5. Pay close attention to the mood, atmosphere, and any symbolic elements mentioned in the poem.
        6. Only include characters, objects, or elements explicitly mentioned or clearly implied in the stanza.
        7. If the poem is abstract or doesn't describe a specific scene, focus on capturing the emotion or concept through color, composition, and symbolic imagery.

        Provide a detailed description (200-250 words) for an illustration that vividly and accurately captures the stanza's specific content, atmosphere, and underlying meaning, suitable for an AI image generation model. Ensure that any characters described are portrayed exactly as in their detailed descriptions."""
        
        update_output(window, "Sending prompt to Claude AI for processing...")
        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": context_prompt}
            ]
        )
        
        prompt = response.content[0].text.strip()
        
        negative_prompt = self.generate_negative_prompt(poem)
        
        end_time = time.time()
        update_output(window, f"Prompt generation completed in {end_time - start_time:.2f} seconds.")
        
        return f"{prompt}\nNegative prompt: {negative_prompt}"

    def generate_negative_prompt(self, poem: Poem) -> str:
        negative_elements = [
            "low quality", "blurry", "distorted", "anachronistic elements",
            "historically inaccurate details", "culturally inappropriate elements",
            "out-of-context objects or settings"
        ]
        
        if poem.time_period != "Not specified":
            negative_elements.append(f"elements not fitting {poem.time_period} time period")
        
        if poem.context != "Not specified":
            negative_elements.append(f"elements inconsistent with {poem.context}")
        
        return ", ".join(negative_elements)


    def identify_stanza_characters(self, stanza: str, all_characters: Dict[str, Character]) -> Dict[str, Character]:
        stanza_characters = {}
        for name, character in all_characters.items():
            if name.lower() in stanza.lower() or any(word.lower() in stanza.lower() for word in name.split()):
                stanza_characters[name] = character
        return stanza_characters


    def add_feedback(self, feedback: str):
        self.feedback_history.append(feedback)
        if len(self.feedback_history) > 5:
            self.feedback_history.pop(0)


class ImageGenerator:
    def generate_image(self, prompt: str, stanza: str, output_path: str, window, version: int = 1) -> str:
        try:
            update_output(window, "Generating illustration...")
            full_prompt = f"""
            Create an image based on the following description:

            {prompt}

            Ensure the image accurately reflects the described scene, mood, and any specified cultural or historical context.
            """
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=full_prompt,
                n=1,
                size="1024x1024",
                quality="hd",
                style="vivid"
            )
            
            image_url = response.data[0].url
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            
            self.imprint_stanza(image, stanza)
            
            image_path = f"{output_path}_v{version}.png"
            image.save(image_path, format='PNG')
            
            update_output(window, f"Image saved to: {image_path}")
            return image_path
        except Exception as e:
            update_output(window, f"Failed to generate image. Error: {str(e)}")
            return None

    def imprint_stanza(self, image: Image, stanza: str):
        try:
            draw = ImageDraw.Draw(image)
            font_size = 20
            font = ImageFont.truetype("arial.ttf", font_size)
            
            text_color = (255, 255, 255)  # White text
            shadow_color = (0, 0, 0)  # Black shadow
            padding = 10
            line_height = font_size + 5
            
            lines = self.wrap_text(stanza, font, image.width - 20)
            text_height = len(lines) * line_height
            
            if image.getpixel((10, 10))[0] < 128:  # Check if top-left pixel is dark
                y = padding  # Place text at the top
            else:
                y = image.height - text_height - padding  # Place text at the bottom
            
            for line in lines:
                shadow_position = (padding + 1, y + 1)
                text_position = (padding, y)
                draw.text(shadow_position, line, font=font, fill=shadow_color)
                draw.text(text_position, line, font=font, fill=text_color)
                y += line_height
        except Exception as e:
            print(f"Failed to imprint stanza on image. Error: {str(e)}")

    def wrap_text(self, text: str, font: ImageFont, max_width: int) -> List[str]:
        lines = []
        for paragraph in text.split('\n'):
            line = []
            for word in paragraph.split():
                line.append(word)
                w, h = font.getbbox(' '.join(line))[2:]
                if w > max_width:
                    if len(line) == 1:
                        lines.append(line.pop())
                    else:
                        lines.append(' '.join(line[:-1]))
                        line = [line[-1]]
            if line:
                lines.append(' '.join(line))
        return lines

class PoetryImageGenerator:
    def __init__(self, window):
        self.window = window
        self.context_analyzer = ContextAnalyzer()
        self.character_initializer = CharacterInitializer()
        self.poem_analyzer = PoemAnalyzer()
        self.prompt_generator = PromptGenerator()
        self.image_generator = ImageGenerator()

    def initialize_poem(self, poem_text: str, output_folder: str) -> Poem:
        update_output(self.window, "Initializing poem analysis...")
        start_time = time.time()
        poem = self.poem_analyzer.analyze_poem(poem_text, self.window)
        context = self.context_analyzer.analyze_context(poem_text, self.window)
        poem.theme = context['theme']
        poem.context = context['context']
        poem.time_period = context['time_period']
        poem.location = context['location']
        poem.style = context['style']
        
        update_output(self.window, "Identifying characters in the poem...")
        characters_info = self.poem_analyzer.identify_characters(poem_text, self.window)
        for char_info in characters_info:
            update_output(self.window, f"Initializing character: {char_info['name']}")
            character = self.character_initializer.initialize_character(char_info['name'], char_info['brief_description'], poem.context, self.window)
            if character.reference_image:
                safe_name = ''.join(c for c in char_info['name'] if c.isalnum() or c in (' ', '_')).rstrip()
                char_image_path = os.path.join(output_folder, f"character_{safe_name}.png")
                character.reference_image.save(char_image_path)
                update_output(self.window, f"Character image saved: {char_image_path}")
            poem.characters[character.name] = character
        
        end_time = time.time()
        update_output(self.window, f"Poem initialization completed in {end_time - start_time:.2f} seconds.")
        return poem

    def process_poem(self, poem_text: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        poem = self.initialize_poem(poem_text, output_folder)
        context = self.context_analyzer.analyze_context(poem_text, self.window)
        poem.context = context['cultural_context']
        poem.time_period = context['time_period']

        for i, stanza in enumerate(poem.stanzas):
            update_output(self.window, f"Processing stanza {i+1}/{len(poem.stanzas)}...")
            self.window['-PROGRESS-'].update(current_count=int((i+1)/len(poem.stanzas)*100))

            version = 1
            while True:
                prompt = self.prompt_generator.generate_prompt(stanza, poem, poem.characters, self.window)
                update_output(self.window, f"Generated prompt: {prompt}")

                image_path = self.image_generator.generate_image(prompt, stanza, os.path.join(output_folder, f"stanza_{i+1:02d}"), self.window, version)
                
                if image_path:
                    update_output(self.window, f"Image generated: {image_path}")
                    self.window['-IMAGE-'].update(filename=image_path)
                    
                    event, values = self.window.read()
                    if event == 'Accept':
                        update_output(self.window, "Image accepted.")
                        break
                    elif event == 'Regenerate':
                        update_output(self.window, "Regenerating image...")
                        version += 1
                    elif event == 'Exit' or event == sg.WINDOW_CLOSED:
                        update_output(self.window, "Exiting...")
                        return
                    
                    feedback = values['-FEEDBACK-']
                    if feedback:
                        self.prompt_generator.add_feedback(feedback)
                        update_output(self.window, f"Feedback added: {feedback}")
                else:
                    update_output(self.window, "Failed to generate image. Moving to next stanza.")
                    break

        update_output(self.window, "Poem processing complete.")

def create_window():
    sg.theme('LightBlue2')

    left_column = [
        [sg.Text("Poetry Image Generator", font=("Helvetica", 20))],
        [sg.Text("Enter your poem here:")],
        [sg.Multiline(size=(60, 10), key="-POEM-")],
        [sg.Text("Output Folder:"), sg.Input(key="-FOLDER-"), sg.FolderBrowse()],
        [sg.Button("Generate Images"), sg.Button("Exit")],
        [sg.Text("Status Messages:")],
        [sg.Multiline(size=(60, 10), key="-OUTPUT-", disabled=True, autoscroll=True)],
        [sg.ProgressBar(100, orientation='h', size=(20, 20), key='-PROGRESS-')]
    ]

    right_column = [
        [sg.Image(key='-IMAGE-', size=(400, 400))],
        [sg.Text("Feedback:"), sg.Input(key='-FEEDBACK-', size=(30, 1))],
        [sg.Button("Accept"), sg.Button("Regenerate")]
    ]

    layout = [
        [sg.Column(left_column), sg.VSeperator(), sg.Column(right_column, vertical_alignment='top')]
    ]

    return sg.Window("Poetry Image Generator", layout, finalize=True, resizable=True)

def update_output(window, message):
    window['-OUTPUT-'].print(message)
    window.refresh()

def main():
    window = create_window()

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Exit":
            break
        elif event == "Generate Images":
            poem_text = values["-POEM-"]
            output_folder = values["-FOLDER-"]
            if poem_text and output_folder:
                window['-OUTPUT-'].update("")  # Clear previous messages
                window['-PROGRESS-'].update(current_count=0)
                generator = PoetryImageGenerator(window)
                threading.Thread(target=generator.process_poem, args=(poem_text, output_folder), daemon=True).start()
            else:
                sg.popup_error("Please enter a poem and select an output folder.")

    window.close()

if __name__ == "__main__":
    main()                