import os
import io
from io import BytesIO
import base64
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont, ImageTk
import requests
from openai import OpenAI
import anthropic
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import json
import re
import threading
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI and Anthropic clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Character:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.reference_image = None
        self.appearance_details = ""

class Poem:
    def __init__(self, text: str):
        self.text = text
        self.stanzas = []
        self.characters: Dict[str, Character] = {}
        self.context = ""
        self.time_period = ""
        self.cultural_setting = ""

class ContextAnalyzer:
    def analyze_context(self, poem_text: str) -> Dict[str, str]:
        print("Analyzing poem context...")
        context_prompt = f"""Analyze the following poem and provide:
        1. The main theme or subject of the poem
        2. The cultural or mythological context (if any)
        3. The setting or time period (if applicable)
        4. Any significant symbols or motifs

        Poem:
        {poem_text}

        Provide your analysis as a clear, structured response with labels for each aspect."""
        
        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": context_prompt}]
        )
        
        analysis_text = response.content[0].text.strip()
        
        # Parse the response manually
        context = {
            "theme": "Not specified",
            "cultural_context": "Not specified",
            "time_period": "Not specified",
            "symbols": "None identified"
        }
        
        for line in analysis_text.split('\n'):
            line = line.strip()
            if line.startswith("Theme:"):
                context["theme"] = line.split("Theme:", 1)[1].strip()
            elif line.startswith("Cultural context:"):
                context["cultural_context"] = line.split("Cultural context:", 1)[1].strip()
            elif line.startswith("Setting/Time period:"):
                context["time_period"] = line.split("Setting/Time period:", 1)[1].strip()
            elif line.startswith("Symbols/Motifs:"):
                context["symbols"] = line.split("Symbols/Motifs:", 1)[1].strip()
        
        print(f"Context analysis:\n{json.dumps(context, indent=2)}")
        return context

class CharacterInitializer:
    def initialize_character(self, name: str, description: str) -> Character:
        character = Character(name, description)
        prompt = f"Create a detailed, high-quality portrait of {description}. Ensure the face is clear and detailed, with natural-looking eyes. The image should be suitable as a reference for consistent character depiction across multiple scenes. Avoid any anachronistic elements."
        
        try:
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="hd",
                style="vivid"
            )

            image_url = response.data[0].url
            image_response = requests.get(image_url)
            character.reference_image = Image.open(BytesIO(image_response.content))
            
            # Generate detailed appearance description
            appearance_prompt = f"Describe the appearance of {name} in great detail, including facial features, body type, clothing, and any distinguishing characteristics. This description will be used to maintain consistency across multiple images."
            appearance_response = claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": appearance_prompt}]
            )
            character.appearance_details = appearance_response.content[0].text.strip()
            
        except Exception as e:
            print(f"Error initializing character {name}: {str(e)}")
            character.reference_image = None
        
        return character




class PoemAnalyzer:
    def analyze_poem(self, poem_text: str) -> Poem:
        poem = Poem(poem_text)
        poem.stanzas = self.identify_stanzas(poem_text)
        context_analysis = self.analyze_poem_context(poem_text)
        poem.context = context_analysis.get('context', 'Unknown context')
        poem.time_period = context_analysis.get('time_period', 'Unknown time period')
        poem.cultural_setting = context_analysis.get('cultural_setting', 'Unknown cultural setting')
        return poem

    def identify_characters(self, poem_text: str) -> List[Dict[str, str]]:
        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"Identify key characters in this poem. For each character, provide a name and a detailed physical description. Format your response as a Python list of dictionaries, where each dictionary has 'name' and 'description' keys. Only include characters that are explicitly mentioned or clearly implied in the poem. For example: [{{\"name\": \"Character Name\", \"description\": \"Character Description\"}}, ...]\n\n{poem_text}"}
            ]
        )
        
        content = response.content[0].text
        
        try:
            characters = json.loads(content)
            if isinstance(characters, list):
                return characters
        except json.JSONDecodeError:
            print("Failed to parse the character list as JSON.")
        
        # If parsing fails, try to manually extract character information
        characters = []
        lines = content.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    name = parts[0].strip().strip('"')
                    description = parts[1].strip().strip('"')
                    characters.append({"name": name, "description": description})
        
        if not characters:
            print("No specific characters identified in the poem.")
        
        return characters

    def identify_stanzas(self, poem_text: str) -> List[str]:
        return [stanza.strip() for stanza in poem_text.split('\n\n') if stanza.strip()]


    def analyze_poem_context(self, poem_text: str) -> Dict[str, str]:
        try:
            response = claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"Analyze the following poem and provide:\n1. A brief summary of its overall context, including cultural and thematic elements.\n2. The specific time period the poem is set in or refers to.\n3. The cultural setting or background of the poem.\n\nFormat your response as a Python dictionary with keys 'context', 'time_period', and 'cultural_setting'.\n\n{poem_text}"}
                ]
            )
            
            # Try to parse the response as JSON
            try:
                context_dict = json.loads(response.content[0].text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the dictionary manually
                content = response.content[0].text
                context_dict = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        context_dict[key.strip().strip("'")] = value.strip().strip("'")

            # Validate and fill missing keys
            required_keys = ['context', 'time_period', 'cultural_setting']
            for key in required_keys:
                if key not in context_dict:
                    context_dict[key] = f"Unknown {key}"

            return context_dict
        except Exception as e:
            print(f"An error occurred during poem context analysis: {str(e)}")
            return {
                'context': 'Unknown context',
                'time_period': 'Unknown time period',
                'cultural_setting': 'Unknown cultural setting'
            }


class PromptGenerator:
    def __init__(self):
        self.feedback_history = []

    def generate_prompt(self, stanza: str, poem: Poem, characters: Dict[str, Character]) -> str:
        print("Analyzing stanza and generating prompt...")
        start_time = time.time()

        stanza_characters = self.identify_stanza_characters(stanza, characters)
        character_descriptions = "\n".join([f"{name}: {char.appearance_details}" for name, char in stanza_characters.items()])
        
        context_prompt = f"""Generate a vivid, creative description for an illustration based on the following stanza and context:

        Stanza: "{stanza}"

        Overall poem context: {poem.context}
        Time period: {poem.time_period}
        Cultural setting: {poem.cultural_setting}

        Characters present in this stanza (only include if explicitly mentioned):
        {character_descriptions}

        Previous feedback: {' '.join(self.feedback_history)}

        Instructions:
        1. Focus on creating a scene that accurately represents the mythological or thematic context.
        2. Emphasize cultural authenticity in all aspects (clothing, architecture, objects, etc.).
        3. Create a detailed, realistic scene rather than a cartoonish representation.
        4. Use rich, vibrant colors associated with the cultural context.
        5. Include traditional symbols and elements relevant to the story or theme.
        6. Ensure all individuals have features and appearances consistent with the cultural and historical context.
        7. Only include characters explicitly mentioned or clearly implied in the stanza.
        8. For generic scenes, focus on the overall atmosphere rather than specific individuals.
        9. Maintain consistency with previously described character appearances.
        10. Incorporate any relevant previous feedback to improve the image.

        Provide a concise description (75-100 words) for an illustration that vividly and creatively captures the stanza's essence, suitable for an AI image generation model."""
        
        print("Sending prompt to Claude AI for processing...")
        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": context_prompt}
            ]
        )
        
        prompt = response.content[0].text.strip()
        
        # Generate dynamic negative prompts based on the poem's context
        negative_prompt = self.generate_negative_prompt(poem)
        
        end_time = time.time()
        print(f"Prompt generation completed in {end_time - start_time:.2f} seconds.")
        
        return f"{prompt}\nNegative prompt: {negative_prompt}"

    def identify_stanza_characters(self, stanza: str, all_characters: Dict[str, Character]) -> Dict[str, Character]:
        print("Identifying characters in the stanza...")
        stanza_characters = {}
        for name, character in all_characters.items():
            if name.lower() in stanza.lower() or any(word.lower() in stanza.lower() for word in name.split()):
                stanza_characters[name] = character
        print(f"Characters identified: {', '.join(stanza_characters.keys())}")
        return stanza_characters

    def generate_negative_prompt(self, poem: Poem) -> str:
        negative_elements = [
            "low quality", "blurry", "distorted faces", "anachronistic elements",
            "historically inaccurate details", "culturally inappropriate elements",
            "out-of-context objects or settings"
        ]
        
        if poem.time_period != "unspecified":
            negative_elements.append(f"elements not fitting {poem.time_period} time period")
        
        if poem.cultural_setting != "unspecified":
            negative_elements.append(f"elements inconsistent with {poem.cultural_setting} culture")
        
        return ", ".join(negative_elements)

    def add_feedback(self, feedback: str):
        self.feedback_history.append(feedback)
        if len(self.feedback_history) > 5:  # Keep only the last 5 feedback items
            self.feedback_history.pop(0)
        print(f"Feedback added: {feedback}")

class ImageGenerator:
    def generate_image(self, prompt: str, stanza: str, output_path: str, version: int = 1) -> str:
        try:
            print("Generating illustration...")
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="hd",
                style="vivid"
            )
            
            image_url = response.data[0].url
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            
            # Imprint stanza on the image
            self.imprint_stanza(image, stanza)
            
            image_path = f"{output_path}_v{version}.png"
            image.save(image_path, format='PNG')
            
            print(f"Image saved to: {image_path}")
            return image_path
        except Exception as e:
            print(f"Failed to generate image. Error: {str(e)}")
            return None

    def imprint_stanza(self, image: Image, stanza: str):
        try:
            draw = ImageDraw.Draw(image)
            font_size = 20
            font = ImageFont.truetype("arial.ttf", font_size)
            
            # Determine text placement (top or bottom)
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


class ImprovedUserInterface:
    def __init__(self, master):
        self.master = master
        self.master.title("Poetry Image Generator")
        self.master.geometry("1200x800")

        # Create left and right frames
        self.left_frame = ttk.Frame(master, padding="10")
        self.right_frame = ttk.Frame(master, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Left frame components
        self.poem_text = scrolledtext.ScrolledText(self.left_frame, wrap=tk.WORD, width=40, height=20)
        self.poem_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.status_text = scrolledtext.ScrolledText(self.left_frame, wrap=tk.WORD, width=40, height=20, state='disabled')
        self.status_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.generate_button = ttk.Button(self.left_frame, text="Generate Images", command=self.process_poem)
        self.generate_button.pack(pady=10)

        # Right frame components (initially hidden)
        self.image_label = ttk.Label(self.right_frame)
        self.feedback_text = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD, width=40, height=5)
        self.accept_button = ttk.Button(self.right_frame, text="Accept", command=self.accept)
        self.regenerate_button = ttk.Button(self.right_frame, text="Regenerate", command=self.regenerate)
        self.exit_button = ttk.Button(self.right_frame, text="Exit", command=self.exit_program)

        self.generator = PoetryImageGenerator(self)
        self.feedback_received = threading.Event()
        self.stop_event = threading.Event()

    def update_status(self, message):
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')
        self.master.update_idletasks()

    def show_image_and_controls(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((400, 400))  # Resize image if too large
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
        self.image_label.pack(pady=10)

        self.feedback_text.pack(padx=10, pady=10, fill=tk.X)
        self.accept_button.pack(side=tk.LEFT, padx=5)
        self.regenerate_button.pack(side=tk.LEFT, padx=5)
        self.exit_button.pack(side=tk.LEFT, padx=5)

    def hide_image_and_controls(self):
        self.image_label.pack_forget()
        self.feedback_text.pack_forget()
        self.accept_button.pack_forget()
        self.regenerate_button.pack_forget()
        self.exit_button.pack_forget()

    def accept(self):
        self.result = "accept"
        self.feedback_received.set()

    def regenerate(self):
        self.result = "regenerate"
        self.feedback_received.set()

    def exit_program(self):
        self.result = "exit"
        self.feedback_received.set()
        self.stop_event.set()
        self.master.quit()

    def process_poem(self):
        poem_text = self.poem_text.get("1.0", tk.END).strip()
        if not poem_text:
            messagebox.showwarning("Warning", "Please enter the poem before proceeding.")
            return

        output_folder = filedialog.askdirectory(title="Select Output Folder")
        if not output_folder:
            messagebox.showwarning("Warning", "No folder selected. Operation cancelled.")
            return

        self.update_status("Starting poem processing...")
        threading.Thread(target=self.generator.process_poem, args=(poem_text, output_folder)).start()

    def wait_for_feedback(self, image_path):
        self.master.after(0, lambda: self.show_image_and_controls(image_path))
        self.update_status("Waiting for user feedback. Please review the image and provide your input.")
        self.feedback_received.wait()
        feedback = self.feedback_text.get("1.0", tk.END).strip()
        self.update_status(f"Feedback received: {self.result}")
        self.hide_image_and_controls()
        self.feedback_received.clear()
        return self.result, feedback

class PoetryImageGenerator:
    def __init__(self, ui):
        self.context_analyzer = ContextAnalyzer()
        self.character_initializer = CharacterInitializer()
        self.poem_analyzer = PoemAnalyzer()
        self.prompt_generator = PromptGenerator()
        self.image_generator = ImageGenerator()
        self.ui = ui

    def initialize_poem(self, poem_text: str) -> Poem:
        self.ui.update_status("Initializing poem analysis...")
        start_time = time.time()
        poem = self.poem_analyzer.analyze_poem(poem_text)
        self.ui.update_status("Identifying characters in the poem...")
        characters_info = self.poem_analyzer.identify_characters(poem_text)
        for char_info in characters_info:
            self.ui.update_status(f"Initializing character: {char_info['name']}")
            character = self.character_initializer.initialize_character(char_info['name'], char_info['description'])
            poem.characters[character.name] = character
        end_time = time.time()
        self.ui.update_status(f"Poem initialization completed in {end_time - start_time:.2f} seconds.")
        return poem

    def process_poem(self, poem_text: str, output_folder: str):
        try:
            poem = self.poem_analyzer.analyze_poem(poem_text)
            self.ui.update_status(f"Poem analysis complete. Identified {len(poem.stanzas)} stanzas.")

            for i, stanza in enumerate(poem.stanzas):
                if self.ui.stop_event.is_set():
                    self.ui.update_status("Processing stopped by user.")
                    break

                self.ui.update_status(f"\nProcessing stanza {i+1}/{len(poem.stanzas)}...")
                self.ui.update_status(f"Stanza content:\n{stanza}")

                version = 1
                while True:
                    if self.ui.stop_event.is_set():
                        break

                    try:
                        self.ui.update_status("Generating prompt...")
                        detailed_prompt = self.prompt_generator.generate_prompt(stanza, poem, poem.characters)
                        self.ui.update_status(f"Generated prompt:\n{detailed_prompt}")

                        self.ui.update_status("Generating image...")
                        image_path = self.image_generator.generate_image(detailed_prompt, stanza, os.path.join(output_folder, f"stanza_{i+1:02d}"), version)
                        if image_path is None:
                            raise Exception("Image generation failed")

                        self.ui.update_status(f"Image generated: {image_path}")

                        result, feedback = self.ui.wait_for_feedback(image_path)

                        if result == "accept":
                            self.ui.update_status(f"Image approved. Feedback: {feedback}")
                            self.prompt_generator.add_feedback(feedback)
                            break
                        elif result == "regenerate":
                            self.ui.update_status(f"Regenerating image. Feedback: {feedback}")
                            self.prompt_generator.add_feedback(feedback)
                            version += 1
                        elif result == "exit":
                            self.ui.update_status("Exiting the program.")
                            return
                    except Exception as e:
                        self.ui.update_status(f"Error processing stanza {i+1}: {str(e)}")
                        if not messagebox.askyesno("Error", f"An error occurred: {str(e)}. Do you want to continue with the next stanza?"):
                            return

            self.ui.update_status("\nProcessing complete.")
        except Exception as e:
            self.ui.update_status(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    # Check if API keys are available
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        messagebox.showerror("Error", "API keys not found. Please check your .env file.")
        exit(1)

    root = tk.Tk()
    app = ImprovedUserInterface(root)
    root.mainloop()