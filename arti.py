# Importing libraries for audio recordings
import glob
import sounddevice as sd
from scipy.io.wavfile import write
import random
import os

# Importing libraries for Whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Importing libraries for LLaMa 2
from llama_cpp import Llama
import torch

# Importing libraries for Stable Diffusion
from diffusers import AutoPipelineForText2Image
import cv2
import numpy as np


class NoWatermark:
    def apply_watermark(self, img):
        return img


# Set necessary variables
fs = 44100
recording_seconds = 600

# Create and define directories
audio_dir = "./Audio_Files/"
image_dir = "./Image_Files/"
whisper_text_dir = "./Whisper_Texts/"
llama_text_dir = "./Llama_Texts/"
if not os.path.isdir(audio_dir):
    os.mkdir(audio_dir)
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
if not os.path.isdir(whisper_text_dir):
    os.mkdir(whisper_text_dir)
if not os.path.isdir(llama_text_dir):
    os.mkdir(llama_text_dir)

opencv_window_name = 'Arti'
cv2.namedWindow(opencv_window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(opencv_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Start a recording while the model is loading for the first time
print("Initial recording started")
my_recording = sd.rec(int(recording_seconds * fs), samplerate=fs, channels=2)

# Display first splash screen
splash_screen_1 = cv2.imread("./Splash_Screens/splash_screen_1.png")
cv2.imshow(opencv_window_name, splash_screen_1)
cv2.waitKey(1)


device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Specifying where to get memory from
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Whisper specifications:
whisper_model_id = "openai/whisper-large-v3"  # Specifying what Whisper model to use
whisper_prompt = "hacker in a room, muted colors, detailed, 8k"

# Stable Diffusion specifications:
stable_diffusion_prompt_pre = ""
stable_diffusion_negative_prompt = "Cartoon, Comic, Nudity, People, Humans"  # Setting negative prompt


# Below, we are going to load the models

# Load Whisper:
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "./Models/Whisper_Model/", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True
    )
whisper_model.to(device)
whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Load LLama:
llm = Llama(
    model_path="./Models/Llama_Model/llama-2-13b-chat.Q6_K.gguf",
    n_gpu_layers=-1,  # Using GPU acceleration
    n_ctx=2048  # Increasing the context window
)

# Load Stable Diffusion:
pipeline_text2image = AutoPipelineForText2Image.from_pretrained("./Models/Stable_Diffusion_Model/",
                                                                torch_dtype=torch.float16, variant="fp16",
                                                                use_safetensors=True, local_files_only=True).to("cuda")
pipeline_text2image.watermark = NoWatermark()

first_run = True

# Create recordings
while True:
    # Start a recording
    print("Starting recording loop")
    sd.stop()
    for x in range(1, recording_seconds):
        if my_recording[x * fs][0] == 0:
            my_recording = my_recording[0:x * fs]
            break
    write(audio_dir + str(random.randint(0, 100000000)) + 'output.mp3', fs, my_recording)
    print("Recording finished.")

    # Display second splash screen
    if first_run:
        splash_screen_2 = cv2.imread("./Splash_Screens/splash_screen_2.png")
        cv2.imshow(opencv_window_name, splash_screen_2)
        first_run = False
        cv2.waitKey(1)

    # Start another recording while the image is being generated
    print("Recording started.")
    my_recording[:] = 0
    my_recording = sd.rec(int(recording_seconds * fs), samplerate=fs, channels=2)

    # Save the first recording
    file_type = r'*.mp3'
    files = glob.glob(audio_dir + file_type)
    max_file = max(files, key=os.path.getctime)
    print("Last recording taken: ", max_file)

    audio_results = whisper_pipe(max_file, generate_kwargs={"language": "english"})
    whisper_prompt = audio_results["text"]
    whisper_prompt = stable_diffusion_prompt_pre + whisper_prompt

    # Give the LLM its prompt
    llm_prompt = ("[INST] <<SYS>>\nYou will be provided with a conversation. Your task is to generate a simple prompt for an AI image generator based on the information in the conversation. You should only provide a single prompt. Do not include any other information as a repsonse. Only respond with the prompt so it can be used as input for an AI generative model. Do not include anything related to nudity in your prompt.\n<</SYS>>\n{" + whisper_prompt + "}[/INST]")
    llm_response = ""

    # Create LLM response
    while llm_response == "":
        llm_response = llm(llm_prompt, max_tokens=512)
        print("LLM_response:", llm_response)
    stable_diffusion_prompt = llm_response['choices'][0]["text"].split("\n")[-1]  # Stable Diffusion's next prompt

    # Print outputs (for user's convenience)
    print("Whisper's output: ", whisper_prompt)
    print("Stable diffusion's prompt: ", stable_diffusion_prompt)

    # Stable Diffusion creating the image:
    image = pipeline_text2image(prompt=stable_diffusion_prompt, negative_prompt=stable_diffusion_negative_prompt, height=1024, width=int(int(1024 * 16 / 9 / 8) * 8)).images[0]

    opencv_converted_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow(opencv_window_name, opencv_converted_image)  # Show the image

    # Save audio, text, and image files
    print("Saving ...")
    cv2.imwrite(image_dir + max_file[:-4].split("/")[-1] + "image.jpg", opencv_converted_image)
    whisper_file = open(whisper_text_dir + max_file[:-4].split("/")[-1] + "whisper_text.txt", "w+")
    whisper_file.write(whisper_prompt)
    llama_file = open(llama_text_dir + max_file[:-4].split("/")[-1] + "llama_text.txt", "w+")
    llama_file.write(stable_diffusion_prompt)

    # If Q was pressed to quit fullscreen mode:
    print("Checking if Q pressed")
    k = cv2.waitKey(1)
    if k == ord('q'):
        cv2.destroyAllWindows()
        break

print("End")
