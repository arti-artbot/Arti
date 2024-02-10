![17102301outputimage_cv](https://github.com/arti-artbot/Arti/assets/159595914/4fb3a936-73a0-4f81-b42a-e88ae78df2c0)


**ARTI**


Art comes in many beautiful forms, but you can only fit so many paintings on your wall. What looks interesting one day might fail to catch your attention the next. What if you could have infinitely diverse artwork generated every few minutes? What about artworked based on your conversations? What if it was completely powered by AI?
Welcome to this tutorial, where we will be showing you how to build Arti, your personal art-bot.\
Arti will capture your conversations and create images in real time. It’s powered by many forms of Generative AI, which help it create accurate yet distinct images relatable to the conversation at hand. It’s also completely open-source, meaning that you can create your own art-bot with just a Jetson.\
We created Arti to constantly generate inspirational, meaningful, and relevant works of art, and we hope you enjoy using it as much as we did. Let’s get started!

**What this tutorial will cover**
- Overview
- Project Specifications
- Creating Arti
- Downloadable Code
- Examples
- Notes
- Troubleshooting
- Conclusion

**Overview**\
Arti turns your conversations into images over three steps:\
First, it starts a new recording of a conversation every 60 seconds. This means the image will change every 60 seconds - you can adjust this according to your preferences. It will save the recording as an audio file and send it to Whisper. Whisper is an open-source speech-to-text software recently released by OpenAI, with automatic speech recognition built in. It will write the audio file into text.\
Next, LLaMa 2, an open-source large language model similar to ChatGPT, will translate this raw text into a prompt which is more understandable for Stable Diffusion (our image generation model, also open-source). This step will allow Stable Diffusion to generate more accurate images representing the topic of the conversation.\
Finally, Stable Diffusion will receive its prompt from LLaMa 2, and generate the image. In this way, your art bot will be able to generate images based on your conversation!\
We won’t be writing code in this tutorial, but you can access it at the bottom of this page under the Downloadable Code section. However, please read through everything first - your device may not satisfy the specifications for this project, and you may not have everything set up on your device.\
Since we didn’t want our conversations to have the potential to be shared, we stored everything in a local directory, which also enabled offline access.

**Project Specifications**
- Jetson: NVIDIA Jetson AGX Orin Dev Kit (or equivalent)
- Microphone: Any microphone that works with python - we used a webcam
- SSD: Samsung 990 Pro SSD 1TB (any SSD with at least 1TB should work)

Some notes:
- If you have never used a Jetson before, Nvidia has a great tutorial which we found very helpful: https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit
- We were using a NVIDIA Jetson AGX Orin, which is a great product that suited our needs perfectly. However, if you are using a different device, please ensure that it has at least 64G in its RAM/GPU and at least 1TB in the SSD. Any less than this will cause the project to crash. You can buy more memory for the SSD but unfortunately the quantity of RAM is fixed.
- We used Jetpack 6.0, so please make sure you have that installed.

Time to create!

**Creating Arti**\
Creating Arti is simple, as we’ve already written all the necessary code for you. Follow the steps below to get started with your art-bot.

**Step One: Initial Setup**\
Create a folder called Arti in your home directory. Then, create a new virtual environment within this folder. (Creating a virtual environment is important because the specific packages you’ll be installing in later steps may otherwise conflict with other projects.)\
In general, you can create a virtual environment by executing this command in your code editor’s terminal:\
`python -m venv /path/to/new/virtual/environment`\
We will put our virtual environment directly into the Arti folder. Our code will do everything else related to folder structure.

**Step Two: Whisper**\
First, we will download Whisper itself. We are using the model whisper-large-v3. If you want to learn more about Whisper and its various models, click here:  https://huggingface.co/openai/whisper-large-v3 \
We will now run some commands. Be careful to run them in your code editor’s terminal while in your project. Running these commands outside of your project means you will also run them outside of your virtual environment. This might cause project dependencies to conflict - which is not good!\
Once you’ve confirmed that your terminal is set to operate within your project, run the following commands (in order):\
`pip install --upgrade pip`\
`pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]`\
`huggingface-cli download openai/whisper-large-v3 --local-dir ./Models/Whisper_Model/`\
We won’t explain commands in great detail here, but the first one checks that your pip command is up to date, the second one downloads and upgrades our transformer, and the last one downloads and installs Whisper.

**Step Three: Recordings**\
In this step, we’ll download the requirements for taking the recordings used in Whisper. First, make sure your terminal is still set to run commands for your project (by default, the terminal stays where it was before, even if you closed your file).\
Now run these commands:\
`sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0`\
`sudo apt-get install -y --no-install-recommends ffmpeg`\
`pip install sounddevice`\
The first two commands install dependencies, and the last one installs sounddevice which is a library we’ll be using to take recordings with.

**Step Four: LLaMa 2**\
Now for LLaMa 2. This is the most complicated step in our installation process, but don’t be discouraged - just run the following commands. Again, double check that your terminal is in the right place and remember to run them in the correct order!\
`pip3 install huggingface-hub>=0.17.1`\
This first command installs a package related to HuggingFace, which is the development platform we are using to deploy our models - Whisper, LLaMa 2, and Stable Diffusion - to our device.\
`huggingface-cli download TheBloke/Llama-2-13B-chat-GGUF llama-2-13b-chat.Q6_K.gguf --local-dir ./Models/Llama_Model/ --local-dir-use-symlinks False`\
This command downloads LLaMa 2. We were using the model llama-2-13b-chat.Q6_K.gguf model, which is a version of LLaMa 2 trained on 13 billion parameters, along with some other specifications. You can find out more about this and other models here: https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF\
There are models trained on 7 billion, 13 billion, and 70 billion parameters. We are using a quantised version of a model trained on 13 billion parameters, as it is fast, performs well, and is less memory-intensive than other models. However, feel free to experiment with another model.

**Step Five: Stable Diffusion**\
As always, check that you are in the correct project.\
First we will install requirements for Stable Diffusion:\
`pip install diffusers --upgrade`\
`pip install invisible_watermark transformers accelerate safetensors`\
Now, we will install Stable Diffusion itself:\
`huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ./Models/Stable_Diffusion_Model`\
We installed it to a local directory because we want to be able to run this project while offline. This also enhances privacy.\
We were using the model stable-diffusion-xl-base-1.0 and like before, this is just what worked best for us but you can always change this in the code. We’ve provided a link which you can use to learn about the different models: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

**Step Six: Torch**\
Now we’ll install Torch, an open-source machine learning library which our various models will use to operate.\
Run these commands:\
`pip uninstall torch`\
`pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+81ea7a4.nv23.12-cp310-cp310-linux_aarch64.whl`\
The first command actually uninstalls Torch. We found this necessary, because a version of Torch was already installed, and the NVIDIA Jetson requires a specific architecture of Torch.\
The other two commands install the version of Torch we want.

**Step Seven: Running the Code**\
You’re ready to download and run our code! You can find this in the next section. Since you already have everything downloaded and set up, all you need to do is download the file arti.py and save it to the Arti folder. Simply click the Run button (usually found on the menu at the top of your code editor).\
The first time you run our code, it will take a while to load up and generate the first image - this is because it has to set up everything you downloaded in the previous steps of this section. However, the next time everything will be downloaded and it will start recording within a few moments.\
While it loads, be sure not to click anywhere on the screen (moving your mouse is fine).\
When you run the program, you will see a splash screen while it generates the first image. This shouldn’t take more than a few minutes.\
If you encounter any problems please see the Troubleshooting section located near the end of this tutorial. We haven’t listed every error, but tried to tackle the most common ones.

**Downloadable Code**\
You can find our code right here at our GitHub page, simply clone this repository to your device. Remember to store everything directly in the Arti folder.

**Examples**\
Following are a few of our favorite examples of the images Arti generated, along with the conversations taking place at those times.

**EXAMPLE 1:**

In this example, two people were having a discussion about nuclear fission and reactors.\
**Conversation:** and this lighter isotope is less tightly bound. Wait, what? Compared to its... And this light... Wait, wait, what happened here? ...lighter isotope is less tightly bound. That's an isotope. Compared to its... And this, but roughly one in every 140, lacks three neutrons, and this lighter isotope is less tightly bound. It's not on the... Compared to its more abundant cousin, a strike by a neutron easily splits the U-235 nuclei into lighter radioactive elements called fission products, in addition to two to three neutrons, gamma rays, and a few neutrinos. During fission, some nuclear mass transforms into energy. A fraction of the newfound energy powers the fast moving neutrons. And if some of them strike uranium nuclei, fission results in a second larger generation of neutrons. So that's a little bit very complicated. It's very complicated. It's not . So there's three types of radioactivity that happen with uranium, alpha, beta, and gamma. And so the gamma first starts and then does these other two. It's more complicated than this. This video will not really explain it very well. But it starts a nuclear reaction. The gamma starts, you know, things that happen. Yeah, but maybe I can more actually show what it looks like. If you want actual uranium? Like this. Well, you want to know the nuclear... No, like this. You want to know the uranium really actually, how it works. Uranium decay, okay? That's what it is. That's if you really want to... Don't take each of your autonomous. This is how uranium becomes the chain reaction. It's a physics one, but it might be complicated. I don't know, just go back for one second. Yeah, this one. This is fission. There's something called fission and fusion in nuclear reactions. Do you want to know? Let's try this one, and if it's too hard. Yeah, but there's another one. Actually, how it works. Do you want to know? Let's try this one if it's too hard. Yeah, but there's another one. Actually how it works. Not like, hmm, but the thing is. Like actually how it works. But do you want to know the reaction of the radio? No, it just goes back to normal. So this new Wi-Fi camera is taking the US by storm. This brand new subscription. Electrobel has seven nuclear power plants, four in Doule and three in Tiange, covering half of the electricity consumption in Belgium without producing CO2. But how exactly does a nuclear power plant work? A nuclear power plant works to a large extent exactly does a nuclear power plant work? A nuclear power plant works to a large extent like a conventional thermal power plant. Water is converted into steam which drives a turbine connected to a generator. This generator converts the mechanical energy into electrical energy. The only difference is that the heat which converts water into steam is produced by nuclear fission and not by burning coal, natural gas or biomass. The nuclear power plants of Doubs and Thiages use fissile uranium.\
**LLaMa's Interpretation:** A nuclear power plant with a glowing core and steam rising from its cooling towers, surrounded by a futuristic cityscape with sleek skyscrapers and neon lights.\
**Image generated:** ![74824001outputimage_cv3](https://github.com/arti-artbot/Arti/assets/159595914/af63149c-3b04-4250-ab20-0cb7a0e33964)

**EXAMPLE 2:**

In this example Whisper didn't interpret all the audio correctly - for example, "San Diego" was its interpretation of an alarm going off on a phone - which was one of the reasons why we chose to use LLaMa 2.\
**Conversation:**  This is what San Diego San Diego San Diego San Diego San Diego San Diego San Diego San Diego San Diego San Diego San Diego San Diego San Diego I'm sorry. Where? I want to. Dad? Yeah? Mom needs you for pancakes. In the middle of something, man. Do I have to? It's fine. They don't want pancakes. That's like, I don't want to. Mom. Mom, I'm in the middle of something with rain. I don't want to do pancakes right now. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. Mom. I don't want to do pancakes right now. Thank you. I'm sorry. . . . . . . . . . . . . . . . . . Remember, because he's doing the virtual environment.\
**LLaMa's Interpretation:** A young boy, surrounded by rain and fog, reluctantly helps his mom make pancakes while using a virtual reality headset.\
**Image generated:** ![88621753outputimage_cv3](https://github.com/arti-artbot/Arti/assets/159595914/dc98932e-1796-4381-8605-602fcbb3ff70)

**EXAMPLE 3:**

In this example a video about training autonomous cars was being played.\
**Conversation:**  not reversing the vehicle, and not requiring online calculations. Additionally, where all previous methods have used low constant speeds, our method uses variable speeds up to 6 m per second and ensures the vehicle remains within the friction limit. We presented a method of safe learning that reformulates reinforcement learning to incorporate the supervisor. The safe learning method was evaluated in the F-10-1-10 simulator at speeds of up to 6 meters per second. The results showed that safe learning presents a 5x or 5 times improvement in sample efficiency, requiring only 10,000 steps. The supervisor and the learning formulation effectively train the agent to not require supervision. The safe learning agents select lower speed profiles than the conventional learning agents. This results in the safe learning agents achieving slower lap times and higher success rates. A major advantage of our methods is that the vehicles never crash during training. Future work should use this method to train agents on board physical vehicles. The ability to train agents for high performance robotic control while ensuring safety during the training process means that these methods can be used to train deep reinforcement learning agents on real world robots, thus bypassing the sim to real problem. Future work should evaluate how well safe learning uses the supervisor performs using the supervisor performs on real-world high-performance platforms. Bypassing a simple gap will mean that there is no difference between the training and testing behavior since both will be on the same physical device. The improvement in sample efficiency means that it is easier to use deep reinforcement learning since training time is reduced. Training more conservative policies leads to safer solutions which are essential for\
**LLaMa's Interpretation:** Generate an image of a high-performance robotic vehicle navigating a challenging track while ensuring safety during training.\
**Image generated:** ![96561433outputimage_cv3](https://github.com/arti-artbot/Arti/assets/159595914/541544bf-4d48-4961-a92b-5bbb681b2480)


**Notes**
- You can hit Q if you don’t want to see the images in full screen anymore. This will return you to the code editor.
- You don’t have to use the models we did as long as your device is powerful enough. Any versions of Whisper, LLaMa 2, and Stable Diffusion are fine. Our recommendations are just what worked best for us but feel free to experiment.
- We set Whisper’s default language to English, but it will give you the option to set it to whatever language you want. Note that LLaMa 2 will still give prompts in English, because Stable Diffusion works best in this language.
- If you would like to, you can always listen to the recording, view the text, and/or see LLaMa 2’s response for a specific image, or even see past images. Our code automatically creates several folders within the Arti folder. Each folder stores something different (e.g., the Image_Files folder stores all the outputted images). You can access old files by going into these folders.

**Troubleshooting**

**Uninstalling or resetting**: Should you ever need to uninstall and reinstall, you can go about it in two ways. Either enter these commands into the code editor’s terminal:\
`pip uninstall [package name]`\
`pip install [package name]`\
Or, if you’re having a problem with our code or if you want to start over entirely, just delete the Arti folder or its problematic portion. Everything will still be installed.\
Of course, remember to replace [package name] with the name of your package.

**“Error: File not found”** or **“Error: ImportError”**: This means you installed something in the wrong location, so the code can’t access it. Take a quick look at the Creating Arti section of this tutorial and move your files/folders around as necessary to match what we did. It could also mean you aren’t running commands from the right place, so double-check that you’re in the Arti folder.

**“Error: Module Not Found”**: This means you haven’t installed one of our packages. Use the following command to install each missing package:\
`pip install [package name]`\
Remember to replace [package name] with the real name of the package!

**Whisper is not recording effectively or at all**: First, make sure that your microphone is not the problem. Then, double check that you have set the appropriate default language - we set it to English. Finally, make sure you are speaking clearly and close to the microphone. If the problem persists, try uninstalling and reinstalling Whisper.

**The image is not showing in fullscreen mode**: In your code editor’s terminal, run the following command:\
`python arti.py`\
Remember, you can always go back to your code editor by hitting Q.

**Conclusion**\
You’ve successfully created your very own AI artist - congratulations. We hope you enjoy using it! Remember that it’s not perfectly accurate, but you may find the implications entertaining.

Thank you and happy generating!
