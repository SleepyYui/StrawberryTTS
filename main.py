import os

import gradio as gr
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
import string

np.random.seed(0)

import ljspeechimportable
import styletts2importable

# from tortoise.utils.text import split_and_recombine_text

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)
# read all audio files from voices folder
voicelist = []
for filename in os.listdir('voices'):
    if filename.endswith(".wav"):
        voicelist.append(filename[:-4])
voices = {}
import phonemizer

global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

import pickle

if os.path.exists('voices.pkl'):
    print('Loading voices from pickle')
    with open('voices.pkl', 'rb') as f:
        voices = pickle.load(f)
    # check if all voices are there
    all_voices = True
    for v in voicelist:
        if v not in voices:
            all_voices = False
            voices[v] = styletts2importable.compute_style(f'voices/{v}.wav')
    if not all_voices:
        print('Saving voices to pickle')
        with open('voices.pkl', 'wb') as f:
            pickle.dump(voices, f)
else:
    for v in voicelist:
        voices[v] = styletts2importable.compute_style(f'voices/{v}.wav')
    print('Saving voices to pickle')
    with open('voices.pkl', 'wb') as f:
        pickle.dump(voices, f)


def split_and_recombine_text(text):
    tokenized = sent_tokenize(text)
    print(tokenized)

    # replace "..." with " " to avoid splitting on ellipses.
    tokenized = [t.replace("...", " ") for t in tokenized]
    print(tokenized)

    # filter out empty strings and strings containing only punctuation.
    """for token in tokenized:
        if token == " ":
            pass
        elif token.strip() == "" or all([c in string.punctuation for c in token]):
            tokenized.remove(token)"""
    tokenized = [t.translate(str.maketrans('', '', string.punctuation)) for t in tokenized if t.strip() != ""]
    print(tokenized)

    return tokenized


# split text into multiple parts, each not longer than 300 characters but still full sentences


def synthesize(text, voice, multispeakersteps):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    # if len(global_phonemizer.phonemize([text])) > 300:
    if len(text) > 300:
        raise gr.Error("Text must be under 300 characters")
    v = voice.lower()
    # return (24000, styletts2importable.inference(text, voices[v], alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1))
    return (24000,
            styletts2importable.inference(text, voices[v], alpha=0.3, beta=0.7, diffusion_steps=multispeakersteps,
                                          embedding_scale=1))


def longsynthesize(text, voice, lngsteps, progress=gr.Progress()):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if lngsteps > 25:
        raise gr.Error("Max 25 steps")
    if lngsteps < 5:
        raise gr.Error("Min 5 steps")
    texts = split_and_recombine_text(text)
    v = voice.lower()
    audios = []
    for t in progress.tqdm(texts):
        audios.append(styletts2importable.inference(t, voices[v], alpha=0.3, beta=0.7, diffusion_steps=lngsteps,
                                                    embedding_scale=1))
    return (24000, np.concatenate(audios))


def ljlongsynthesize(text, progress=gr.Progress()):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    texts = split_and_recombine_text(text)
    audios = []
    noise = torch.randn(1, 1, 256).to('cuda' if torch.cuda.is_available() else 'cpu')
    for t in progress.tqdm(texts):
        audios.append(
            ljspeechimportable.inference(t, noise, diffusion_steps=7, embedding_scale=1))
    return (24000, np.concatenate(audios))


def clsynthesize(text, voice, vcsteps):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    # if global_phonemizer.phonemize([text]) > 300:
    if len(text) > 400:
        raise gr.Error("Text must be under 400 characters")
    # return (24000, styletts2importable.inference(text, styletts2importable.compute_style(voice), alpha=0.3, beta=0.7, diffusion_steps=20, embedding_scale=1))
    return (24000, styletts2importable.inference(text, styletts2importable.compute_style(voice), alpha=0.3, beta=0.7,
                                                 diffusion_steps=vcsteps, embedding_scale=1))


def ljsynthesize(text):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    # if global_phonemizer.phonemize([text]) > 300:
    if len(text) > 400:
        raise gr.Error("Text must be under 400 characters")
    noise = torch.randn(1, 1, 256).to('cuda' if torch.cuda.is_available() else 'cpu')
    return (24000, ljspeechimportable.inference(text, noise, diffusion_steps=7, embedding_scale=1))


with gr.Blocks() as libritts:
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Text",
                             info="What would you like StrawberryTTS to read?",
                             interactive=True)
            voice = gr.Dropdown(voicelist, label="Voice", info="Select a default voice.", value='m-us-2',
                                interactive=True)
            multispeakersteps = gr.Slider(minimum=5, maximum=15, value=7, step=1, label="Diffusion Steps",
                                          info="Higher = better quality, but slower", interactive=True)
            # use_gruut = gr.Checkbox(label="Use alternate phonemizer (Gruut) - Experimental")
        with gr.Column(scale=1):
            btn = gr.Button("Synthesize", variant="primary")
            audio = gr.Audio(interactive=False, label="Synthesized Audio")
            btn.click(synthesize, inputs=[inp, voice, multispeakersteps], outputs=[audio], concurrency_limit=4)

with gr.Blocks() as clone:
    with gr.Row():
        with gr.Column(scale=1):
            clinp = gr.Textbox(label="Text",
                               info="What would you like StrawberryTTS to read?",
                               interactive=True)
            clvoice = gr.Audio(label="Voice", interactive=True, type='filepath', max_length=300)
            vcsteps = gr.Slider(minimum=5, maximum=20, value=20, step=1, label="Diffusion Steps",
                                info="Higher = better quality, but slower", interactive=True)
        with gr.Column(scale=1):
            clbtn = gr.Button("Synthesize", variant="primary")
            claudio = gr.Audio(interactive=False, label="Synthesized Audio")
            clbtn.click(clsynthesize, inputs=[clinp, clvoice, vcsteps], outputs=[claudio], concurrency_limit=4)

with gr.Blocks() as longText:
    with gr.Row():
        with gr.Column(scale=1):
            lnginp = gr.Textbox(label="Text",
                                info="What would you like StrawberryTTS to read?",
                                interactive=True)
            lngvoice = gr.Dropdown(voicelist, label="Voice", info="Select a default voice.", value='m-us-1',
                                   interactive=True)
            lngsteps = gr.Slider(minimum=5, maximum=25, value=10, step=1, label="Diffusion Steps",
                                 info="Higher = better quality, but slower", interactive=True)
        with gr.Column(scale=1):
            lngbtn = gr.Button("Synthesize", variant="primary")
            lngaudio = gr.Audio(interactive=False, label="Synthesized Audio")
            lngbtn.click(longsynthesize, inputs=[lnginp, lngvoice, lngsteps], outputs=[lngaudio],
                         concurrency_limit=4)

with gr.Blocks() as ljlongText:
    with gr.Row():
        with gr.Column(scale=1):
            ljlnginp = gr.Textbox(label="Text",
                                  info="What would you like StrawberryTTS to read?",
                                  interactive=True)
        with gr.Column(scale=1):
            ljlngbtn = gr.Button("Synthesize", variant="primary")
            ljlngaudio = gr.Audio(interactive=False, label="Synthesized Audio")
            ljlngbtn.click(ljlongsynthesize, inputs=[ljlnginp], outputs=[ljlngaudio],
                           concurrency_limit=4)

with gr.Blocks() as lj:
    with gr.Row():
        with gr.Column(scale=1):
            ljinp = gr.Textbox(label="Text",
                               info="What would you like StrawberryTTS to read?",
                               interactive=True)
        with gr.Column(scale=1):
            ljbtn = gr.Button("Synthesize", variant="primary")
            ljaudio = gr.Audio(interactive=False, label="Synthesized Audio")
            ljbtn.click(ljsynthesize, inputs=[ljinp], outputs=[ljaudio], concurrency_limit=4)

with gr.Blocks(title="StrawberryTTS", css="footer{display:none !important}", theme=theme) as web:

    gr.Markdown("""
    # StrawberryTTS
    TTS based on [StyleTTS2](https://github.com/yl4579/StyleTTS2)
    ## Limitations
    - Limited character amount in non-long-text modes
    - Mispronunciations and general pausing in the middle of sentences e.g. "1950s" -> "nineteen (long pause) fifty (short pause) s" and "post-war" -> "post (long pause) war"
    """)

    gr.TabbedInterface([libritts, clone, lj, longText, ljlongText],
                       ['Multi-Voice', 'Voice Cloning', 'LJSpeech', 'Long Text', 'LJSpeech Long Text [Experimental]'])
    # gr.TabbedInterface([libritts, clone, lj], ['Multi-Voice', 'Voice Cloning', 'LJSpeech'])

    gr.Markdown("""
    ## Come Join The Discord
    """)

    gr.HTML("""
    <iframe src="https://discord.com/widget?id=1159260121998827560&theme=dark" width="350" height="500" allowtransparency="true" frameborder="0"></iframe>
    """)

    gr.Markdown("""
    ## Credits
    - [SleepyYui](https://github.com/sleepyyui)
    - [StyleTTS2 Contributors](https://github.com/yl4579/StyleTTS2)
    - [fakerybakery](https://github.com/fakerybakery)
    """)

if __name__ == "__main__":
    web.queue(api_open=False, max_size=15).launch(show_api=False)
