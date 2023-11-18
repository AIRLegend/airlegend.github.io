---
title: "How I generated professional headshots for my LinkedIn profile"
author: Alvaro
date: 2023-11-18 10:33:00 +0800
categories: [ML, Deep Learning]
tags: [maths, programming, python, pytorch]
math: true
mermaid: false
pin: true
image:
   path: https://i.imgur.com/tChisdZ.png
   width: 200
   height: 200
---

The other day I was bored, wandering through LinkedIn and realized that I probably needed some new profile pics. You know, the ones very corporate-guys share on their posts. Something cool with a suit and some office-styled background. So I thought: What if I finetune an image generation model on myself and use it to generate them? 


## How can I generate photos of myself?

Just to begin, there are countless of image generation models, being the most famous Midjourney, DALL-e and Stable Diffusion – being the latter the only one which is public – and we can prompt them to generate custom images! ([see some examples on the Lexica browser](https://lexica.art/?q=d8fce142-23e4-4bd9-9ae1-6e32db4d3ddd))

Last July, Stability.ai (the company behind the Stable Diffusion models) released it's best iteration of their model: **Stable Diffusion XL 1.0** (SDXL from now on), which is a step forward in quality from its previous "gold-standard model", Stable Diffusion 1.5. 

So, it would be nice if I could finetune it on several of my photos and use it for generating me in whichever type of pic I wanted.

But there are a few problems: 

1. I don't currently have access to my GPU machine (or any cloud GPU)
2. I don't want to spend much on this endeavor
3. I don't have too much time to train a "complex" model on my old Macbook's CPU.

The ideal solution, then, is to finetune this model on [Google Colab](https://colab.google/) – which is the hardest part–, download the weights and run inference anywhere with the prompt I want.


## Finetuning SDXL 1.0

The first thing we need with any ML model is to gather data. Regardless, with a model such big and complex as SDXL one could expect hundreds or thousands of images were required, and it wouldn't be wrong. Yet, there are techniques for easing this requirement, being one of them "[Dreambooth](https://arxiv.org/abs/2208.12242)", which lets anyone finetune their diffusion model using only approximately 5 images (obviously, the more images and the more diverse they are, the better results one will get).

### Gathering the data

I quickly ran through my phone's gallery just to collect about 12 photos of myself. Trying to get diverse poses, angles, lightning and backgrounds. I've probably could have tried with more pics, but I got more-or-less good results with this amount.

![training data](assets/img/posts/dreambooth/trainingdata.png)

The next step is to resize them to a common aspect ratio, usually squared. For this I used a generic free photo resizer tool I found on Google, called [BIRME](https://www.birme.net/?target_width=1024&target_height=1024) (feel free to use if you want, it's pretty nice). I personally found better generation results using a size of 1024x1024 pixels, although 512x512 wasn't too far behind for me and also requires fewer memory (and maybe also faster).

Once done, download the .zip with all the resized photos.


### Finetuning SDXL on Google Colab

I've made a Google Colab notebook with all the training code I wrote. 

<center>
    <a target="_blank" href="https://colab.research.google.com/github/AIRLegend/notebook-examples/blob/master/dreambooth_lora/GenerateImagesOfYourself.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</center>

I used Huggingface's [Advanced-Autotrain](https://github.com/huggingface/autotrain-advanced/tree/main) package, which comes with a very handy script for finetuning SDXL (or any model really)  with only one command: `autotrain dreambooth`. On top of that, it handles a critical thing for being able to do this on the low memory's T4 GPU that Colab offers: [using LoRA, which I cover on a previous post](https://airlegend.github.io/posts/lora/).

The notebook is meant to be self explanatory, but basically, you put your images on the `img/` folder, run the training and download the `safetensors` file, which contains the LoRA weights of your tuned model. After that, you can run inference with them wherever you want (although the notebook contains a section for doing it). 

SDXL comes with a "Refiner" model which is meant to "fix" inconsistencies the base model made (for example on eyes, hands, text, etc.). However, while on Google Colab, this model cannot be held into memory simultaneously with the base one. We must save the image, restart the session and run the refiner separately, which it's a bummer.

If you have Colab Pro, or access to more memory, I encourage you to run both steps. However, on my experience, only with the first part, good results can be attained.

I should point out, though, that Colab isn't reliable for long train runs and could kill your session even though you don't reach the maximum allowed GPU memory consumption. Use the `--checkpointing_steps` parameters accordingly to save a checkpoints of your progress.


### Results

After having the model trained one can use the "**inference section**" of the same notebook to produce new images given a prompt. I've left some placeholder prompt for generating them, but feel free to explore your own! The prompt engineering part is very important (and pretty much alchemy!), having even more impact on the quality than the finetuning step. I'm pretty much a newbie on it, but I learnt quite a few tricks from this nice guide: "[Stable Diffusion prompt: a definitive guide](https://stable-diffusion-art.com/prompt-guide/)". I strongly encourage you check it out!

Here are some examples of the generations I got with 1k steps of finetuning and the inference parameters on the notebook (The inference time for each image you generate is about 1 minute).

![badeyes](assets/img/posts/dreambooth/gridexamples.png){: width="512" }



### BONUS: Improving face quality

After generating photos, sometimes things as the the eyes are not perfect. They can contain artifacts that make obvious the photos are artificially generated. Even running the refiner model those could stay there, which is kind of a bummer, because the rest of the image can look pretty legit! See an example of what I mean:

![badeyes](assets/img/posts/dreambooth/badeyes1.png){: width="250" }


After investigating this issue I came with a technique (that luckily comes with code) called [CodeFormer](https://github.com/sczhou/CodeFormer), which was primarily meant to restore images of faces, but apparently is also commonly used among "generation artists" to fix the faces of their generations.

You can install it via the instructions on their README, it's quite straightforward. After doing it, you can  fix the faces of your generated images via the following command (replace the `--input-path` with one to your own image!)

```
python inference_codeformer.py -w 0.5 --input_path /Users/air/Downloads/generated_image-3.png
```

As you probably noticed in the above command, there is the `-w` parameter thing set to `0.5`. This an specific parameter for the method that controls the "strength" of the effect. With heavier values (near 1) the model will "overwrite" more of the original image and we'll lose some of details of SDXL (your skin will be softer, you'll have less wrinkles, freckles, scars, etc) and the image will lose quality. On the other hand, with low values (near 0) the artifacts couldn't be removed. So, in my tests, most of the time, values around `0.5` were pretty much okay! But depending on each particular generation you'd need to increase it!

After running it, the fixed image will be saved by default on the same directory as your `CodeFormer` installation (inside the `results` folder).

The following image shows the fixed version of the above. Pretty cool, uh?

![goodeyes](assets/img/posts/dreambooth/goodeyes1.png){: width="250" }


## Closing up

I know there are more user friendly ways of doing the same as I shared on this post. For example: `Kohya`,  which is a tool that adds an abstraction layer from all the training  code. You can take a look at this [Kohya tutorial](https://www.youtube.com/watch?v=TpuDOsuKIBo) in case you're interested. However, as a "code guy" myself, I find my way more straightforward to follow this tutorial as long as one is familiar with Jupyter notebooks and some Python!

It's very impressive how easy one can use state of the art techniques for something as foolish as generating photos of himself with this few amount of effort – free Google Colab, a pre-built training script, open sourced models and a few training photos – and get results of unthinkable quality just one or two years ago! 

There's only one question left: what we'll have two years from now?