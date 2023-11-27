---
title: "The AI Coder: Building a website using AI"
author: Alvaro
date: 2023-11-25 10:33:00 +0800
categories: [Programming, AI, Development]
tags: [programming, python, gpt, coding, ai]
math: false
mermaid: false
pin: true
image:
   path: https://i.imgur.com/WZhtICw.jpg
   width: 600
   height: 600
---

Last week I decided to tackle a quite silly side-project I left stopped about 2 years ago: building my own (guitar) chord / tab site so I can store all my music sheets.

The main motivation for doing this is that I got tired of using [UltimateGuitar](https://en.wikipedia.org/wiki/Ultimate_Guitar), from which I've been being a user since around 2013. Back then it was vey handy to have a site where I could store all my guitar tabs, search for other people's and have a rating on whether each one of them was good or not. They also had a nice phone app, Tabs Pro, which I loved, but got removed a few years ago (or deprecated, really) in favor of their current one, "Tabs".

My main issue with it is that their current business goals have become abusive and made them to add lots of things that, IMO, completely ruin the experience. I could write a lot more about this, but here are some points: 

- Their latest price for the Pro version is $10 / month, which, as an amateur guitar player, is not worthy for me.
- Their mobile app is very bad. It randomly crashes, is very bloated with things I'll never use (like its TikTok clone for sharing clips of yourself playing).
- You must pass a nonsense review process for every tab/sheet you upload, even if it's only for yourself, which forces you to rewrite (tag for them, really) it in their own format
- They claim they pay fees and licenses for all their tabs on their site, but it's easy to find content scrapped from other smaller sites without even adding a reference.

The thing is that, at first it was bearable. For example, they had a nice program that removed the ads for a month if you shared a tab (but it was discontinued). 

Even with all the inconveniences, and no similar alternatives, I paid the "Pro" tier for a few months in the hope those issues got fixed. But they didn't. 

## Starting the adventure: Getting the data

During all my time using UltimateGuitar I stored around 500 tabs there, which are efectively blocked unless I pay the Premium subscription (meaning that I can only access them using their web/app). Some time ago, though, while it was allowed and when I was a subscriptor, I downloaded a fraction of them and stored in a backup. However, there are plenty I dont't have. 

So I had no other option than to use shady techniques to retrieve them back, which I'm not covering here.


![](https://media.tenor.com/CLErADjseZMAAAAC/johnny-depp-captain-jack-sparrow.gif)


The important thing is that I could retrieve most of them (yay!). I'll try to get the remaining ones in the future, though.

![](assets/img/posts/aidevelopment/lstabs.png)

Each tab has been downloaded alongside its metadata (votes, stars, chord variations, etc.), which I doubt I'll ever use. I just want the plain ASCII content of the tabs.

Alghough JSON format isn't too bad, it's probably better to store them in some sort of database so the site/app is easier to maintain. `SQLite` is quite nice for what I want: privately storing and reading my tabs, not building a wide scale product (if I wanted to do that in the future, migrating from SQLite to something like PosgreSQL would be easy).

So, I asked ChatGPT for, given a set of JSON files, inserting them into a SQLite table (I gave it the structure of the table) and it gave me this:

```python
#... reading the JSONs...

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

insert_query = "INSERT INTO sheets (id, songname, artistname, artistid, tuning, capo, rating, chords, votes, content) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

values_to_insert = [
    (t['id'], t['songname'], t['artistname'], t['artistid'], t['tuning'], t['capo'], t['rating'], t['chords'], t['votes'])
    for t in filtered_tabs
]

cursor.executemany(insert_query, values_to_insert)

conn.commit()
cursor.close()
conn.close()
```

*(It still amazes me how it "understands" that, for inserting in bulk it has to use the `excutemany` command instead of a regular `execute` inside a `for` loop)*

After executing the above, I ended up with an `.sqlite` file will all my songs 🥳

![](assets/img/posts/aidevelopment/sqlite.png)


## Builidng the backend

The backend is very straightforward. Just a Flask app with a few endpoints to retrieve the data that's inside the above described `sqlite` (just returning the songs in a JSON format, nothing fancy), and two endpoints which render the list of all the available sheets and each one of the tabs via Flask (Jinja) templates.

These "templates", in case you don't know, are just regular HTML files in which you can inject Python variables, for example, the lyrics of a song, its chords, etc. The HTMLs are serverside built and then a rendered version is returned to the client (browser).

I could write more about them, but that's not my goal with this post, so I'll refer you to this site, in which they're futher explained: [https://flask.palletsprojects.com/en/2.3.x/tutorial/templates/](https://flask.palletsprojects.com/en/2.3.x/tutorial/templates/).

The main issue here is that my frontend development skills are mediocre, so I sought some help here.


## AI to the rescue: Creating the frontend

We're entering in the part that sparked the idea of doing this. The other day I was messing around on Twitter and found some guy sharing this tool called [`screenshot-to-code`](https://github.com/abi/screenshot-to-code). Here is a quick video of what it can do:


<center>
<video width="640" height="480" controls>
  <source src="https://private-user-images.githubusercontent.com/23818/283664580-503eb86a-356e-4dfc-926a-dabdb1ac7ba1.mp4" type="video/mp4">
</video>
</center>


It basically uses an screenshot of a webpage and the GPT4-Vision API to automatically build the HTML, CSS and JS code for the site. I got quite mindblown on how simple and how well it worked on the examples. So I had to try it for this chords site.

I began by cloning the repo and setting it up according to the instructions. It basically needs an OpenAI API token and (if you want it) Docker:

```
git clone git@github.com:abi/screenshot-to-code.git
cd screenshot-to-code
echo "OPENAI_API_KEY=sk-your-key" > .env
docker-compose up -d --build 
```
And I was good to go!

So, I thought: *what if I build a quick mockup of the website on something like [Figma](https://www.figma.com), take an screenshot and pass it to `screenshot-to-code`?* I did it.

![goodeyes](assets/img/posts/aidevelopment/figma.png)

*As you'll probably see, I won't be remembered as a exceptional designer. But I don't mind, I only need something simple (the more minimalist, the better).*

The next step was to pass the screenshot to the aforementioned tool. Here is a quick video of the result I got:

<center>
    <iframe width="640" height="400" src="https://www.youtube.com/embed/0TD9k1ICTaQ?si=rx1AMcHUdyRiaSd9" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</center>

It even generated a responsive (mobile) version of the web! 🤯

I then copied the code and adapted it using the aforementioned **Flask templates** (which consisted basically in putting a `%block%` in which I would pass the `content` of the tab):

![](assets/img/posts/aidevelopment/code1.png)

Just in about 10 minutes I covered the all frontend part and almost without coding! 


## Meet OpenChords
After connecting all the components I got to the first usable version. I had to add a little bit more of Javascript to the generated UI and also include the "templates" part for Flask to properly render dynamic versions depending on the selected tab/sheet, but I mostly used ChatGPT for doing it, just to force myself to the "AI assisted" development workflow.

The experience was quite nice. Not everything worked right out of the box, but I only had to make minor changes (sometimes GhatGPT even corrected itself!).

This "v1" version is very simple, but functionally, it's all what I wanted (and mostly all of what I used on UltimateGuitar). Take a look!

![](assets/img/posts/aidevelopment/openchords.png)

I also checked on my phone whether the sheets were displayed properly, and they did! I hadn't to make any special adjustement! GPT generated responsive code quite nicely!

![](assets/img/posts/aidevelopment/iphone.jpeg){: width="250" }

As for future work, some things I'll probably take a look at are:

1. Adding a way to post new sheets right from the site. Currently I'm adding them via a Python script, which isn't convenient
2. Creating an iOS app that takes the `.sqlite` db and gives the same funcionality. All locally.
3. Use GPT4-Vision to parse even older sheets I still have in PDF format.


As for now, the site serves me well as a repository, I can say it has been a personal success.

![](https://media.tenor.com/9qn_nI927TYAAAAM/ateam-plan.gif)