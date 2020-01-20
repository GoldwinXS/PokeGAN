import requests
import bs4,os

target_url = 'https://pokemondb.net/sprites/'
save_folder = 'pokemon/'

# request html from main sprite page
request = requests.get(target_url,'GET')
soup = bs4.BeautifulSoup(request.content,'html.parser')

# get all relevant links for pokemon sprites
infocard_class = 'infocard'
infocards = soup.find_all(class_=infocard_class)
pokemon_links = []
for card in infocards:
    pokemon_links.append(target_url+card.text.lstrip(' '))

# one link per pokemon! If you want only the OG 150, just slice the list for that
pokemon_links = pokemon_links

# these are all of the classes I want to take. This could be a for loop, but I decided to list them all to maintain control
img_classes = ['img-fixed img-sprite-v1',
               'img-fixed img-sprite-v2',
               'img-fixed img-sprite-v3',
               'img-fixed img-sprite-v4',
               'img-fixed img-sprite-v5',
               'img-fixed img-sprite-v6',
               'img-fixed img-sprite-v7',
               'img-fixed img-sprite-v8',
               'img-fixed img-sprite-v9',
               'img-fixed img-sprite-v10',
               'img-fixed img-sprite-v11',
               'img-fixed img-sprite-v12',]


img_count = len(os.listdir(save_folder)) # this is just in case you're resuming the script later

# now for the downloading!
for pokemon_link in pokemon_links:
    request = requests.get(pokemon_link,'GET')
    soup = bs4.BeautifulSoup(request.content,'html.parser')
    img_links = []


    print('Saving images for: ',pokemon_link.lstrip(target_url))

    try:


        srcs = [tag['src'] for tag in soup.find_all('img')]

        for src in srcs:
            with open(save_folder+str(img_count)+'.png','wb') as handle:
                img_count+=1
                resp = requests.get(src)
                handle.write(resp.content)


    except IndexError:
        print('IndexError encountered... ignoring... ')


