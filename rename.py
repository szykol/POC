import os

os.chdir('img/')

for i in range(1, 16):
    os.rename(f'Obraz ({i+2}).jpg', f'monety{i}.jpg')