mkdir datasets
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -O ./datasets/Flickr8k_Dataset.zip
unzip datasets/Flickr8k_Dataset.zip -d ./datasets/
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -O ./datasets/Flickr8k_text.zip
unzip datasets/Flickr8k_text.zip -d ./datasets/
rm datasets/Flickr8k_text.zip datasets/Flickr8k_Dataset.zip
