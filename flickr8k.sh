mkdir datasets
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
mv Flickr8k_Dataset.zip datasets/Flickr8k_Datasets.zip
unzip datasets/Flickr8k_Dataset.zip -d ./datasets/
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
mv Flickr8k_text.zip datasets/Flickr8k_text.zip
unzip datasets/Flickr8k_text.zip -d ./datasets/
unzip ./datasets/Flickr8k_Datasets.zip -d ./datasets/
rm datasets/Flickr8k_text.zip datasets/Flickr8k_Datasets.zip
