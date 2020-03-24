function download_20m {
	echo "Download ml-20m"
	curl -O http://files.grouplens.org/datasets/movielens/ml-20m.zip
	mkdir ml-20m
	mv ml-20m.zip ml-20m/
}

function download_1m {
	echo "Downloading ml-1m"
	curl -O http://files.grouplens.org/datasets/movielens/ml-1m.zip
	mkdir ml-1m
	mv ml-1m.zip ml-1m/
}

if [[ $1 == "ml-1m" ]]
then
	download_1m
else
	download_20m
fi
