# cat ../train/train.txt | python mapper_knn.py | python reducer.py  > centers.txt
# python evaluate.py centers.txt ../train/train.txt

cat ../train/train.txt | python mapper.py | python reducer.py  > centers.txt
python evaluate.py centers.txt ../train/train.txt