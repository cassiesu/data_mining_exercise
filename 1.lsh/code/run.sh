python mapper_template.py < ../data/training.txt | sort | python reducer_template.py | sort > ../data/predict.txt
python check.py ../data/predict.txt ../data/duplicates.txt