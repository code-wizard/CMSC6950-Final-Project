report: report.tex classification_noise.png classification_score.png desc_plot.png training.png test.png
	latexmk  -pdf -c

classification_noise.png: data/iris.csv main.py
	python main.py

classification_score.png: data/iris.csv main.py
	python main.py

desc_plot.png: data/iris.csv main.py
	python main.py

training.png: data/iris.csv main.py
	python main.py

test.png: data/iris.csv main.py
	python main.py

data/iris.csv:
	cd data && wget https://owlya.s3.us-east-2.amazonaws.com/iris.csv


.PHONY: clean almost_clean

clean: almost_clean
	rm report.pdf
	rm *.png

almost_clean:
	latexmk -c