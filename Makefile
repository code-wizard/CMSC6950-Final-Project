report: report.tex GenDataCompare4.png 'Iris Data Classification Accuracies.png' IrisData_noise_Results.png 'Sample Data Classification Accuracies.png'
	latexmk  -pdf -c

GenDataCompare4.png: data/iris.csv main.py data/matlab_Data.mat
	python main.py

'Iris Data Classification Accuracies.png': data/iris.csv main.py data/matlab_Data.mat
	python main.py

'Sample Data Classification Accuracies.png': data/iris.csv main.py data/matlab_Data.mat
	python main.py

IrisData_noise_Results.png: data/iris.csv main.py data/matlab_Data.mat
	python main.py

data/iris.csv:
	cd data && wget https://owlya.s3.us-east-2.amazonaws.com/iris.csv

data/matlab_Data.mat:
	cd data && wget https://owlya.s3.us-east-2.amazonaws.com/matlab_Data.mat

.PHONY: clean almost_clean

clean: almost_clean
	rm report.pdf
	rm *.png

almost_clean:
	latexmk -c