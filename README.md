Breast Cancer Neural Network in .NET
====================================

This project is an example of how to implement a neural network (using back-propagation) in C# (.NET). It is based on James McCaffrey's demo at Build 2014 in San Francisco but, rather than using the Iris sample data set, it uses the Diagnostic Wisconsin Breast Cancer Database.

Compared to McCaffrey's original code, this example is a little tidier and easier to navigate in.

You can find the original source code here: http://quaetrix.com/Build2014.html

Sample Data Source
==================
The sample data used in this example comes the University of Wisconsin. The features in the data set are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

In this example, the data set has been slightly re-formatted. The CSV file in the project does not include the ID number of each observation. Moreover, the original class variable has been re-encoded in the following manner:

	2 => 1,0
	4 => 0,1

The original data set can be found here: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

a) Creators: 

	Dr. William H. Wolberg, General Surgery Dept., University of
	Wisconsin,  Clinical Sciences Center, Madison, WI 53792
	wolberg@eagle.surgery.wisc.edu

	W. Nick Street, Computer Sciences Dept., University of
	Wisconsin, 1210 West Dayton St., Madison, WI 53706
	street@cs.wisc.edu  608-262-6619

	Olvi L. Mangasarian, Computer Sciences Dept., University of
	Wisconsin, 1210 West Dayton St., Madison, WI 53706
	olvi@cs.wisc.edu 

b) Donor: Nick Street

c) Date: November 1995
