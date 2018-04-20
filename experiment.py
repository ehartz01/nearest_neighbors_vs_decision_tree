#Ethan Hartzell Machine Learning
#3.1 run the default algorithms and plot accuracy
#note: you must be in bash and run this: export CLASSPATH=/r/aiml/ml-software/weka-3-6-11/weka.jar:$CLASSPATH
import glob
import os, sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
#find the data files
data = glob.glob("data/*.arff")

#a function to fine matching training and test sets based on number of features in filename
def match_sets(tset,num):
        train = ""
        test = ""
        for file in data:
                if "Train" in file:
                        if str(num) in file:
                                train = file
                if "Test" in file:
                        if str(num) in file:
                                test = file
        return [train,test]

#this function takes the algorithm command and the number of features
#and it returns the accuracy in a tuple with the number of features 
def experiment(num,algo):
        train = ""
        test = ""
        tset = [train,test]
        tset = match_sets(tset,num)
        #calls weka and runs the specified data files
        tmp = os.popen(algo + "-t " + tset[0] + " -T " + tset[1]).read().split()
        td = False #just a bool to help keep track of where in the output we are
        percent = ""
        #reads the output and finds the accuracy on the test set
        for i,word in enumerate(tmp):
                if word == "test":
                        td = True
                if td:
                        if word == "Correctly":
                                percent = tmp[i+4]
        return (num,float(percent))
#does the same as above but with the training and test sets already specified for 3.2
def exp(algo,tset):
        #calls weka and runs the specified data files
        tmp = os.popen(algo + "-t " + tset[0] + " -T " + tset[1]).read().split()
        td = False #just a bool to help keep track of where in the output we are
        percent = ""
        #reads the output and finds the accuracy on the test set
        for i,word in enumerate(tmp):
                if word == "test":
                        td = True
                if td:
                        if word == "Correctly":
                                percent = tmp[i+4]
        #detect the number on the end of the file name (won't be more than three digits)
        num = 0
        if tset[0][-3] != '_':
                num = int(tset[0][-3] + tset[0][-2] + tset[0][-1])
        else:
                num = int(tset[0][-2]tset[0][-1])
        return (num,float(percent))

#here we run the experiment of 3.1 on each condition (# of features) and record the results
results = []
#first we use a decision tree algorithm
algo = "java weka.classifiers.trees.J48 "
results.append(experiment(14,algo))
results.append(experiment(24,algo))
results.append(experiment(34,algo))
results.append(experiment(44,algo))
results.append(experiment(54,algo))
results.append(experiment(64,algo))
results.append(experiment(74,algo))
results.append(experiment(84,algo))
results.append(experiment(94,algo))
print results
n_results = []
#next we use the nearest neighbors algorithm
algo = "java weka.classifiers.lazy.IBk "
n_results.append(experiment(14,algo))
n_results.append(experiment(24,algo))
n_results.append(experiment(34,algo))
n_results.append(experiment(44,algo))
n_results.append(experiment(54,algo))
n_results.append(experiment(64,algo))
n_results.append(experiment(74,algo))
n_results.append(experiment(84,algo))
n_results.append(experiment(94,algo))

print n_results
#we convert the results to a form amenable to matplotlib
x = []
y1= []
print results[0]
print results[1]
for i,j in results:
        x.append(i)
        y1.append(j)
print x
print y1
y2 = []
for i,j in n_results:
        y2.append(j)
#we plot the results, they pop up in a separate window. this graph is also saved in the zip file
plt.plot(x,y1,"r", x,y2)
plt.title("IBk (blue) vs. J48 (red)")
plt.xlabel("number of features")
plt.ylabel("accuracy out of 100%")
plt.show()

#here we begin 3.2 

#this class can read an arff file and have its data manipulated and then have another rewritten
class arff_file:
        def __init__(filename):
                self.relation = ""
                self.attributes = []
                self.data = []
                self.file = filename
                f = open(filename)
                temp = f.readlines()
                data = False
                #store the relevent parts of the file
                for line in tmp:
                        for word in line.split():
                                if word == "@Relation":
                                        self.relation = tmp[i+1]
                                if word == "@Attribute":
                                        self.attributes.append((tmp[i+1],tmp[i+2]))
                                if word == "@Data":
                                        data = True
                        if data:
                                self.data.append(line):
                tmp.close()

        def write_file(new_file_name):
                #write a new arff file based on whats recorded in the class
                f = open(new_file_name)
                f.write("@Relation " + self.relation
                f.write("\n")
                for at in self.attributes:
                        f.write("@Attribute " + at[0] + " " + at[1])
                f.write("\n")
                f.write("@Data")
                for example in self.data:
                        f.write(example)
                f.close()
        def get_data():
                return self.data
        def replace_data(nd):
                self.data = nd

train14 = arff_file("data/EEGTrainingData_14.arff")
d = train14.get_data()
import random
#randomizes the data and runs a trial on each size for both algorithms and returns the accuracies
def rando_trial():
        random.shuffle(d)
        for i in range(50,550,50):
                new_arr = []
                for count, line in d:
                        if count < i:
                                new_arr.append(line)
                train14.replace_data(new_arr)
                nfn = "data/EEGTrainingData_14_"
                train14.write_file(nfn + str(i) +".arff")
        training_files = glob.glob("*.arff")
        test_file = "data/EEGTestingData_14.arff"
        t14_j48_res=[]
        t14_ibk_res=[]
        for file in training_files:
                algo = "java weka.classifiers.lazy.IBk "
                t14_ibk_res.append(exp(algo,(file,test_file)))
                algo = "java weka.classifiers.trees.J48 "
                t14_j48_res.append(exp(algo,(file,test_file)))
        return (t14_j48_res,t14_ibk_res)
t14_j48_avg = [] #holds the means
t14_ibk_avg = [] #holds the means
j48_scores = [] #holds all the trial data
ibk_scores = [] 
#run 10 trials with randomized data
for k in range(10):
        if t14_j48_avg == [] and t14_ibk_avg == []:
                j = rando_trial()
                t14_j48_avg = j[0]
                t14_ibk_avg = j[1]
                j48_scores.append(j[0])
                ibk_scores.append(j[1])
        else:
                tup = rando_trial
                for count, score in enumerate(tup[0]):
                        t14_j48_avg[count] += score
                for count, score in enumerate(tup[1]):
                        t14_ibk_avg[count] += score
#finish calculating the means
for i in t14_j48_avg:
        i = i/10.0
for i in t14_ibk_avg:
        i = i/10.0
t14_j48_sd = [] #holds the standard deviations
t14_ibk_sd = []
#calculate the sd
for i in j48_scores:
        for count,j in enumerate(i):
                t14_j48_sd[count] += (j-t14_j48_avg[count])**2
for i in t14_j48_sd:
        i = i/10.0
for i in ibk_scores:
        for count,j in enumerate(i):
                t14_ibk_sd[count] += (j-t14_ibk_avg[count])**2
for i in t14_ibk_sd:
        i = i/10
print t14_j48_sd
#now lets do it all again for the 54 feature data
train54 = arff_file("data/EEGTrainingData_54.arff")
os.system("rm *arff")
d = train54.get_data()
#randomizes the data and runs a trial on each size for both algorithms and returns the accuracies
def rando_trial():
        random.shuffle(d)
        for i in range(50,550,50):
                new_arr = []
                for count, line in d:
                        if count < i:
                                new_arr.append(line)
                train54.replace_data(new_arr)
                nfn = "data/EEGTrainingData_54_"
                train54.write_file(nfn + str(i) +".arff")
        training_files = glob.glob("*.arff")
        test_file = "data/EEGTestingData_54.arff"
        t14_j48_res=[]
        t14_ibk_res=[]
        print training_files
        training_files.sort()
        print training_files
        for file in training_files:
                algo = "java weka.classifiers.lazy.IBk "
                t14_ibk_res.append(exp(algo,(file,test_file)))
                algo = "java weka.classifiers.trees.J48 "
                t14_j48_res.append(exp(algo,(file,test_file)))
        return (t14_j48_res,t14_ibk_res)
t14_j48_avg = [] #holds the means
t14_ibk_avg = [] #holds the means
j48_scores = [] #holds all the trial data
ibk_scores = [] 
#run 10 trials with randomized data
for k in range(10):
        if t14_j48_avg == [] and t14_ibk_avg == []:
                j = rando_trial()
                t14_j48_avg = j[0]
                t14_ibk_avg = j[1]
                j48_scores.append(j[0])
                ibk_scores.append(j[1])
        else:
                tup = rando_trial
                for count, score in enumerate(tup[0]):
                        t14_j48_avg[count] += score
                for count, score in enumerate(tup[1]):
                        t14_ibk_avg[count] += score
#finish calculating the means
for i in t14_j48_avg:
        i = i/10.0
for i in t14_ibk_avg:
        i = i/10.0
t14_j48_sd = [] #holds the standard deviations
t14_ibk_sd = []
#calculate the sd
for i in j48_scores:
        for count,j in enumerate(i):
                t14_j48_sd[count] += (j-t14_j48_avg[count])**2
for i in t14_j48_sd:
        i = i/10.0
for i in ibk_scores:
        for count,j in enumerate(i):
                t14_ibk_sd[count] += (j-t14_ibk_avg[count])**2
for i in t14_ibk_sd:
        i = i/10
